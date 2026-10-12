"""Physics-inspired simulator for an industrial centrifugal pump.

Produces the sensor streams Harmonic consumes — triaxial-ish vibration at
25.6 kHz plus slow channels at 1 Hz — for a healthy machine and for four
fault modes, with ground truth attached.

The model is deliberately *signature-accurate* rather than physically
complete: each fault is built from the features a vibration analyst would
actually look for, because those are the features the engine encodes.

    sim = PumpSimulator(seed=7)
    sim.inject("bearing_wear", severity=1.0, ramp_hours=8)
    frame = sim.step(1.0)          # one second of data

Everything is deterministic given the seed: the same seed and the same call
sequence produce bit-identical output, which is what makes a scenario
reproducible and a regression meaningful.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np

# --- Machine constants -------------------------------------------------------
# Pump P-101: 15 kW class centrifugal pump, rigid mount, 6308-style deep groove
# ball bearing. These mirror the reference profile mint_brain writes into the
# brain header; the simulator is the other half of that contract.

NOMINAL_RPM = 2950.0
BEARING_PITCH_DIA_MM = 65.0
BEARING_BALL_DIA_MM = 15.0
BEARING_N_BALLS = 8
BEARING_CONTACT_ANGLE_RAD = 0.0

VANE_COUNT = 6  # impeller vanes -> blade pass frequency

FS_VIB_DEFAULT = 25600.0
FS_SLOW_DEFAULT = 1.0

# Healthy baselines, in engineering units.
HEALTHY_1X_MM_S = 0.55
HEALTHY_2X_MM_S = 0.18
HEALTHY_3X_MM_S = 0.09
HEALTHY_BPF_MM_S = 0.12
HEALTHY_NOISE_MM_S = 0.22

AMBIENT_C = 22.0
BEARING_TEMP_C = 48.0
MOTOR_TEMP_C = 61.0
DISCHARGE_BAR = 6.2
SUCTION_BAR = 1.1
FLOW_M3H = 84.0
CURRENT_A = 26.5

FAULT_MODES = ("bearing_wear", "misalignment", "cavitation", "seal_leak")

# Bearing defects at the other two locations (HARM-S9-T15). They are kept OUT of
# FAULT_MODES on purpose: the production brain, its training frames and every
# consumer of FAULT_MODES stay exactly as they were. `bearing_wear` is the
# outer race. A candidate brain that names the location trains on all six.
LOCATION_FAULT_MODES = ("bearing_inner_race", "bearing_rolling_element")
ALL_FAULT_MODES = FAULT_MODES + LOCATION_FAULT_MODES

_FAULT_PROFILES = None


def location_fault_physics(mode: str) -> dict[str, float]:
    """The signal parameters of a location fault, from sim/fault_profiles.yaml."""
    global _FAULT_PROFILES
    if _FAULT_PROFILES is None:
        from pathlib import Path

        import yaml

        path = Path(__file__).with_name("fault_profiles.yaml")
        _FAULT_PROFILES = yaml.safe_load(path.read_text(encoding="utf-8"))["faults"]
    return {k: float(v) for k, v in _FAULT_PROFILES[mode]["physics"].items()}


def mode_seed(base: int, mode: str, variant: str = "") -> int:
    """A reproducible per-fault seed.

    Python randomises str.__hash__ per process unless PYTHONHASHSEED is fixed,
    so `base + hash(mode)` silently produces different data on every run. That
    quietly broke the determinism this project depends on: two runs of the same
    command generated different machines, and a benchmark could pass once and
    fail the next time for no visible reason.
    """
    offset = ALL_FAULT_MODES.index(mode) * 101 if mode in ALL_FAULT_MODES else 0
    variant_offset = sum(ord(c) for c in variant) * 7
    return base + offset + variant_offset


def bearing_frequencies(rpm: float) -> dict[str, float]:
    """Bearing defect frequencies in Hz for a given shaft speed.

    Standard rolling-element kinematics: the rolling elements pass a fixed
    defect at a rate set by the bearing's geometry and the shaft rate.
    """
    fr = rpm / 60.0
    ratio = (BEARING_BALL_DIA_MM / BEARING_PITCH_DIA_MM) * math.cos(
        BEARING_CONTACT_ANGLE_RAD
    )
    n = BEARING_N_BALLS
    return {
        "shaft": fr,
        "bpfo": (n / 2.0) * (1.0 - ratio) * fr,
        "bpfi": (n / 2.0) * (1.0 + ratio) * fr,
        "bsf": (BEARING_PITCH_DIA_MM / (2.0 * BEARING_BALL_DIA_MM))
        * (1.0 - ratio * ratio)
        * fr,
        "ftf": 0.5 * (1.0 - ratio) * fr,
    }


@dataclass
class Fault:
    """One injected fault.

    `severity` is the level the fault ramps *to*, on 0..1. With `ramp_hours`
    it rises linearly from zero over that period of machine time; with
    `ramp_hours=0` it applies at full severity from `onset_s`.
    """

    mode: str
    severity: float = 1.0
    onset_s: float = 0.0
    ramp_hours: float = 0.0

    def level(self, t_s: float) -> float:
        """Severity actually in effect at machine time `t_s`."""
        if t_s < self.onset_s:
            return 0.0
        if self.ramp_hours <= 0.0:
            return self.severity
        elapsed_h = (t_s - self.onset_s) / 3600.0
        return self.severity * min(1.0, elapsed_h / self.ramp_hours)


@dataclass
class Frame:
    """One window of sensor data, as the HAL would deliver it."""

    t_s: float          # machine time, which fault ramps follow
    wall_s: float       # signal time; equals t_s unless time is compressed
    vibration: np.ndarray  # mm/s, radial, fs_vib
    vibration_axial: np.ndarray  # mm/s, axial, fs_vib
    rpm: float
    bearing_temp_c: float
    motor_temp_c: float
    discharge_bar: float
    suction_bar: float
    flow_m3h: float
    current_a: float
    # Ground truth, for training labels and for scoring detection.
    severities: dict[str, float] = field(default_factory=dict)

    @property
    def label(self) -> str:
        """Dominant fault at this instant, or "healthy"."""
        active = {m: s for m, s in self.severities.items() if s > 0.05}
        if not active:
            return "healthy"
        return max(active, key=lambda m: active[m])

    @property
    def labels(self) -> list[str]:
        """All faults active at this instant. Faults can co-occur."""
        return sorted(m for m, s in self.severities.items() if s > 0.05)


class PumpSimulator:
    """Centrifugal pump with injectable fault modes.

    Vibration is synthesised as a sum of harmonic components plus noise, in
    mm/s velocity so it can be compared directly against ISO 10816 zones.
    Phase is carried across calls, so consecutive frames form one continuous
    signal rather than a sequence of unrelated windows.
    """

    def __init__(
        self,
        rpm: float = NOMINAL_RPM,
        fs_vib: float = FS_VIB_DEFAULT,
        fs_slow: float = FS_SLOW_DEFAULT,
        seed: int = 0,
        load: float = 1.0,
        time_scale: float = 1.0,
    ) -> None:
        self.rpm_nominal = rpm
        self.rpm = rpm
        self.fs_vib = fs_vib
        self.fs_slow = fs_slow
        self.load = load
        # Time compression for the demo: each step still SYNTHESISES `dt`
        # seconds of signal, but the machine's clock — and so every fault ramp
        # — advances by `dt * time_scale`. Degradation that really takes days
        # can then be watched in two minutes without faking the waveform.
        self.time_scale = time_scale
        self.rng = np.random.default_rng(seed)

        self.t_s = 0.0        # machine time (fault clock)
        self.wall_s = 0.0     # signal time actually synthesised
        self._phase = 0.0  # shaft phase, revolutions — keeps frames continuous
        self.faults: list[Fault] = []

    # --- fault control ---------------------------------------------------

    def inject(
        self,
        mode: str,
        severity: float = 1.0,
        onset_s: float = 0.0,
        ramp_hours: float = 0.0,
    ) -> Fault:
        """Schedule a fault. Returns the Fault so a caller can adjust it."""
        if mode not in ALL_FAULT_MODES:
            raise ValueError(f"unknown fault mode {mode!r}; known: {ALL_FAULT_MODES}")
        if not 0.0 <= severity <= 1.0:
            raise ValueError(f"severity must be in 0..1, got {severity}")
        fault = Fault(mode=mode, severity=severity, onset_s=onset_s, ramp_hours=ramp_hours)
        self.faults.append(fault)
        return fault

    def clear_faults(self) -> None:
        self.faults.clear()

    def severities(self) -> dict[str, float]:
        """Severity of each mode at the current time, summed if injected twice.

        The location modes appear only once one is injected, so a run of the
        original four reports exactly the keys it always did."""
        out = {m: 0.0 for m in FAULT_MODES}
        for f in self.faults:
            if f.mode in LOCATION_FAULT_MODES:
                out.update({m: 0.0 for m in LOCATION_FAULT_MODES})
                break
        for f in self.faults:
            out[f.mode] = min(1.0, out[f.mode] + f.level(self.t_s))
        return out

    # --- simulation ------------------------------------------------------

    def step(self, dt: float = 1.0) -> Frame:
        """Advance `dt` seconds of machine time and return the window."""
        if dt <= 0:
            raise ValueError("dt must be positive")
        n = round(self.fs_vib * dt)
        sev = self.severities()

        bearing = sev["bearing_wear"]
        misalign = sev["misalignment"]
        cavitation = sev["cavitation"]
        leak = sev["seal_leak"]

        # Shaft speed: small random walk, plus the slip a cavitating or
        # leaking pump shows as its hydraulic load changes.
        rpm_target = self.rpm_nominal * (1.0 - 0.004 * cavitation + 0.002 * leak)
        self.rpm += 0.05 * (rpm_target - self.rpm) + self.rng.normal(0.0, 0.6)
        fr = self.rpm / 60.0

        # Continuous phase across frames.
        rev = np.arange(n, dtype=np.float64) / self.fs_vib * fr + self._phase
        self._phase = float((self._phase + n / self.fs_vib * fr) % 1.0)
        two_pi = 2.0 * math.pi

        freqs = bearing_frequencies(self.rpm)

        # --- healthy structure: 1X, 2X, 3X, blade pass, broadband floor ---
        a1 = HEALTHY_1X_MM_S * (0.85 + 0.3 * self.load)
        a2 = HEALTHY_2X_MM_S
        a3 = HEALTHY_3X_MM_S

        # Misalignment is classically a 2X (and 3X) phenomenon, with a strong
        # axial component: that ratio is the signature, not the overall level.
        a2 += 2.6 * HEALTHY_1X_MM_S * misalign
        a3 += 1.1 * HEALTHY_1X_MM_S * misalign

        vib = (
            a1 * np.sin(two_pi * rev)
            + a2 * np.sin(two_pi * 2.0 * rev + 0.7)
            + a3 * np.sin(two_pi * 3.0 * rev + 1.9)
            + HEALTHY_BPF_MM_S * np.sin(two_pi * VANE_COUNT * rev + 0.3)
        )
        vib += self.rng.normal(0.0, HEALTHY_NOISE_MM_S, n)

        # --- bearing wear: impulses at BPFO, amplitude-modulated by 1X ------
        # A spall in the outer race is struck once per rolling-element pass.
        # The strikes ring the structure, so the signature is a train of
        # decaying impulses at BPFO — which is why envelope analysis finds it
        # long before the overall RMS moves.
        if bearing > 0.0:
            # Severity grows faster than linearly: a spall spreads, and each
            # increment of damage makes the next strike harder. The exponent
            # and scale are set so severity walks the machine cleanly through
            # the ISO zones — roughly 2.6 mm/s at a third severity (zone B),
            # 7.7 at two thirds (zone C) and 15 at full (zone D) — which is
            # what makes the ISO threshold calibration testable at all.
            amp = 70.0 * bearing**1.5
            vib += self._impulse_train(
                n, freqs["bpfo"], rev, amplitude=amp, damping=850.0
            )
            # Sidebands at +-1X from load-zone modulation.
            sb = 0.12 * amp
            vib += sb * np.sin(two_pi * (freqs["bpfo"] - fr) / fr * rev + 0.4)
            vib += sb * np.sin(two_pi * (freqs["bpfo"] + fr) / fr * rev + 2.1)
            # Late-stage wear raises the broadband floor as the race degrades.
            vib += self.rng.normal(0.0, 0.10 * amp * bearing, n)

        # --- cavitation: broadband, random, high-frequency weighted ---------
        # Imploding vapour bubbles are a random process, not a periodic one:
        # the tell is a raised, spiky noise floor, hence high kurtosis.
        if cavitation > 0.0:
            # Cavitation is a shot-noise process: many small bubble collapses,
            # each ringing the casing. Its signature is impulsiveness
            # (kurtosis, crest) far more than loudness — most of the energy
            # sits above the 10-1000 Hz band ISO 10816 measures, which is
            # exactly why overall RMS is a poor cavitation detector and why
            # the feature set carries kurtosis and crest alongside it.
            noise = self.rng.normal(0.0, 1.0, n)
            # The floor only becomes significant at high severity, so the
            # impulsive character survives at the low severities where early
            # detection actually matters.
            bb = 0.15 * cavitation + 0.8 * cavitation**2
            vib += bb * np.diff(noise, prepend=noise[0])

            n_impl = int(self.rng.poisson(max(1.0, 350.0 * cavitation * dt)))
            if n_impl > 0:
                # Both the collapse rate and its violence grow as the NPSH
                # margin falls.
                mag = self.rng.gamma(2.0, 2.8 * (0.4 + 0.6 * cavitation), size=n_impl)
                signs = self.rng.choice([-1.0, 1.0], size=n_impl)
                starts = self.rng.integers(0, n, size=n_impl)
                # Each implosion rings briefly rather than being a single
                # sample, so it survives anti-alias filtering.
                ring_len = 24
                decay = np.exp(-np.arange(ring_len) / 5.0) * np.sin(
                    2.0 * math.pi * 0.28 * np.arange(ring_len)
                )
                for start, m, sgn in zip(starts, mag, signs, strict=True):
                    end = min(n, start + ring_len)
                    vib[start:end] += sgn * m * decay[: end - start]

        # --- bearing defects at the inner race and on a rolling element ------
        # Only when one is injected: a run of the original four modes draws
        # the same random numbers and synthesises the same samples as before.
        inner = sev.get("bearing_inner_race", 0.0)
        rolling = sev.get("bearing_rolling_element", 0.0)
        if inner > 0.0 or rolling > 0.0:
            vib += self._location_faults(n, freqs, rev, inner, rolling)

        # Axial channel: mostly a fraction of radial, but misalignment drives
        # it hard — that is what distinguishes it from imbalance.
        axial = 0.28 * vib + (1.8 * misalign) * a1 * np.sin(two_pi * rev + 0.9)
        axial += self.rng.normal(0.0, 0.12, n)

        # --- slow channels ---------------------------------------------------
        # Bearing friction heats as the race degrades; the rise lags the
        # vibration, which is why temperature alone is a late indicator.
        bearing_temp = (
            BEARING_TEMP_C
            + 26.0 * bearing**1.5
            + 4.0 * misalign
            + 3.0 * leak
            + self.rng.normal(0.0, 0.25)
        )
        motor_temp = (
            MOTOR_TEMP_C
            + 9.0 * bearing
            + 6.0 * misalign
            + 2.0 * cavitation
            + self.rng.normal(0.0, 0.3)
        )

        # Cavitation oscillates discharge pressure at a low rate as the
        # vapour cavity grows and collapses; a seal leak simply bleeds it off.
        cav_osc = 0.35 * cavitation * math.sin(2.0 * math.pi * 0.3 * self.t_s)
        discharge = (
            DISCHARGE_BAR * (1.0 - 0.18 * leak)
            + cav_osc
            + self.rng.normal(0.0, 0.02)
        )
        suction = (
            SUCTION_BAR * (1.0 - 0.45 * cavitation) + self.rng.normal(0.0, 0.012)
        )
        flow = (
            FLOW_M3H * (1.0 - 0.15 * leak - 0.08 * cavitation)
            + self.rng.normal(0.0, 0.25)
        )
        current = (
            CURRENT_A
            * (1.0 + 0.05 * bearing + 0.04 * misalign - 0.03 * leak)
            + self.rng.normal(0.0, 0.08)
        )

        if inner > 0.0 or rolling > 0.0:
            for mode, level in (("bearing_inner_race", inner),
                                ("bearing_rolling_element", rolling)):
                if level > 0.0:
                    p = location_fault_physics(mode)
                    bearing_temp += p["temp_rise_c"] * level ** p["exponent"]
                    motor_temp += p["motor_temp_rise_c"] * level
                    current += CURRENT_A * p["current_rise"] * level

        frame = Frame(
            t_s=self.t_s,
            wall_s=self.wall_s,
            vibration=vib,
            vibration_axial=axial,
            rpm=self.rpm,
            bearing_temp_c=bearing_temp,
            motor_temp_c=motor_temp,
            discharge_bar=discharge,
            suction_bar=suction,
            flow_m3h=flow,
            current_a=current,
            severities=sev,
        )
        self.t_s += dt * self.time_scale
        self.wall_s += dt
        return frame

    def run(self, duration_s: float, dt: float = 1.0) -> Iterator[Frame]:
        """Yield frames covering `duration_s` of machine time."""
        steps = round(duration_s / dt)
        for _ in range(steps):
            yield self.step(dt)

    @property
    def elapsed_machine_hours(self) -> float:
        return self.t_s / 3600.0

    # --- internals -------------------------------------------------------

    def _impulse_train(
        self, n: int, freq_hz: float, rev: np.ndarray, amplitude: float, damping: float
    ) -> np.ndarray:
        """A train of exponentially-decaying impulses at `freq_hz`.

        Modulated by shaft rotation, because a rolling element carries more
        load — and so strikes a defect harder — in the loaded half of its
        orbit. That modulation is what produces the 1X sidebands an analyst
        looks for around the defect frequency.
        """
        t = np.arange(n, dtype=np.float64) / self.fs_vib
        period = 1.0 / freq_hz
        # Time since the most recent impulse, for every sample.
        since = np.mod(t + self._impulse_offset(freq_hz), period)
        ring = np.exp(-damping * since) * np.sin(2.0 * math.pi * 2400.0 * since)
        load_zone = 0.65 + 0.35 * np.sin(2.0 * math.pi * rev)
        return amplitude * ring * load_zone

    def _location_faults(self, n: int, freqs: dict[str, float],
                         rev: np.ndarray, inner: float, rolling: float) -> np.ndarray:
        """Inner-race and rolling-element defects, from sim/fault_profiles.yaml.

        Inner race: a strike at BPFI. The defect turns with the shaft, through
        the load zone once a revolution, so the strikes are amplitude-modulated
        at 1X: side-peaks at BPFI +-1X in the envelope.

        Rolling element: the defect on a ball meets the inner and the outer
        race once each per spin, a strike at 2xBSF, unequal in strength
        (the two paths to the sensor differ). The ball is carried round the
        load zone by the cage, so the strikes are modulated at FTF.
        """
        t = self.wall_s + np.arange(n, dtype=np.float64) / self.fs_vib
        out = np.zeros(n)
        if inner > 0.0:
            p = location_fault_physics("bearing_inner_race")
            rate = p["rate_x_bpfi"] * freqs["bpfi"]
            since = np.mod(t, 1.0 / rate)
            ring = np.exp(-p["damping"] * since) * np.sin(2.0 * math.pi * p["ring_hz"] * since)
            depth = p["modulation_depth"]
            mod = (1.0 - depth) + depth * 0.5 * (1.0 + np.sin(2.0 * math.pi * rev))
            amp = p["amplitude"] * inner ** p["exponent"]
            out += amp * ring * mod
            out += self.rng.normal(0.0, p["floor"] * amp * inner, n)
        if rolling > 0.0:
            p = location_fault_physics("bearing_rolling_element")
            rate = p["rate_x_bsf"] * freqs["bsf"]
            since = np.mod(t, 1.0 / rate)
            strike = np.floor(t * rate)
            alternate = 1.0 + p["alternation"] * np.where(np.mod(strike, 2.0) < 1.0, 1.0, -1.0)
            ring = np.exp(-p["damping"] * since) * np.sin(2.0 * math.pi * p["ring_hz"] * since)
            cage = freqs["ftf"] * t
            depth = p["modulation_depth"]
            mod = (1.0 - depth) + depth * 0.5 * (1.0 + np.sin(2.0 * math.pi * cage))
            amp = p["amplitude"] * rolling ** p["exponent"]
            out += amp * ring * alternate * mod
            out += self.rng.normal(0.0, p["floor"] * amp * rolling, n)
        return out

    def _impulse_offset(self, freq_hz: float) -> float:
        """Phase offset keeping the impulse train continuous across frames."""
        return (self.t_s * freq_hz % 1.0) / freq_hz
