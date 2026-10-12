"""MFPT bearing dataset -> Harmonic frames.

The Society for Machinery Failure Prevention Technology publishes bearing
vibration recordings with known outer-race and inner-race faults. They are the
reality check on the simulator: if the feature extractor finds BPFO in MFPT's
real outer-race data with the same code path it uses on synthetic data, the
synthetic bearing model is doing its job.

Each MFPT record is a MATLAB file holding a `bearing` struct:

    gs    acceleration samples (g)
    sr    sample rate (Hz)
    rate  shaft rate (Hz), or load in some records
    load  applied load (lbs)

Acceleration is integrated to velocity, because Harmonic works in mm/s so it
can be compared against ISO 10816 zones directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sim.pump_sim import Frame

G_TO_MM_S2 = 9806.65  # 1 g in mm/s^2


@dataclass
class MFPTRecord:
    """One MFPT recording, in engineering units."""

    name: str
    acceleration_g: np.ndarray
    sample_rate_hz: float
    shaft_rate_hz: float
    load_lbs: float
    fault: str  # "healthy", "outer_race", "inner_race"

    @property
    def duration_s(self) -> float:
        return len(self.acceleration_g) / self.sample_rate_hz


def _fault_from_name(name: str) -> str:
    low = name.lower()
    if "outer" in low or "or_" in low:
        return "outer_race"
    if "inner" in low or "ir_" in low:
        return "inner_race"
    if "baseline" in low or "normal" in low or "healthy" in low:
        return "healthy"
    return "unknown"


def load_mfpt(path: str | Path) -> MFPTRecord:
    """Read one MFPT `.mat` file.

    Requires scipy, which is a loader-only dependency: the engine and the
    simulator do not need it.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. MFPT data is not vendored — download it from "
            "https://www.mfpt.org/fault-data-sets/ into sim/datasets/raw/mfpt/"
        )
    try:
        from scipy.io import loadmat
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "loading MFPT .mat files needs scipy: pip install scipy"
        ) from exc

    mat = loadmat(str(p), squeeze_me=True, struct_as_record=False)
    bearing = mat.get("bearing")
    if bearing is None:
        raise ValueError(f"{p} has no 'bearing' struct; is this an MFPT file?")

    return MFPTRecord(
        name=p.stem,
        acceleration_g=np.asarray(bearing.gs, dtype=np.float64).ravel(),
        sample_rate_hz=float(bearing.sr),
        shaft_rate_hz=float(getattr(bearing, "rate", 0.0)),
        load_lbs=float(getattr(bearing, "load", 0.0)),
        fault=_fault_from_name(p.stem),
    )


def acceleration_to_velocity(
    accel_g: np.ndarray, fs: float, highpass_hz: float = 2.0
) -> np.ndarray:
    """Integrate acceleration (g) to velocity (mm/s).

    Integration amplifies low-frequency noise without bound, so the result is
    high-passed. Without that, a tiny DC offset in the accelerometer becomes an
    ever-growing velocity ramp and swamps the signal.
    """
    accel = np.asarray(accel_g, dtype=np.float64) * G_TO_MM_S2
    accel = accel - np.mean(accel)
    velocity = np.cumsum(accel) / fs

    # First-order high-pass, applied as a detrend in the frequency domain.
    spec = np.fft.rfft(velocity)
    freqs = np.fft.rfftfreq(len(velocity), d=1.0 / fs)
    with np.errstate(divide="ignore", invalid="ignore"):
        response = freqs / np.sqrt(freqs**2 + highpass_hz**2)
    response[0] = 0.0
    return np.fft.irfft(spec * response, n=len(velocity))


def mfpt_to_frames(
    record: MFPTRecord, window_s: float = 1.0, load: float = 1.0
) -> list[Frame]:
    """Slice a record into Frames the rest of the pipeline already understands.

    The slow channels MFPT does not measure are left at zero rather than
    invented: a downstream consumer must be able to tell a real missing channel
    from a plausible-looking fabrication.
    """
    velocity = acceleration_to_velocity(record.acceleration_g, record.sample_rate_hz)
    n_per_window = round(record.sample_rate_hz * window_s)
    if n_per_window <= 0:
        raise ValueError("window_s too small for this sample rate")

    rpm = record.shaft_rate_hz * 60.0
    frames: list[Frame] = []
    n_windows = len(velocity) // n_per_window
    severity = 0.0 if record.fault == "healthy" else 1.0
    mode = {"outer_race": "bearing_wear", "inner_race": "bearing_wear"}.get(
        record.fault, ""
    )

    for i in range(n_windows):
        chunk = velocity[i * n_per_window : (i + 1) * n_per_window]
        sev = {"bearing_wear": 0.0, "misalignment": 0.0, "cavitation": 0.0, "seal_leak": 0.0}
        if mode:
            sev[mode] = severity
        frames.append(
            Frame(
                t_s=i * window_s,
                wall_s=i * window_s,
                vibration=chunk,
                vibration_axial=np.zeros_like(chunk),  # MFPT records one axis
                rpm=rpm,
                bearing_temp_c=0.0,   # not measured
                motor_temp_c=0.0,     # not measured
                discharge_bar=0.0,    # not measured
                suction_bar=0.0,      # not measured
                flow_m3h=0.0,         # not measured
                current_a=0.0,        # not measured
                severities=sev,
            )
        )
    return frames
