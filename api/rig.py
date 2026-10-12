"""ctypes binding for `libharmonic_rig` (HARM-S5-T02, the C half).

The API drives the real engine, not a Python re-implementation of it. Every
number the dashboard shows — the health index, the class scores, the surprise
per feature — comes out of the same compiled code the benchmark measures and
the determinism fixtures pin.

A subprocess would have been simpler to write and wrong to live with: the state
that matters here is per-tick and the engine is stateful, so a subprocess would
need a protocol, and the protocol would become a second place where the
pipeline order is defined.

**The struct layouts are verified, not assumed.** `harmonic_rig_out_size()` and
`harmonic_rig_frame_size()` are called at load time and compared against
`ctypes.sizeof`. A binding that disagrees with the C struct does not fail — it
silently reads the wrong fields, which is exactly what the packed frame record
cost this project once already.
"""

from __future__ import annotations

import contextlib
import ctypes
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

MAX_FEATURES = 32
MAX_CLASSES = 64
SLOW_CHANNELS = 7

Q16 = 65536.0


def q16(value: int) -> float:
    return value / Q16


def to_q16(value: float) -> int:
    return round(value * Q16)


class FrameRecord(ctypes.Structure):
    """Mirrors `harmonic_frame_record`, which is packed on disk and in the ABI."""

    _pack_ = 1
    _fields_ = [
        ("timestamp_us", ctypes.c_uint64),
        ("rpm_q16", ctypes.c_int32),
        ("load_q16", ctypes.c_int32),
        ("label_mask", ctypes.c_uint32),
        ("slow", ctypes.c_float * SLOW_CHANNELS),
        ("present", ctypes.c_uint32),
    ]


class RigOut(ctypes.Structure):
    """Mirrors `harmonic_rig_out`. Not packed: the C struct is not either."""

    _fields_ = [
        ("status", ctypes.c_int32),
        ("state", ctypes.c_int32),
        ("iso_state", ctypes.c_int32),
        ("regime", ctypes.c_int32),
        ("warm", ctypes.c_int32),
        ("hhi_q16", ctypes.c_int32),
        ("hhi_fast_q16", ctypes.c_int32),
        ("hhi_slow_q16", ctypes.c_int32),
        ("slope_q16_per_tick", ctypes.c_int32),
        ("rul_hours_q16", ctypes.c_int32),
        ("vib_rms_q16", ctypes.c_int32),
        ("feature_count", ctypes.c_uint32),
        ("class_count", ctypes.c_uint32),
        ("features_q16", ctypes.c_int32 * MAX_FEATURES),
        ("surprise_q16", ctypes.c_int32 * MAX_FEATURES),
        ("class_scores_q16", ctypes.c_int32 * MAX_CLASSES),
        ("event", ctypes.c_int32),
        ("tick", ctypes.c_uint64),
        ("vib_velocity_q16", ctypes.c_int32),
        ("rpm_q16", ctypes.c_int32),
        ("unrecognised", ctypes.c_int32),
        ("early", ctypes.c_int32),
        ("spread", ctypes.c_int32),
    ]


@dataclass
class Tick:
    """One frame's worth of engine output, in real units."""

    tick: int
    state: str
    iso_state: str
    regime: str
    warm: bool
    hhi: float
    hhi_fast: float
    hhi_slow: float
    slope_per_tick: float
    rul_hours: float
    vib_rms_mm_s: float           # the channel's RMS, in its own unit despite the name
    event: bool
    # RMS velocity in mm/s, what ISO zones are written in: the channel itself
    # for a velocity sensor, integrated over 10-1000 Hz for an accelerometer.
    vib_velocity_mm_s: float | None = None
    rpm: float = 0.0               # the speed the engine used: given, or estimated
    features: dict[str, float] = field(default_factory=dict)
    surprise: dict[str, float] = field(default_factory=dict)
    class_scores: list[float] = field(default_factory=list)
    # The state stands on an unrecognised change alone: the machine has moved
    # far from its baseline on several features, and no trained class and no
    # ISO zone accounts for it.
    unrecognised: bool = False
    # A fault the classes are sure of, held at WARNING because no measurement
    # is severe yet: recognised early, not yet a reason to shut down.
    early: bool = False
    # The spread floor raised the state to WARNING: evidence spread thinly over
    # several features, each under its own threshold. Always unrecognised too.
    spread: bool = False

    def top_surprised(self, n: int = 3, floor: float = 0.25) -> list[dict[str, object]]:
        """The features the engine is most surprised by, for the KG query.

        Surprise is signed — observed minus predicted, in log2 units — so the
        magnitude ranks and the sign gives the direction the graph models as
        `up` or `down`. A floor is applied because an event carrying three
        features, two of which barely moved, makes the graph rank a failure
        mode on noise.
        """
        ranked = sorted(self.surprise.items(), key=lambda kv: -abs(kv[1]))
        out: list[dict[str, object]] = []
        for kind, value in ranked[:n]:
            if abs(value) < floor:
                continue
            out.append({
                "kind": kind,
                "sigma": round(value, 3),
                "direction": "up" if value > 0 else "down",
                "value": round(self.features.get(kind, 0.0), 4),
            })
        return out


def _library_name() -> str:
    if sys.platform == "win32":
        return "harmonic_rig.dll"
    if sys.platform == "darwin":
        return "libharmonic_rig.dylib"
    return "libharmonic_rig.so"


def find_library() -> Path | None:
    """Search the usual build directories, then the loader path.

    `HARMONIC_RIG_LIB` overrides everything, which is what the container image
    uses: the library is built at image build time and lives outside the source
    tree.
    """
    override = os.environ.get("HARMONIC_RIG_LIB")
    if override:
        p = Path(override)
        return p if p.exists() else None
    name = _library_name()
    for candidate in (REPO / "build" / name, REPO / "build-clang" / name,
                      REPO / name, Path("/usr/local/lib") / name):
        if candidate.exists():
            return candidate
    return None


class RigUnavailable(RuntimeError):
    """The shared library is missing or unusable.

    Raised rather than papered over. The API can run without the engine — it
    still serves the graph and the manuals — but it must say that the live
    numbers are absent rather than serve zeros that look like a healthy machine.
    """


class HarmonicRig:
    """One engine instance. Not thread-safe: the engine is stateful per tick."""

    def __init__(self, brain_path: str | Path, window_samples: int = 16384,
                 sample_rate_hz: int = 25600, event_cooldown_sec: int = 0,
                 library: str | Path | None = None):
        lib_path = Path(library) if library else find_library()
        if lib_path is None or not Path(lib_path).exists():
            raise RigUnavailable(
                f"{_library_name()} not found. Build it with: "
                f"cmake --build build -j, or set HARMONIC_RIG_LIB."
            )
        self.lib = ctypes.CDLL(str(lib_path))
        self._declare()
        self._verify_layout()

        brain = Path(brain_path)
        if not brain.exists():
            raise RigUnavailable(f"brain not found: {brain}")

        self._handle = self.lib.harmonic_rig_create(
            str(brain).encode(), window_samples, sample_rate_hz, event_cooldown_sec)
        if not self._handle:
            raise RigUnavailable(self.lib.harmonic_rig_last_error().decode() or
                                 "harmonic_rig_create failed")

        self.brain_path = brain
        self.window_samples = window_samples
        self.sample_rate_hz = sample_rate_hz
        self.feature_count = int(self.lib.harmonic_rig_feature_count(self._handle))
        self.class_count = int(self.lib.harmonic_rig_class_count(self._handle))
        self.commissioned = bool(self.lib.harmonic_rig_is_commissioned(self._handle))
        self.nominal_rpm = q16(self.lib.harmonic_rig_nominal_rpm_q16(self._handle))
        # A feature's name is its kind, plus its label when the brain gives
        # one: two order features are both VIB_ORDER, and "VIB_ORDER:gear_mesh"
        # tells them apart.
        self.feature_kinds = []
        for i in range(self.feature_count):
            kind = self.lib.harmonic_rig_feature_kind_name(
                self.lib.harmonic_rig_feature_kind(self._handle, i)).decode()
            label = self.lib.harmonic_rig_feature_label(self._handle, i).decode()
            self.feature_kinds.append(f"{kind}:{label}" if label else kind)
        self.feature_units = [
            self.lib.harmonic_rig_feature_unit(self._handle, i).decode()
            for i in range(self.feature_count)
        ]
        # Each feature's dead zone, twice its healthy extreme: the scale a
        # surprise is judged on.
        self.feature_dead_zones = [
            q16(self.lib.harmonic_rig_feature_dead_zone_q16(self._handle, i))
            for i in range(self.feature_count)
        ]

    def _declare(self) -> None:
        lib = self.lib
        lib.harmonic_rig_create.restype = ctypes.c_void_p
        lib.harmonic_rig_create.argtypes = [ctypes.c_char_p, ctypes.c_int,
                                            ctypes.c_int, ctypes.c_uint32]
        lib.harmonic_rig_feed.restype = ctypes.c_int
        lib.harmonic_rig_feed.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(FrameRecord),
            ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_uint64,
            ctypes.POINTER(RigOut),
        ]
        lib.harmonic_rig_destroy.restype = None
        lib.harmonic_rig_destroy.argtypes = [ctypes.c_void_p]
        for name in ("harmonic_rig_state_name", "harmonic_rig_regime_name",
                     "harmonic_rig_feature_kind_name"):
            getattr(lib, name).restype = ctypes.c_char_p
            getattr(lib, name).argtypes = [ctypes.c_int]
        lib.harmonic_rig_feature_label.restype = ctypes.c_char_p
        lib.harmonic_rig_feature_label.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.harmonic_rig_feature_kind.restype = ctypes.c_int
        lib.harmonic_rig_feature_kind.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.harmonic_rig_feature_unit.restype = ctypes.c_char_p
        lib.harmonic_rig_feature_unit.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.harmonic_rig_set_guard.restype = None
        lib.harmonic_rig_set_guard.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.harmonic_rig_feature_dead_zone_q16.restype = ctypes.c_int32
        lib.harmonic_rig_feature_dead_zone_q16.argtypes = [ctypes.c_void_p, ctypes.c_int]
        for name in ("harmonic_rig_feature_count", "harmonic_rig_class_count"):
            getattr(lib, name).restype = ctypes.c_uint32
            getattr(lib, name).argtypes = [ctypes.c_void_p]
        lib.harmonic_rig_is_commissioned.restype = ctypes.c_int
        lib.harmonic_rig_is_commissioned.argtypes = [ctypes.c_void_p]
        lib.harmonic_rig_nominal_rpm_q16.restype = ctypes.c_int32
        lib.harmonic_rig_nominal_rpm_q16.argtypes = [ctypes.c_void_p]
        lib.harmonic_rig_last_error.restype = ctypes.c_char_p
        lib.harmonic_rig_last_error.argtypes = []
        for name in ("harmonic_rig_out_size", "harmonic_rig_frame_size"):
            getattr(lib, name).restype = ctypes.c_uint32
            getattr(lib, name).argtypes = []

    def _verify_layout(self) -> None:
        """Ask the library how big its structs are and insist on agreeing."""
        for struct, probe in ((RigOut, "harmonic_rig_out_size"),
                              (FrameRecord, "harmonic_rig_frame_size")):
            theirs = int(getattr(self.lib, probe)())
            ours = ctypes.sizeof(struct)
            if theirs != ours:
                raise RigUnavailable(
                    f"{struct.__name__} is {ours} bytes here and {theirs} in the "
                    f"library. The binding would read the wrong fields rather "
                    f"than fail, so it refuses to load."
                )

    def feed(self, samples, *, rpm: float, load: float = 0.65,
             slow: list[float] | None = None, timestamp_us: int = 0,
             dt_us: int = 1_000_000, present: int = 0x7F) -> Tick:
        """One frame through the whole pipeline."""
        if self._handle is None:
            raise RigUnavailable("this rig has been closed")
        n = len(samples)
        buf = (ctypes.c_float * n)(*samples) if not isinstance(
            samples, ctypes.Array) else samples

        frame = FrameRecord()
        frame.timestamp_us = timestamp_us
        frame.rpm_q16 = to_q16(rpm)
        frame.load_q16 = to_q16(load)
        frame.label_mask = 1
        channels = list(slow or [])[:SLOW_CHANNELS]
        channels += [0.0] * (SLOW_CHANNELS - len(channels))
        frame.slow = (ctypes.c_float * SLOW_CHANNELS)(*channels)
        frame.present = present

        out = RigOut()
        self.lib.harmonic_rig_feed(self._handle, ctypes.byref(frame), buf, n,
                                   ctypes.c_uint64(dt_us), ctypes.byref(out))
        return self._to_tick(out)

    def _to_tick(self, out: RigOut) -> Tick:
        n = min(int(out.feature_count), len(self.feature_kinds))
        return Tick(
            tick=int(out.tick),
            state=self.lib.harmonic_rig_state_name(out.state).decode(),
            iso_state=self.lib.harmonic_rig_state_name(out.iso_state).decode(),
            regime=self.lib.harmonic_rig_regime_name(out.regime).decode(),
            warm=bool(out.warm),
            hhi=q16(out.hhi_q16),
            hhi_fast=q16(out.hhi_fast_q16),
            hhi_slow=q16(out.hhi_slow_q16),
            slope_per_tick=q16(out.slope_q16_per_tick),
            rul_hours=q16(out.rul_hours_q16),
            vib_rms_mm_s=q16(out.vib_rms_q16),
            vib_velocity_mm_s=q16(out.vib_velocity_q16) if out.vib_velocity_q16 >= 0 else None,
            rpm=q16(out.rpm_q16),
            event=bool(out.event),
            unrecognised=bool(out.unrecognised),
            early=bool(out.early),
            spread=bool(out.spread),
            features={self.feature_kinds[i]: q16(out.features_q16[i]) for i in range(n)},
            surprise={self.feature_kinds[i]: q16(out.surprise_q16[i]) for i in range(n)},
            class_scores=[q16(out.class_scores_q16[i])
                          for i in range(int(out.class_count))],
        )

    def set_guard(self, on: bool) -> None:
        """The encoder's learning guard: on by default; off while measuring
        what normal is (the noise floor), which the guard would judge by."""
        self.lib.harmonic_rig_set_guard(self._handle, 1 if on else 0)

    def close(self) -> None:
        if getattr(self, "_handle", None):
            self.lib.harmonic_rig_destroy(self._handle)
            self._handle = None

    def __enter__(self) -> HarmonicRig:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        # Interpreter teardown can already have torn down ctypes; a failure
        # here would surface as noise during shutdown and nothing else.
        with contextlib.suppress(Exception):
            self.close()
