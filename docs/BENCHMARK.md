# The Harmonic benchmark

This is written for
anyone deciding whether to trust Harmonic on their own machine: a
reliability engineer, not a judge or an investor. Misses are in here on
purpose.

## What we tested

One fixed method, run unchanged over every public bearing-failure dataset we
could get our hands on, plus one real gearbox. Nothing below is tuned per
dataset — the commissioning rule, the noise-floor rule and the alarm rule
are identical in every row.

**The method, in plain words:**

1. **Commission on the healthy start of the recording.** Harmonic watches
   the first stretch of normal running (the "commissioning window" — how
   long it is per dataset is in the table below) and learns what normal
   vibration looks like for *this* machine, on *its own* sensors.
2. **The noise floor is measured, per sensor, from that window alone** — not
   copied from another machine, not hand-tuned to the fault we're about to
   show it. (One exception, stated where it applies: the four IMS runs share
   one profile, measured once on the first of them, precisely to test
   whether that's a fair simplification — see "What this gets wrong,
   honestly" below.)
3. **Everything after that window is played through unchanged** — the
   recording, in order, exactly as it happened, with no relabelling and no
   cherry-picking of a "good" stretch.
4. **Harmonic raises a state (WATCH → WARNING → CRITICAL) when a reading is
   far enough from what commissioning learned**, and the knowledge graph
   names a likely failure mode from which measurements moved and by how
   much.

This is exactly the method in `scripts/evaluate_ims.py` and
`scripts/evaluate_xjtu.py`; `scripts/benchmark.py` (HARM-S10-T08) runs it,
and the two older, dataset-specific scripts it reuses
(`scripts/evaluate_cwru.py`, `scripts/evaluate_gearbox_real.py`), over every
dataset below in one pass and writes the numbers this page quotes.

## What we measured, per run

- **Detection and lead** — when Harmonic first raised an alarm, and how long
  before the recording actually ends (or the failure the dataset's authors
  recorded).
- **False alarms while learning normal** — how many of the commissioning
  readings themselves alarmed. This is the honesty check on the noise floor:
  if commissioning already looks broken, the method has failed before it
  started.
- **The fault named, right or wrong** — what the knowledge graph said broke,
  compared with what the dataset's authors say actually broke. Wrong names
  are counted, not excluded.
- **A fair classical comparator** — a plain 3-sigma-over-normal threshold on
  overall vibration RMS, measured on the same commissioning window, with two
  variants: fires on the first reading over threshold, or only once 3
  readings in a row are over threshold (the same "3 in a row" rule Harmonic's
  own WATCH state uses internally, so the comparison isn't stacked against
  the simple method by demanding less patience of it). This shows what a
  reliability engineer's simplest possible check would have caught, on the
  exact same data.

## The datasets

| Dataset | What it is | Runs in this benchmark |
|---|---|---|
| **NASA IMS** | Four bearings run to failure on a test rig, recorded until they broke | All four recorded failures: test 2 bearing 1 (outer race), test 1 bearings 3 and 4 (inner race, roller element), test 3 bearing 3 (outer race) |
| **XJTU-SY** | Fifteen bearings run to failure under three speed/load conditions | All 15, both accelerometers (horizontal and vertical), each commissioned as its own sensor |
| **CWRU** | A 2 hp motor with seeded bearing faults, drive-end, six fault records plus a held-out healthy load | All 6 faults, plus the held-out normal record |
| **Real gearbox (OpenEI)** | A test gearbox recorded healthy and with a physically broken tooth, at ten loads | The broken tooth at all 10 loads, plus 5 held-out healthy loads |

## Results

The full per-run table — every bearing, every load, every fault record, with
its detection time, lead, false-alarm count, name and comparator result — is
generated straight from the runner and kept alongside it, not retyped here
by hand: **[`docs/artifacts/benchmark.md`](artifacts/benchmark.md)**
(machine-readable version: [`docs/artifacts/benchmark.json`](artifacts/benchmark.json)).
What follows is the honest summary.

| Dataset | Runs | Detected (alarmed at least once) | False alarms while learning normal | Fault named, right/wrong | Fair 3σ-sustained comparator, ever fired |
|---|---:|---:|---:|---:|---:|
| NASA IMS | 4 failures | **4 of 4** | 106 of 1,367 commissioning readings (all from one bearing — see below) | 522 / 1,355 (per alarmed reading) | 4 of 4 |
| XJTU-SY | 15 bearings | **15 of 15** (3_5 on its vertical accelerometer only) | 0 of 5,422 (both sensors) | 1,552 / 779 (per alarmed reading) | 15 of 15 |
| CWRU | 6 faults + 1 held-out normal | **6 of 6** faults; 0 false alarms on the held-out 3 hp normal | 0 of 1,589 | 6 / 0 — see the caveat below, this is coarser than it looks | 6 of 6 |
| Real gearbox (OpenEI) | 10 broken-tooth loads + 5 held-out normal | **10 of 10**; 0 false alarms on any of the 5 held-out normal loads | 0 of 6,150 | 4 / 6 | 0 of 10 — the classical comparator never catches this fault at all |

**Read the "named right/wrong" column carefully — it hides two different kinds
of "right":**

- **IMS and XJTU-SY** count every alarmed *reading* and compare the graph's
  name against the dataset's own labelled failure, sub-fault included (inner
  race vs. outer race vs. roller/cage). This is the strict count, and it's
  where most of the wrong names are: 1,355 of 1,877 alarmed IMS readings and
  779 of 2,331 alarmed XJTU-SY readings (both sensors) are named wrong — almost always
  "outer race" on a bearing that actually failed some other way (see
  `docs/METRICS.md`, "IMS: all four recorded failures").
- **CWRU's "6/0" looks perfect, and it isn't a sub-fault check at all.** The
  motor family has exactly one bearing-failure mode — every alarm the
  bearing_wear class produces is named "outer race", full stop — so
  "expected" and "diagnosis" are the same string for every one of the six
  records, whether the seeded fault is on the outer race, the inner race or
  a ball. All six come back "right" by construction, not because the engine
  told inner race from outer race: it can't, on this family. The true
  picture is the raw name per record (`docs/METRICS.md`, "Real data for the
  new families"): two of six (130, 234) are genuinely outer-race faults
  named outer race; the other four (two inner-race, two ball) are *detected*
  but named outer race regardless of what actually failed. The real
  gearbox's "4/6" is the honest version of the same idea, because the
  gearbox family has more than one fault class to choose between: of the
  ten broken-tooth loads, four are named the broken tooth exactly (50%, 70%,
  80%, 90%), three are named a gear fault of the wrong kind (tooth wear: 0%,
  10%, 40%), and three are named the wrong component (a bearing: 20%, 30%,
  60%) — all correctly counted as wrong here.

**IMS's 106 commissioning false alarms are entirely test 1, bearing 4.** The
other three IMS runs, all 15 XJTU-SY bearings, CWRU and the real gearbox
have **zero** false alarms while learning normal — 14,086 clean commissioning
readings in total (925 + 5,422 + 1,589 + 6,150), across four different real
rigs, none of them seen during development. Bearing 4
is the one place the fixed, test-2-derived noise floor doesn't fit:
`docs/METRICS.md` traces it to a real kurtosis episode 170–250 hours into
that specific bearing's commissioning window, not a general failure of
"measure the floor once and reuse it."

## What this gets wrong, honestly

- **The class neurons only know the faults they were trained on.** On IMS
  and CWRU, an inner-race or roller/ball fault is very often detected —
  Harmonic raises the alarm — but *named* "outer race", because the trained
  bearing-wear class was tuned toward outer-race symptoms. That's a wrong
  name on a right detection, and it's counted as wrong above, not hidden.
  See `docs/METRICS.md` ("IMS: all four recorded failures") for the full
  count and the root cause.
- **A fault the simulator never taught in a form that transfers is often
  not named right.** The real gearbox's broken tooth looks different in real
  vibration (mesh energy rises, kurtosis *falls*) than the simulated broken
  tooth it was trained on (an impact, kurtosis rises). Before the spread
  floor (HARM-S11-T12, `docs/DETECTION_MISSES.md`) four of the ten loads
  never alarmed: each feature moved only 1–2 standard deviations, under its
  own threshold. The floor adds that evidence up, and all ten now alarm,
  11.7–17.2 h before the end; six are still named wrong.
- **XJTU-SY 3_5 is caught by one sensor only.** Its horizontal accelerometer
  never alarms; its vertical one does, 1.0 h before the end. A machine
  watched by the horizontal sensor alone would not be warned.
- **The IMS profile is deliberately not re-tuned per test or per bearing** —
  it's measured once, on test 2 bearing 1, and reused unchanged for the
  other three. That's the honest, harder setting; a floor measured on each
  test separately would score better and prove less. One consequence: test 1
  bearing 4's WATCH/WARNING land inside its own commissioning window, not
  cleanly after it — reported as measured in `docs/METRICS.md`, not tuned
  away.
- **The classical 3-sigma comparator is a low bar, and it still sometimes
  wins on raw lead time** on data with a single very sharp outlier — that's
  what the "no-sustain" variant is for: without a persistence requirement, a
  single noisy reading can look like an early catch. The sustained variant
  (3 in a row, Harmonic's own rule) is the fairer comparison and is reported
  alongside it, not instead of it.

## Licences and citations

Every dataset here is used exactly as its licence allows for evaluation and
publication (checked in `docs/BENCHMARK_DATASETS.md`, HARM-S9-T12):

- **NASA IMS** — a U.S. Government Work (public domain), data donated by the
  University of Cincinnati's Intelligent Maintenance Systems (IMS) center.
  Please acknowledge the repository and the donors if you reuse it.
- **XJTU-SY** — no formal licence file; the authors' README says the data
  are publicly available for validating prognostics algorithms and asks for
  a citation:
  B. Wang, Y. Lei, N. Li, N. Li, "A Hybrid Prognostics Approach for
  Estimating Remaining Useful Life of Rolling Element Bearings," *IEEE
  Transactions on Reliability*, 69(1), 401–412, 2020.
  (Evaluation and publication only — not used to train anything Harmonic
  ships; see `docs/BENCHMARK_DATASETS.md`.)
- **CWRU Bearing Data Center** — no licence or terms stated on the Case
  School of Engineering's own site; used here with the customary citation of
  the Bearing Data Center (https://engineering.case.edu/bearingdatacenter).
- **Real gearbox** — "Gearbox Fault Diagnosis Data," OpenEI submission 623,
  **CC BY 4.0**.

## How to reproduce this

Everything here comes from data files, not from us:

```
datasets/ims/...                                  # NASA IMS, extracted (see sim/datasets/ims_loader.py)
datasets/xjtu-sy/XJTU-SY_Bearing_Datasets/...      # XJTU-SY, extracted (see sim/datasets/xjtu_loader.py)
datasets/cwru/<n>.mat                              # CWRU, from the Bearing Data Center
datasets/openei_gearbox/{Healthy Data,BrokenTooth Data}/*.txt   # OpenEI submission 623

python scripts/benchmark.py
# writes docs/artifacts/benchmark.json and docs/artifacts/benchmark.md
```

No dataset is downloaded by the runner itself, no result is hand-edited
after the fact, and nothing is retuned between datasets — the same
commissioning rule, the same noise-floor rule and the same alarm rule run
everywhere. If you get different numbers than this page, that's a bug
report we want: open an issue with your `benchmark.json`.

`scripts/benchmark.py` also reuses `scripts/evaluate_ims.py`,
`scripts/evaluate_xjtu.py`, `scripts/evaluate_cwru.py` and
`scripts/evaluate_gearbox_real.py` directly (it does not reimplement any of
their method) — each is runnable on its own for a closer look at one
dataset, and each has its own docstring with the exact commands.
