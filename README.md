# Harmonic — predictive maintenance on Google Cloud, with no failure history

**AI Builder Cup 2026 · notnullai (KHALM LLC)**

Harmonic watches rotating machines: pumps, motors, fans and gearboxes. It
learns what normal is for each machine, so it needs no history of past
failures. When a machine's vibration changes, Google Cloud turns that change
into a diagnosis, a safety briefing and a work order that an engineer can
check line by line.

## Try it

- **On your own data, no sign-up:** <https://harmonic-try-ggp7bau73a-as.a.run.app/try/>.
  Upload a vibration recording, ideally from a machine that later failed. It
  tells you whether Harmonic would have warned you, and when. A built-in
  sample (a real NASA bearing, run to failure) is one click away.
- **The benchmark, misses included:** <https://harmonic-try-ggp7bau73a-as.a.run.app/try/benchmark>.

The operator console is private; it is shown in the demo video.

## How one alarm travels

1. A small detector on the machine's controller notices that the vibration has
   moved away from normal, and raises an alarm.
2. **Pub/Sub** carries the event to the Harmonic service on **Cloud Run**.
3. **Spanner Graph** narrows the change to the failures this machine can
   actually have. **Vertex AI embeddings** find the matching sections of that
   machine's own manual.
4. **Gemini** on Vertex AI writes the diagnosis, the root causes, the safety
   steps and a ranked action plan, from the graph and those sections only.
5. A guardrail checks the schema, every citation and the confidence.
   **Firestore** keeps the event, and a work order goes to the maintenance
   system.

## Built on Google Cloud

| Job | Service |
|---|---|
| Diagnose, plan, answer questions, read the manuals | Gemini 2.5 Flash on Vertex AI |
| Find the right manual section | Vertex AI embeddings (`text-embedding-005`) |
| Know what a symptom means on this machine | Spanner Graph |
| Keep every alarm, verdict and work order | Firestore |
| Bring in live sensor readings from the plant | Pub/Sub |
| Hold machines, recordings and manuals | Cloud Storage |
| Serve the console, the API and the public try page | Cloud Run |
| Build and ship | Cloud Build, Artifact Registry |

How it is deployed: [`docs/DEPLOY_GCP.md`](docs/DEPLOY_GCP.md).

## Results on real failures

Public datasets. Nothing was trained on these machines; each learned normal
from its own healthy running.

| Real data | Failures caught | Notes |
|---|---|---|
| NASA IMS bearing rig, run to failure | 4 of 4 | fault location named right on 2 of 4 |
| XJTU-SY bearing rig, 15 bearings | 15 of 15 | 0 false alarms while learning normal; one bearing caught by only one of its two sensors |
| CWRU motor bearings, seeded faults | 6 of 6 | 0 false alarms on held-out healthy running |
| Real gearbox with a broken tooth (OpenEI), 10 loads | 10 of 10 | named "broken tooth" at only 4 of 10 |

Catching a fault is not the same as naming it: naming which part failed is the
known weak point. The method, the full numbers and the misses are in
[`docs/BENCHMARK.md`](docs/BENCHMARK.md); the datasets and their licences are
in [`docs/BENCHMARK_DATASETS.md`](docs/BENCHMARK_DATASETS.md).

## What is in this repository

| Folder | What it holds |
|---|---|
| `api/` | The service (Python, FastAPI): the console's API, the public try mode (`try_mode.py`), live data in from Pub/Sub and MQTT (`adapters/`), the Gemini hook and chat, work orders, the Firestore store |
| `web/` | The operator console and the public try page (React, TypeScript) |
| `kg/` | The knowledge graph: its model, the machine-family data, the Spanner Graph and in-process stores, the diagnosis queries |
| `rag/` | The manuals, their index, the prompt, the guardrail and the diagnosis pipeline |
| `sim/` | Machine simulators and the loaders for the public datasets |
| `profiles/` | The machine-family catalogue, bearing data and vibration standards |
| `deploy/gcp/` | The deploy scripts for both Cloud Run services |
| `scripts/` | The benchmark and evaluation runners |
| `tests/` | The Python tests |
| `docs/` | The benchmark, the datasets and their licences, the knowledge-graph schema and the deploy guide |

## What is not in this repository

**The edge detector.** The detector that runs on the machine's controller is a
separate, proprietary component of ours and is not published. This submission
is the Google Cloud system built around it: the knowledge graph, the Gemini
diagnosis and its guardrail, the console and the public try page. The service loads it as a compiled library
through `api/rig.py`, and the container build (`api/Dockerfile`) expects its
source. So this repository does not build or run end to end by itself; the
live product linked above is the whole system.

## Licence

- **Code** in this repository: Apache License 2.0, see [`LICENSE`](LICENSE).
- **The machine-family library** (`kg/families/`, `rag/manuals/`, `profiles/`
  and the simulators in `sim/`): © 2026 KHALM LLC, all rights reserved for
  now. It is published for evaluation only; see
  [`LICENSE-FAMILY-LIBRARY.md`](LICENSE-FAMILY-LIBRARY.md).
- **The sample recording** in `api/try_samples/` is from the NASA IMS bearing
  dataset (NSF I/UCR Center for Intelligent Maintenance Systems, University of
  Cincinnati, via the NASA Prognostics Data Repository), a U.S. Government
  work.

## Contact

contact@khalm.ai · security reports: see [`SECURITY.md`](SECURITY.md).
