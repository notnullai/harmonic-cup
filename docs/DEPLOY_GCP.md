# Harmonic on Google Cloud

Harmonic has two halves. The **engine** runs at the edge, on the machine's own
controller — a deterministic C library, no cloud in the loop. The **cloud
half** — the console, the fleet, the diagnosis, the knowledge, the records —
runs on Google Cloud:

| Part | Google Cloud | In the code |
|---|---|---|
| API, console, the fleet of engines | **Cloud Run** (one always-on instance) | `api/`, `web/` |
| Knowledge graph — 367 nodes, 595 weighted relations, four machine families, traversed per event | **Spanner Graph**: a property graph over two tables, every traversal a GQL query (free-trial instance, 90 days) | `kg/store_spanner.py`, `kg/query.py` |
| Diagnosis and chat | **Vertex AI — Gemini** | `rag/provider.py` |
| Retrieval over the manuals | **Vertex AI embeddings**, searched in process | `rag/index.py` (`VectorIndex`) |
| Events, verdicts, work orders | **Firestore** | `api/store_firestore.py` |
| Live sensors from the plant | **Pub/Sub** (MQTT too, for a plant broker) | `api/adapters/pubsub.py` |
| Machines, uploads, recordings, embeddings cache | **Cloud Storage**, mounted as the data folder | `HARMONIC_DATA_DIR=/data` |
| Build | **Cloud Build**, **Artifact Registry** | `deploy/gcp/cloudbuild.yaml` |

The knowledge graph is a typed property graph, not a lookup table: a diagnosis
walks feature -> Symptom -INDICATES{confidence}-> FailureMode -AFFECTS->
Component <-HAS_COMPONENT- Equipment (to rule out what the machine cannot
have), then FailureMode -MITIGATED_BY-> Action -DOCUMENTED_IN-> ManualSection.
It sits behind one five-method store interface (`kg/store.py`) with three
stores behind it: Spanner Graph on Google Cloud, NetworkX in process, Neo4j
locally. Against the Spanner emulator the GQL traversals return the same edges
and the same diagnoses, score for score, as the in-process graph
(`tests/test_spanner_graph.py`). Most of it is curated by hand, one data file per
machine family (`kg/families/*.json`); the rest is extracted from the manuals
by the rules extractor (`kg/extract.py`); the Gemini extractor was run against it
on Vertex AI (HARM-S8-G08, docs/METRICS.md). The same code runs on a PC with SQLite, BM25 and mosquitto when no Google
Cloud project is configured — that is how the tests run.

## One-time setup (the project owner)

These steps use your Google account; nothing here is done with anyone else's
credentials.

1. **A project with billing.** In the console, create a project
   (<https://console.cloud.google.com/projectcreate>) and link a billing
   account. Note the **project ID**.
2. **gcloud in WSL (Ubuntu):**

   ```bash
   sudo apt-get install -y apt-transport-https ca-certificates gnupg curl
   curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
   echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee /etc/apt/sources.list.d/google-cloud-sdk.list
   sudo apt-get update && sudo apt-get install -y google-cloud-cli
   ```

3. **Billing.** Link a billing account to the project (Cloud Run, Spanner and
   Vertex AI refuse to start without one):
   `https://console.cloud.google.com/billing/linkedaccount?project=YOUR_PROJECT_ID`.
4. **Log in.** WSL cannot open a browser itself ("Operation not supported"):
   copy the printed link into the Windows browser, and on the consent page tick
   **every** box — an unticked "Google Cloud data" box fails the second login
   with "cloud-platform scope is required but not consented".

   ```bash
   gcloud auth login
   gcloud auth application-default login
   gcloud config set project YOUR_PROJECT_ID
   ```

5. **Pick a region.** Near the plant, where the models are offered. Verified:
   `asia-southeast1` (Singapore) serves Gemini 2.5 Flash, `text-embedding-005`
   and the Spanner free trial; `us-central1` has the widest model list.

## Deploy

From the repository root, in WSL:

```bash
PROJECT=YOUR_PROJECT_ID REGION=asia-southeast1 deploy/gcp/deploy.sh
```

It enables the APIs, creates the Artifact Registry repository, the bucket,
the Firestore database, the Spanner free-trial instance, a demo Pub/Sub topic,
a service account for the service and one for the build, builds the image with Cloud Build (both machine families'
brains are trained during the build — a few minutes), and deploys to Cloud
Run. Running it again only rebuilds and redeploys.

**Access.** The service is private by default: open it through
`gcloud run services proxy harmonic --region REGION` and
<http://localhost:8080>. `PUBLIC=1` makes the URL open to anyone — needed if
judges must use it without a Google login, at the price that anyone can
inject faults, upload recordings and spend Gemini calls (one instance at most
caps the cost).

**Cost.** One always-on Cloud Run instance, 2 vCPU and 4 GiB with CPU always
allocated — the fleet ticks its engines continuously — is roughly $30–60 a
month; Firestore, Pub/Sub and Storage stay in or near their free tiers;
Gemini is a few cents per diagnosis. When the judging is over:

```bash
gcloud run services delete harmonic --region REGION
```

### Try it: the public try service (HARM-S10-T02)

```bash
PROJECT=YOUR_PROJECT_ID REGION=asia-southeast1 deploy/gcp/deploy_try.sh
```

The same image, as a **separate** service `harmonic-try` with
`HARMONIC_MODE=try` (only `/try` is served; every console route is 404). It
meets the deployment conditions in `docs/TRY_IT_SPEC.md`:

- its own bucket `gs://PROJECT-harmonic-try` mounted at `/data`, with public
  access prevention and a lifecycle rule deleting objects after 8 days
  (`LIFECYCLE_DAYS`) — a backstop for the app's 7-day sweep, which runs only
  while an instance is up (the service scales to zero). **Trade-off:** the
  rule also removes runs whose visitor ticked "keep" (90 days in the app)
  after 8 days; set `LIFECYCLE_DAYS=91` to keep them, at the price that an
  idle service can hold other runs past 7 days until its next start;
- its own service account `harmonic-try` holding only `storage.objectAdmin`
  on that bucket: no project role, so no Spanner, Vertex, Gemini, Firestore
  or Pub/Sub;
- `max-instances=1`, 2 vCPU / 4 GiB, CPU allocated while up, min 0;
- the NetworkX graph built into the image (`HARMONIC_KG_BACKEND=networkx`),
  never the console's Spanner graph; no Gemini or Vertex settings;
- `HARMONIC_TRY_DISK_QUOTA_MB=2048`, below the 4 GiB memory (Cloud Run's
  local disk is RAM, and the bucket mount stages files being written there);
- `HARMONIC_CONTACT_URL=mailto:contact@khalm.ai` (`CONTACT_URL` to change).

It reuses the image repository and build account `deploy.sh` made, builds only
when the image for the current commit is missing (`BUILD=1` forces it), and
never touches the console service or its bucket. **Private by default;**
`PUBLIC=1` grants `roles/run.invoker` to `allUsers` on `harmonic-try` only,
and a later run without it takes the grant away.

**Smoke tests at every deploy** (private: call the run.app URL directly with
`Authorization: Bearer $(gcloud auth print-identity-token)`, not through the
proxy, so requests pass Google's front end):

1. `/try/` and `/try/limits` answer; `/health`, `/machines` and the other
   console routes are 404.
2. Forwarded address: create runs with a spoofed, changing left
   `X-Forwarded-For` (and a second spoofed header line); the per-visitor
   limit (20 creates an hour) must still answer 429 on the 21st. The spec's
   two-network test (each network counted separately) needs a second network.
3. A full run: a CSV of 180 snapshots, configure, start, report, delete; the
   run's objects must be gone from the bucket.
4. A 20–40 MB file in 16 MB parts (`?offset=`) is assembled and configures.
5. The service account's roles: only the bucket binding above.

Last run 2026-09-29 (commit `e8a8bc3`), private,
<https://harmonic-try-ggp7bau73a-as.a.run.app/try/>: all five passed; the
two-network test is still open.

## A live sensor on Pub/Sub

The deploy creates the topic `harmonic-demo-vibration`. Publish a simulated
gearbox into it from the PC:

```bash
python scripts/mqtt_publish.py --broker pubsub:YOUR_PROJECT_ID \
    --topic harmonic-demo-vibration --family gearbox --rate 2 --inject broken_tooth@300
```

In the console's commissioning wizard, **Live sensor** with broker `pubsub`
and topic `harmonic-demo-vibration` records it, commissions from it and keeps
watching it.

## Gemini from the PC

With the application-default login above, the local API, the pipeline CLI and
`kg.extract --extractor gemini` use Vertex AI too:

```bash
export GOOGLE_GENAI_USE_VERTEXAI=1 GOOGLE_CLOUD_PROJECT=YOUR_PROJECT_ID        GOOGLE_CLOUD_LOCATION=asia-southeast1
gcloud auth application-default set-quota-project YOUR_PROJECT_ID
```
