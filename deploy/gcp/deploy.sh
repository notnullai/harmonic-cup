#!/usr/bin/env bash
# Deploy Harmonic's cloud half to Google Cloud (HARM-S8-G04).
#
#   PROJECT=my-project REGION=us-central1 deploy/gcp/deploy.sh
#
# Idempotent: every step checks before it creates, so a second run only builds
# and redeploys. Run from the repository root, logged in with gcloud
# (`gcloud auth login`); it uses your credentials and never asks for others.
#
# What it creates, all in REGION:
#   APIs         Cloud Run, Cloud Build, Artifact Registry, Vertex AI,
#                Firestore, Pub/Sub, Cloud Storage
#   Artifact Registry repository "harmonic"        the image
#   bucket gs://$PROJECT-harmonic-data             machines, uploads, embeddings
#   Firestore database (default), native mode      events, feedback, work orders
#   Spanner free-trial instance "harmonic", db "kg"  the knowledge graph (Spanner
#                                                   Graph; free for 90 days)
#   Pub/Sub topic "harmonic-demo-vibration"        a live sensor for the demo
#   service account harmonic-run                   what Cloud Run runs as
#   Cloud Run service "harmonic"                   the API and the console
#
# Cost: Cloud Run runs ONE always-on instance (the fleet ticks engines
# continuously), 2 vCPU / 4 GiB, CPU always allocated — roughly $30-60 a month
# while it exists. The Spanner free trial ends 90 days after it is created; the
# graph can then go back in process (HARMONIC_KG_BACKEND=networkx) or the
# instance be upgraded. Delete the service when the judging is over:
#   gcloud run services delete harmonic --region $REGION
#
# Access: the service is PRIVATE by default (only accounts you grant can open
# it). Set PUBLIC=1 to allow anyone with the URL — needed if judges must open
# it without a Google login — accepting that anyone could then inject faults,
# upload recordings and spend Gemini calls. Max one instance caps the spend.
set -euo pipefail

# Cup submission freeze: once the tag v1.0-submission exists, the live services
# are built only from it, with a clean tree. Engine work goes on, on main,
# without reaching what is submitted and shown. FORCE_DEPLOY=1 overrides.
FREEZE_TAG="v1.0-submission"
if [ "${FORCE_DEPLOY:-0}" != "1" ] && git rev-parse -q --verify "refs/tags/${FREEZE_TAG}" >/dev/null 2>&1; then
    if [ "$(git rev-parse HEAD)" != "$(git rev-parse "${FREEZE_TAG}^{commit}")" ] \
        || [ -n "$(git status --porcelain --untracked-files=no)" ]; then
        echo "Refusing to deploy: the submission is frozen at ${FREEZE_TAG}." >&2
        echo "Check that tag out with a clean tree (git checkout ${FREEZE_TAG})," >&2
        echo "or set FORCE_DEPLOY=1 if you mean to replace the submitted build." >&2
        exit 1
    fi
fi

: "${PROJECT:?set PROJECT to your Google Cloud project id}"
REGION="${REGION:-us-central1}"
PUBLIC="${PUBLIC:-0}"
LLM_MODEL="${LLM_MODEL:-gemini-2.5-flash}"
EMBED_MODEL="${EMBED_MODEL:-text-embedding-005}"
REPO=harmonic
BUCKET="${PROJECT}-harmonic-data"
SA="harmonic-run@${PROJECT}.iam.gserviceaccount.com"
BUILD_SA="harmonic-build@${PROJECT}.iam.gserviceaccount.com"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/harmonic"
TAG="$(git rev-parse --short HEAD 2>/dev/null || date +%s)"

say() { printf '\n== %s\n' "$*"; }
gcloud config set project "$PROJECT" >/dev/null

say "APIs"
gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
    artifactregistry.googleapis.com aiplatform.googleapis.com firestore.googleapis.com \
    pubsub.googleapis.com storage.googleapis.com iam.googleapis.com spanner.googleapis.com

say "Artifact Registry"
gcloud artifacts repositories describe "$REPO" --location "$REGION" >/dev/null 2>&1 ||
    gcloud artifacts repositories create "$REPO" --repository-format docker \
        --location "$REGION" --description "Harmonic images"

say "Cloud Storage bucket gs://$BUCKET"
gcloud storage buckets describe "gs://$BUCKET" >/dev/null 2>&1 ||
    gcloud storage buckets create "gs://$BUCKET" --location "$REGION" \
        --uniform-bucket-level-access

say "Firestore"
gcloud firestore databases describe --database "(default)" >/dev/null 2>&1 ||
    gcloud firestore databases create --location "$REGION" --type firestore-native

say "Spanner Graph: a free-trial instance and the database for the knowledge graph"
gcloud spanner instances describe harmonic >/dev/null 2>&1 ||
    gcloud spanner instances create harmonic --instance-type free-instance \
        --config "regional-$REGION" --description "Harmonic knowledge graph"
gcloud spanner databases describe kg --instance harmonic >/dev/null 2>&1 ||
    gcloud spanner databases create kg --instance harmonic

say "Pub/Sub topic for the demo sensor"
gcloud pubsub topics describe harmonic-demo-vibration >/dev/null 2>&1 ||
    gcloud pubsub topics create harmonic-demo-vibration

say "Service account"
gcloud iam service-accounts describe "$SA" >/dev/null 2>&1 ||
    gcloud iam service-accounts create harmonic-run --display-name "Harmonic on Cloud Run"
for role in roles/aiplatform.user roles/datastore.user roles/pubsub.editor \
            roles/logging.logWriter roles/monitoring.metricWriter; do
    gcloud projects add-iam-policy-binding "$PROJECT" --member "serviceAccount:$SA" \
        --role "$role" --condition None >/dev/null
done
gcloud storage buckets add-iam-policy-binding "gs://$BUCKET" \
    --member "serviceAccount:$SA" --role roles/storage.objectAdmin >/dev/null
# The service loads the graph at start-up and creates its tables the first time.
gcloud spanner instances add-iam-policy-binding harmonic \
    --member "serviceAccount:$SA" --role roles/spanner.databaseAdmin >/dev/null

say "The IMS real-machine demo, into the bucket"
if [ -f var/ims/test2/stamps.json ]; then
    gcloud storage rsync --recursive var/ims "gs://$BUCKET/ims"
else
    echo "var/ims not exported (python scripts/export_ims.py); the IMS demo will be absent"
fi

say "Build service account"
# New projects no longer give Cloud Build's default account (the Compute
# Engine one) the rights to read its own source upload, so the build runs as a
# dedicated account holding only what it needs.
gcloud iam service-accounts describe "$BUILD_SA" >/dev/null 2>&1 ||
    gcloud iam service-accounts create harmonic-build --display-name "Harmonic image build"
gcloud storage buckets describe "gs://${PROJECT}_cloudbuild" >/dev/null 2>&1 ||
    gcloud storage buckets create "gs://${PROJECT}_cloudbuild" --location "$REGION" \
        --uniform-bucket-level-access
gcloud storage buckets add-iam-policy-binding "gs://${PROJECT}_cloudbuild" \
    --member "serviceAccount:$BUILD_SA" --role roles/storage.objectViewer >/dev/null
gcloud artifacts repositories add-iam-policy-binding "$REPO" --location "$REGION" \
    --member "serviceAccount:$BUILD_SA" --role roles/artifactregistry.writer >/dev/null
gcloud projects add-iam-policy-binding "$PROJECT" --member "serviceAccount:$BUILD_SA" \
    --role roles/logging.logWriter --condition None >/dev/null

say "Build (Cloud Build; trains both brains, a few minutes)"
gcloud builds submit --config deploy/gcp/cloudbuild.yaml \
    --service-account "projects/$PROJECT/serviceAccounts/$BUILD_SA" \
    --substitutions "_REGION=$REGION,_REPO=$REPO,_TAG=$TAG" .

say "Deploy to Cloud Run"
access=(--no-allow-unauthenticated)
[ "$PUBLIC" = "1" ] && access=(--allow-unauthenticated)
gcloud run deploy harmonic --image "$IMAGE:$TAG" --region "$REGION" \
    --service-account "$SA" --port 8000 --cpu 2 --memory 4Gi \
    --min-instances 1 --max-instances 1 --no-cpu-throttling --timeout 3600 \
    --execution-environment gen2 \
    --add-volume "name=data,type=cloud-storage,bucket=$BUCKET" \
    --add-volume-mount "volume=data,mount-path=/data" \
    --set-env-vars "HARMONIC_DATA_DIR=/data,HARMONIC_STORE=firestore,HARMONIC_KG_BACKEND=spanner,HARMONIC_SPANNER_INSTANCE=harmonic,HARMONIC_SPANNER_DATABASE=kg,GOOGLE_GENAI_USE_VERTEXAI=1,GOOGLE_CLOUD_PROJECT=$PROJECT,GOOGLE_CLOUD_LOCATION=$REGION,HARMONIC_LLM_MODEL=$LLM_MODEL,HARMONIC_EMBED_MODEL=$EMBED_MODEL" \
    "${access[@]}"

URL="$(gcloud run services describe harmonic --region "$REGION" --format 'value(status.url)')"
say "Deployed: $URL"
if [ "$PUBLIC" != "1" ]; then
    echo "Private: open it with   gcloud run services proxy harmonic --region $REGION"
    echo "and browse http://localhost:8080 — or grant a person roles/run.invoker."
fi
echo "Live demo sensor:  python scripts/mqtt_publish.py --broker pubsub:$PROJECT \\"
echo "                       --topic harmonic-demo-vibration --family gearbox --rate 2"
