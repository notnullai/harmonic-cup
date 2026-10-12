#!/usr/bin/env bash
# Deploy Harmonic's public "Try it" service to Google Cloud (HARM-S10-T02).
#
#   PROJECT=my-project REGION=asia-southeast1 deploy/gcp/deploy_try.sh
#
# The same image as the console (deploy/gcp/deploy.sh), run as a SEPARATE Cloud
# Run service with HARMONIC_MODE=try: only /try is served, no fleet, no console,
# no Gemini. It meets the deployment conditions in docs/TRY_IT_SPEC.md:
#   1. its own service, "harmonic-try";
#   2. its own bucket, gs://$PROJECT-harmonic-try, for HARMONIC_DATA_DIR, with
#      two lifecycle rules as a backstop to the app's own sweep (api/try_mode.py) —
#      the app sweeps only while an instance is up, and this service scales to
#      zero: LIFECYCLE_DAYS (default 8) deletes anything under try/, a few days
#      after the app's own 7-day deletion of an unkept run; KEPT_LIFECYCLE_DAYS
#      (default 91) deletes anything under kept/, a day after the app's own
#      90-day deletion of a kept one. Two prefixes, not one nested in the
#      other, because a GCS lifecycle condition can only include a prefix,
#      never exclude one — kept/ must sit outside try/ or the shorter rule
#      would also catch it;
#   3. max-instances=1: rate limits, queue and disk accounting live in memory;
#   4. the in-image NetworkX graph, never the console's Spanner graph, and its
#      own service account "harmonic-try" that can reach that one bucket and
#      nothing else — no Spanner, Vertex, Firestore or Pub/Sub role;
#   5. HARMONIC_TRY_DISK_QUOTA_MB below the instance memory (Cloud Run's local
#      disk is RAM, and the bucket mount stages files being written there too),
#      and HARMONIC_CONTACT_URL set;
#   6. the forwarded-address smoke test is run by hand at every deploy (see
#      docs/DEPLOY_GCP.md).
#
# Idempotent: every step checks before it creates. The image is built only if
# the tag for the current commit is missing (BUILD=1 forces a build). It reuses
# the Artifact Registry repository and the build account deploy.sh created, and
# never changes the console service "harmonic" or its bucket.
#
# Access: PRIVATE by default. PUBLIC=1 switches off the invoker IAM check on
# this one service (the organisation refuses allUsers grants) — the owner's
# decision, never the default. A plain re-run (PUBLIC unset) never touches
# that setting either way; UNPUBLIC=1 is the separate, explicit way back.
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
UNPUBLIC="${UNPUBLIC:-0}"
BUILD="${BUILD:-0}"
LIFECYCLE_DAYS="${LIFECYCLE_DAYS:-8}"
KEPT_LIFECYCLE_DAYS="${KEPT_LIFECYCLE_DAYS:-91}"
MEMORY="${MEMORY:-4Gi}"
DISK_QUOTA_MB="${DISK_QUOTA_MB:-2048}"          # below MEMORY, see condition 5
CONTACT_URL="${CONTACT_URL:-mailto:contact@khalm.ai}"
SERVICE=harmonic-try
REPO=harmonic
BUCKET="${PROJECT}-harmonic-try"
SA="harmonic-try@${PROJECT}.iam.gserviceaccount.com"
BUILD_SA="harmonic-build@${PROJECT}.iam.gserviceaccount.com"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/harmonic"
TAG="$(git rev-parse --short HEAD 2>/dev/null || date +%s)"

say() { printf '\n== %s\n' "$*"; }
gcloud config set project "$PROJECT" >/dev/null

say "APIs"
gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
    artifactregistry.googleapis.com storage.googleapis.com iam.googleapis.com

say "Cloud Storage bucket gs://$BUCKET (try runs only)"
gcloud storage buckets describe "gs://$BUCKET" >/dev/null 2>&1 ||
    gcloud storage buckets create "gs://$BUCKET" --location "$REGION" \
        --uniform-bucket-level-access --public-access-prevention
lifecycle="$(mktemp)"
trap 'rm -f "$lifecycle"' EXIT
# Two rules, two prefixes (HARM-S10-T05): try/ (an unkept run) at
# LIFECYCLE_DAYS, kept/ (a consented, finished run — api/try_mode.py
# KEPT_DIR) at KEPT_LIFECYCLE_DAYS. matchesPrefix only ever *includes*, so
# kept/ has to be a prefix of its own, never nested under try/, or this
# first rule would delete it too.
printf '{"rule":[
  {"action":{"type":"Delete"},"condition":{"age":%d,"matchesPrefix":["try/"]}},
  {"action":{"type":"Delete"},"condition":{"age":%d,"matchesPrefix":["kept/"]}}
]}\n' "$LIFECYCLE_DAYS" "$KEPT_LIFECYCLE_DAYS" >"$lifecycle"
gcloud storage buckets update "gs://$BUCKET" --lifecycle-file "$lifecycle" >/dev/null

say "Service account (this bucket only)"
gcloud iam service-accounts describe "$SA" >/dev/null 2>&1 ||
    gcloud iam service-accounts create harmonic-try --display-name "Harmonic Try it on Cloud Run"
gcloud storage buckets add-iam-policy-binding "gs://$BUCKET" \
    --member "serviceAccount:$SA" --role roles/storage.objectAdmin >/dev/null

say "Image $IMAGE:$TAG"
if [ "$BUILD" != "1" ] &&
    gcloud artifacts docker images describe "$IMAGE:$TAG" >/dev/null 2>&1; then
    echo "already built"
else
    gcloud artifacts repositories describe "$REPO" --location "$REGION" >/dev/null 2>&1 || {
        echo "no Artifact Registry repository '$REPO': run deploy/gcp/deploy.sh once first" >&2
        exit 1
    }
    gcloud builds submit --config deploy/gcp/cloudbuild.yaml \
        --service-account "projects/$PROJECT/serviceAccounts/$BUILD_SA" \
        --substitutions "_REGION=$REGION,_REPO=$REPO,_TAG=$TAG" .
fi

say "Deploy to Cloud Run ($SERVICE)"
# Neither --allow-unauthenticated nor --no-allow-unauthenticated is passed:
# the deploy leaves the service's access settings exactly as they are, so a
# plain re-run never changes whether the service is public.
# Scales to zero; CPU stays allocated while an instance is up, since a run
# is worked on in a background thread after its request has returned.
deploy_args=(
    "$SERVICE" --image "$IMAGE:$TAG" --region "$REGION"
    --service-account "$SA" --port 8000 --cpu 2 --memory "$MEMORY"
    --min-instances 0 --max-instances 1 --no-cpu-throttling --timeout 900
    --execution-environment gen2
    --add-volume "name=data,type=cloud-storage,bucket=$BUCKET"
    --add-volume-mount "volume=data,mount-path=/data"
    --set-env-vars "HARMONIC_MODE=try,HARMONIC_DATA_DIR=/data,HARMONIC_KG_BACKEND=networkx,HARMONIC_TRY_DISK_QUOTA_MB=$DISK_QUOTA_MB,HARMONIC_CONTACT_URL=$CONTACT_URL"
)
gcloud run deploy "${deploy_args[@]}"

# Public access without an allUsers grant: the project's organisation enforces
# domain-restricted sharing (iam.allowedPolicyMemberDomains), which refuses
# allUsers. Cloud Run's invoker IAM check, switched off on this one service,
# is the documented way to serve it publicly under that policy; nothing else
# in the organisation or project changes.
if [ "$PUBLIC" = "1" ]; then
    say "Public: invoker IAM check off on $SERVICE only"
    gcloud run services update "$SERVICE" --region "$REGION" --no-invoker-iam-check >/dev/null
elif [ "$UNPUBLIC" = "1" ]; then
    say "Private again: invoker IAM check on for $SERVICE"
    gcloud run services update "$SERVICE" --region "$REGION" --invoker-iam-check >/dev/null
    gcloud run services remove-iam-policy-binding "$SERVICE" --region "$REGION" \
        --member allUsers --role roles/run.invoker >/dev/null 2>&1 || true
fi

URL="$(gcloud run services describe "$SERVICE" --region "$REGION" --format 'value(status.url)')"
say "Deployed: $URL/try/"
if [ "$PUBLIC" != "1" ]; then
    echo "Private unless made public separately. Call it with:  curl -H \"Authorization: Bearer \$(gcloud auth print-identity-token)\" $URL/try/"
fi
echo "Run the forwarded-address smoke test now (docs/DEPLOY_GCP.md, 'Try it')."
