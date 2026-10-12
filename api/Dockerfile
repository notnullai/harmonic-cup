# The API image carries the engine, so it is a two-stage build.
#
# `sim/artifacts/` is generated and not committed: a trained brain is a build
# artefact, not source. So the builder compiles the engine, mints, encodes,
# trains and commissions, and the runtime stage copies out the shared library
# and the brain. The alternative — shipping a committed brain — would put a
# binary blob in git that nothing could regenerate reproducibly.
#
# The runtime stage has no compiler and no build dependencies.

# --- builder ------------------------------------------------------------------
FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
# The simulator's dependencies; the brain cannot be built without them.
# pyyaml comes in through sim/scenario.py, which sim/cli.py imports.
RUN pip install --no-cache-dir "numpy>=1.26" "scipy>=1.11" "pyyaml>=6.0"

COPY CMakeLists.txt /src/
COPY engine /src/engine
COPY tests /src/tests
COPY sim /src/sim
COPY scripts /src/scripts
COPY profiles /src/profiles
# build_brain.py reads the family catalogue (profiles/families.json) through kg.
COPY kg /src/kg

RUN cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build -j \
        --target harmonic_rig mint_brain harmonic_sim_words harmonic_train_cli commission

# The documented chain, not a shortcut: the words come from the real C encoder
# and the weights from the real trainer, or the brain would not match the
# runtime byte for byte.
# Every family in the catalogue, so adding one is a data change, never an edit
# here (a family missing from the image fails every run of that type).
RUN set -e; python scripts/build_brain.py; \
    for f in $(python -c "import json; print(' '.join(k for k in json.load(open('profiles/families.json'))['families'] if k != 'pump'))"); do \
        python scripts/build_brain.py --family "$f"; \
    done; \
    mkdir -p /src/brains && cp sim/artifacts/*.harmonic /src/brains/

# --- the web app ----------------------------------------------------------------
# Built here and copied in as static files: the runtime image carries no Node.
FROM node:22-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json /web/
RUN npm ci --no-audit --no-fund
COPY web /web
# The try site's benchmark page is rendered from this at build time
# (web/scripts/gen-benchmark.mjs reads ../../docs/BENCHMARK.md).
COPY docs/BENCHMARK.md /docs/BENCHMARK.md
RUN npm run build

# --- runtime ------------------------------------------------------------------
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY api/requirements.txt /app/api/requirements.txt
RUN pip install --no-cache-dir -r /app/api/requirements.txt

COPY --from=builder /src/build/libharmonic_rig.so /usr/local/lib/libharmonic_rig.so
# Every machine family: the commissioned demo brains, and the trained
# networks the commissioning wizard grafts onto a new machine.
COPY --from=builder /src/brains/ /app/sim/artifacts/
# The tools the wizard runs to commission a machine.
COPY --from=builder /src/build/mint_brain /src/build/commission /app/build/
COPY profiles /app/profiles
COPY scripts /app/scripts

COPY api /app/api
COPY sim /app/sim
COPY kg /app/kg
COPY rag /app/rag
COPY docs/GENAI_SCHEMA.json /app/docs/GENAI_SCHEMA.json
COPY --from=web /web/dist /app/web/dist

# Built at image build time and loaded from outside the source tree.
ENV HARMONIC_RIG_LIB=/usr/local/lib/libharmonic_rig.so
ENV PYTHONUNBUFFERED=1

# Extract the triples and build the NetworkX store into the image, so a
# container run without Neo4j still has a graph.
RUN python -m rag.ingest --check \
    && python -m kg.extract \
    && python -m kg.load --backend networkx --rebuild

# Under compose the store is Neo4j, in a separate container with its own
# volume — a different store from the one built above. The entrypoint loads
# whichever one this deployment actually resolves to.
COPY api/entrypoint.sh /usr/local/bin/harmonic-entrypoint
RUN chmod +x /usr/local/bin/harmonic-entrypoint

EXPOSE 8000
ENTRYPOINT ["/usr/local/bin/harmonic-entrypoint"]
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
