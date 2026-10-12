#!/bin/sh
# Load the knowledge graph into whichever store this deployment actually uses,
# then serve.
#
# The image builds the graph into the NetworkX JSON store at image build time,
# which is what a checkout with no Docker runs against. Under compose the store
# is Neo4j, and Neo4j is a SEPARATE container with its own volume — so the
# graph built into the image is not the graph the API will query. Without this
# step a fresh `docker compose up` serves an empty graph and every diagnosis
# comes back "insufficient evidence", while a machine that happened to load
# Neo4j once by hand works fine. That is the worst kind of bug: it only appears
# on someone else's computer.
#
# The loader is idempotent, so a restart costs a few seconds and changes
# nothing.
set -e

echo "harmonic: loading the knowledge graph"
python -m kg.load --rebuild || {
    echo "harmonic: graph load failed; the API will start and report it" >&2
}

exec "$@"
