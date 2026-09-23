#!/usr/bin/env bash
# Lays the generated code and conformance suite from a local jennah-api checkout
# over this one, exactly as CI does. None of it is committed (see .gitignore).
#
# Usage: scripts/dev-generate.sh   (JENNAH_API defaults to ../jennah-api)
set -euo pipefail
cd "$(dirname "$0")/.."
API=${JENNAH_API:-../jennah-api}
(cd "$API" && buf generate)
"$API/ci/python/assemble.sh" .
