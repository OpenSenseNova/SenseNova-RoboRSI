#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$project_root"
exec "${ROBORSI_PYTHON:-python}" -m robots.vln_shared.cli "$@"
