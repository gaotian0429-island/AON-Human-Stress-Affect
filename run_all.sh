#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_dir"
python scripts/compute_d_isc.py
python scripts/compute_f_emotions.py
python scripts/compute_fc.py --workers "${AON_WORKERS:-16}"
python scripts/compute_g_permutation.py --workers "${AON_WORKERS:-16}"
