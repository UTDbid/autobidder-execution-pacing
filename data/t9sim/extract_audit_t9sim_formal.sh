#!/usr/bin/env bash
set -euo pipefail

ROOT="${POMS_DATA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RAW="$ROOT/02_data_raw/t9sim"
CLEAN="$ROOT/03_data_clean/t9sim/formal_10m"
LOG="$ROOT/07_logs/audits/t9sim_formal_extract.log"
mkdir -p "$CLEAN" "$(dirname "$LOG")"
test -s "$ROOT/00_admin/audits/T9SIM_DOWNLOAD_AUDIT.json"
python3 - "$ROOT/00_admin/audits/T9SIM_DOWNLOAD_AUDIT.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
if not d.get("all_passed"):
    raise SystemExit("download audit is not all_passed")
PY

export ROOT RAW CLEAN LOG
find "$RAW" -maxdepth 1 -type f -name 't9_v10_10m_seed*.zip' -print0 | sort -z | \
  xargs -0 -P 4 -n 1 bash -c '
    set -e
    zip="$1"
    base=$(basename "$zip" .zip)
    target="$CLEAN/$base"
    echo "[$(date -Is)] TEST $zip" >>"$LOG"
    unzip -t "$zip" >/dev/null
    mkdir -p "$target"
    unzip -qo "$zip" -d "$target"
    parquet=$(find "$target" -name auctions.parquet -print -quit)
    test -s "$parquet"
    echo "[$(date -Is)] EXTRACTED $base $parquet" >>"$LOG"
  ' _

find "$CLEAN" -name auctions.parquet -printf '%p\t%s\n' | sort \
  >"$ROOT/00_admin/manifests/t9sim_formal_parquets.tsv"
test "$(wc -l <"$ROOT/00_admin/manifests/t9sim_formal_parquets.tsv")" -eq 10
echo "[$(date -Is)] ALL_EXTRACTED" >>"$LOG"
