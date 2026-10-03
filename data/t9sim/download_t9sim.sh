#!/usr/bin/env bash
set -euo pipefail

ROOT="${POMS_DATA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RAW="$ROOT/02_data_raw/t9sim"
LOG="$ROOT/07_logs/downloads/t9sim_download.log"
RECORD="$RAW/zenodo_record_21533031.json"
LIST="$ROOT/00_admin/manifests/t9sim_downloads.tsv"

mkdir -p "$RAW" "$(dirname "$LOG")" "$(dirname "$LIST")"

curl -fsSL --retry 20 --retry-all-errors \
  https://zenodo.org/api/records/21533031 -o "$RECORD"

python3 - "$RECORD" "$LIST" <<'PY'
import json
import sys

record_path, list_path = sys.argv[1:]
record = json.load(open(record_path, encoding="utf-8"))
with open(list_path, "w", encoding="utf-8") as out:
    for item in sorted(record["files"], key=lambda row: row["key"]):
        url = item["links"].get("content") or item["links"]["self"]
        checksum = item["checksum"].split(":", 1)[1]
        out.write(f'{item["key"]}\t{item["size"]}\t{checksum}\t{url}\n')
PY

export ROOT RAW LOG
# Ten independent 10M archives dominate transfer time.  Each wget is resumable;
# one worker per archive improves utilization on the high-latency Zenodo path.
cut -f1,4 "$LIST" | xargs -P 10 -n 2 bash -c '
  name="$1"
  url="$2"
  echo "[$(date -Is)] START $name" >> "$LOG"
  wget -c --retry-connrefused --waitretry=5 --timeout=60 --read-timeout=60 \
    --tries=0 --no-verbose -O "$RAW/$name" "$url" >> "$LOG" 2>&1
  echo "[$(date -Is)] DONE $name" >> "$LOG"
' _

python3 - "$RAW" "$LIST" "$ROOT/00_admin/audits/T9SIM_DOWNLOAD_AUDIT.json" <<'PY'
import hashlib
import json
import pathlib
import sys

raw = pathlib.Path(sys.argv[1])
rows = []
ok = True
for line in open(sys.argv[2], encoding="utf-8"):
    name, expected_size, expected_md5, _ = line.rstrip("\n").split("\t")
    path = raw / name
    size = path.stat().st_size if path.exists() else None
    digest = None
    if size == int(expected_size):
        h = hashlib.md5()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
                h.update(block)
        digest = h.hexdigest()
    passed = size == int(expected_size) and digest == expected_md5
    ok &= passed
    rows.append({"file": name, "size": size, "expected_size": int(expected_size),
                 "md5": digest, "expected_md5": expected_md5, "passed": passed})
payload = {"record": 21533031, "doi": "10.5281/zenodo.21533031",
           "license": "CC-BY-4.0", "all_passed": ok, "files": rows}
pathlib.Path(sys.argv[3]).write_text(json.dumps(payload, indent=2), encoding="utf-8")
if not ok:
    raise SystemExit("T9Sim download integrity audit failed")
PY

echo "[$(date -Is)] ALL_FILES_VERIFIED" >> "$LOG"
