#!/usr/bin/env bash
set -euo pipefail

ROOT="${POMS_DATA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RAW="$ROOT/02_data_raw/ipinyou"
LOG="$ROOT/07_logs/downloads/ipinyou_full_download.log"
FILE="$RAW/ipinyou.contest.dataset.7z"
URL="https://www.dropbox.com/scl/fi/dlam6yqswcwu4x0dnooig/ipinyou.contest.dataset.7z?rlkey=h7q2ey4f7u8n0w7y8euo3zx20&dl=1"
EXPECTED_SIZE="6304754182"

mkdir -p "$RAW" "$(dirname "$LOG")"
echo "[$(date -Is)] START authors-linked Dropbox full archive" >> "$LOG"
wget -c --retry-connrefused --waitretry=5 --timeout=60 --read-timeout=60 \
  --tries=0 --no-verbose -O "$FILE" "$URL" >> "$LOG" 2>&1

actual_size="$(stat -c '%s' "$FILE")"
if [[ "$actual_size" != "$EXPECTED_SIZE" ]]; then
  echo "[$(date -Is)] FAIL size=$actual_size expected=$EXPECTED_SIZE" >> "$LOG"
  exit 2
fi
sha256sum "$FILE" > "$ROOT/00_admin/manifests/ipinyou_full_archive.sha256"
7z t "$FILE" > "$ROOT/07_logs/audits/ipinyou_full_7z_test.log"
echo "[$(date -Is)] VERIFIED size=$actual_size archive_test=passed" >> "$LOG"
