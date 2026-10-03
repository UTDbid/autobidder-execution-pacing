#!/usr/bin/env bash
set -euo pipefail

ROOT="${POMS_DATA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RAW="$ROOT/02_data_raw/ipinyou"
LOG="$ROOT/07_logs/downloads/ipinyou_season2_download.log"
FILE="$RAW/ipinyou.contest.dataset-season2.zip"
URL="https://ndownloader.figshare.com/files/10082688"
EXPECTED_SIZE="3610249162"
EXPECTED_MD5="9ee18bf8c4dd19d1c41e6e77088367f9"

mkdir -p "$RAW" "$(dirname "$LOG")"
echo "[$(date -Is)] START Figshare season2 CC0" >> "$LOG"
wget -c --retry-connrefused --waitretry=5 --timeout=60 --read-timeout=60 \
  --tries=0 --no-verbose -O "$FILE" "$URL" >> "$LOG" 2>&1

actual_size="$(stat -c '%s' "$FILE")"
actual_md5="$(md5sum "$FILE" | awk '{print $1}')"
if [[ "$actual_size" != "$EXPECTED_SIZE" || "$actual_md5" != "$EXPECTED_MD5" ]]; then
  echo "[$(date -Is)] FAIL size=$actual_size md5=$actual_md5" >> "$LOG"
  exit 2
fi
echo "[$(date -Is)] VERIFIED size=$actual_size md5=$actual_md5" >> "$LOG"
