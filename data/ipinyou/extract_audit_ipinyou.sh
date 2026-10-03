#!/usr/bin/env bash
set -euo pipefail

ROOT="${POMS_DATA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RAW="$ROOT/02_data_raw/ipinyou/ipinyou.contest.dataset.7z"
TARGET="$ROOT/02_data_raw/ipinyou/extracted"
AUDIT="$ROOT/07_logs/audits"
EXPECTED=6304754182
mkdir -p "$TARGET" "$AUDIT"
actual=$(stat -c %s "$RAW")
test "$actual" -eq "$EXPECTED"
sha256sum "$RAW" | tee "$AUDIT/ipinyou_archive_sha256.txt"
7z t "$RAW" >"$AUDIT/ipinyou_7z_test.log"
7z x -y -o"$TARGET" "$RAW" >"$AUDIT/ipinyou_7z_extract.log"
dataset=$(find "$TARGET" -maxdepth 2 -type f -name files.md5 -printf '%h\n' | head -1)
test -n "$dataset"
(cd "$dataset" && md5sum -c files.md5) >"$AUDIT/ipinyou_internal_md5.log" 2>&1
find "$dataset" -maxdepth 2 -type f -printf '%p\t%s\n' | sort \
  >"$ROOT/00_admin/manifests/ipinyou_raw_inventory.tsv"
printf '%s\n' "$dataset" >"$ROOT/00_admin/manifests/ipinyou_dataset_root.txt"
echo "IPINYOU_EXTRACT_AUDIT_PASSED $dataset"
