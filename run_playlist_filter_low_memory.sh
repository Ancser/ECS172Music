#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ ! -f "data/spotify_millsongdata.csv" ]]; then
  echo "Missing data/spotify_millsongdata.csv"
  echo "Run: python download_data.py --songs"
  exit 1
fi

if ! ls data/mpd.slice.*.json >/dev/null 2>&1; then
  echo "Missing MPD playlist slices in data/"
  echo "Run: python download_data.py --playlists"
  exit 1
fi

"$PYTHON_BIN" playlistFilter.py \
  --lyrics-csv data/spotify_millsongdata.csv \
  --mpd-path data \
  --out-dir dataFiltered \
  --max-files "${MAX_FILES:-0}"
