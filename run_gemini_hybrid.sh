#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ -z "${GEMINI_API_KEY:-}" ]]; then
  echo "Set your Google AI Studio key first:"
  echo '  export GEMINI_API_KEY="api_key_here"'
  exit 1
fi

if [[ ! -f "data/spotify_millsongdata.csv" ]]; then
  echo "Missing data/spotify_millsongdata.csv"
  echo "Run: python download_data.py --songs"
  exit 1
fi

if [[ ! -f "dataFiltered/spotify_playlist_50percent_50item.csv" ]]; then
  echo "Missing dataFiltered/spotify_playlist_50percent_50item.csv"
  echo "Run the playlist setup first:"
  echo "  python download_data.py --playlists"
  echo "  python playlistMarker.py --workers 2"
  echo "  python playlistFilter.py"
  exit 1
fi

python recommandation.py \
  --lyrics-csv data/spotify_millsongdata.csv \
  --playlist-csv dataFiltered/spotify_playlist_50percent_50item.csv \
  --max-playlists 800 \
  --max-eval-cases 800 \
  --min-playlist-len 20 \
  --holdout-k 10 \
  --stage1-mode hybrid \
  --playlist-semantics llm \
  --semantic-provider gemini \
  --semantic-model gemini-2.5-flash-lite \
  --semantic-cache dataFiltered/playlist_semantics_gemini_free_tier.jsonl \
  --semantic-max-generate "${SEMANTIC_MAX_GENERATE:-20}"
