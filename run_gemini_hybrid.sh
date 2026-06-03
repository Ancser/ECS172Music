#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON_BIN:-python3}"

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
  echo "  ./run_playlist_filter_low_memory.sh"
  exit 1
fi

"$PYTHON_BIN" recommandation.py \
  --lyrics-csv data/spotify_millsongdata.csv \
  --playlist-csv dataFiltered/spotify_playlist_50percent_50item.csv \
  --max-playlists 800 \
  --max-eval-cases 800 \
  --min-playlist-len 20 \
  --holdout-k 10 \
  --stage1-mode hybrid \
  --playlist-semantics llm \
  --semantic-provider gemini \
  --semantic-model "${SEMANTIC_MODEL:-gemma-4-31b-it}" \
  --semantic-requests-per-minute "${SEMANTIC_REQUESTS_PER_MINUTE:-15}" \
  --semantic-cache dataFiltered/playlist_semantics_gemini_free_tier.jsonl \
  --semantic-max-generate "${SEMANTIC_MAX_GENERATE:-50}" \
  --lemon-stage2 \
  --lemon-weight "${LEMON_WEIGHT:-0.01}"
