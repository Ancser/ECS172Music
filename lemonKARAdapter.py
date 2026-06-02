#!/usr/bin/env python3
"""Neural adapter over CF, metadata, LEMON, and KAR features.

This is a learned alignment layer for the LEMON+KAR smoke prototype. It trains
on 80% of evaluation playlists, then reports train/test/all-799 ranking metrics.
The adapter does not call the LLM when the Qwen song-semantic cache is complete.
"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import lemon
import lemonKAR
import recommandation as rec


ROOT = Path(__file__).resolve().parent
DEFAULT_PLAYLIST_CSV = ROOT / "dataFiltered" / "playlist_50%_50c_799.csv"
DEFAULT_SONG_SEMANTIC_CSV = ROOT / "dataFiltered" / "song_semantics_fine_keywords_800_heuristic.csv"
DEFAULT_SONG_LLM_CACHE = ROOT / "dataFiltered" / "song_semantics_fine_keywords_qwen.jsonl"


@dataclass
class FeaturePack:
    features_by_case: list[np.ndarray]
    labels_by_case: list[np.ndarray]
    candidates_by_case: list[list[str]]
    mean: np.ndarray
    std: np.ndarray
    feature_names: list[str]


class AdapterMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a neural adapter for LEMON+KAR reranking")
    parser.add_argument("--lyrics-csv", type=Path, default=rec.DEFAULT_LYRICS_CSV)
    parser.add_argument("--playlist-csv", type=Path, default=DEFAULT_PLAYLIST_CSV)
    parser.add_argument("--song-semantic-csv", type=Path, default=DEFAULT_SONG_SEMANTIC_CSV)
    parser.add_argument("--song-semantic-source", choices=["csv", "qwen"], default="qwen")
    parser.add_argument("--song-llm-cache", type=Path, default=DEFAULT_SONG_LLM_CACHE)
    parser.add_argument("--song-llm-max-generate", type=int, default=0, help="0 means generate every missing eval song")
    parser.add_argument("--song-llm-lyrics-chars", type=int, default=220)
    parser.add_argument("--song-llm-max-new-tokens", type=int, default=130)
    parser.add_argument("--song-llm-batch-size", type=int, default=4)
    parser.add_argument("--semantic-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--semantic-model-cache-dir", type=Path, default=rec.DEFAULT_LLM_CACHE_DIR)
    parser.add_argument("--semantic-device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--max-playlists", type=int, default=0, help="0 means no cap")
    parser.add_argument("--max-eval-cases", type=int, default=0, help="0 means no cap")
    parser.add_argument("--min-playlist-len", type=int, default=20)
    parser.add_argument("--holdout-k", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--stage1-force-ratio", type=float, default=0.25)
    parser.add_argument("--stage1-recent-window", type=int, default=10)
    parser.add_argument("--lemon-recent-window", type=int, default=8)
    parser.add_argument("--kar-recent-window", type=int, default=12)
    parser.add_argument("--cold-start-threshold", type=int, default=1)
    parser.add_argument("--cf-neighbors", type=int, default=100)
    parser.add_argument("--cf-weight", type=float, default=0.35)
    parser.add_argument("--artist-weight", type=float, default=0.10)
    parser.add_argument("--pop-weight", type=float, default=0.10)
    parser.add_argument("--lemon-weight", type=float, default=0.01)
    parser.add_argument("--kar-weight", type=float, default=0.01)
    parser.add_argument("--kar-dim", type=int, default=32)
    parser.add_argument("--kar-max-features", type=int, default=4000)
    parser.add_argument("--kar-lyrics-chars", type=int, default=260)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--loss", choices=["pairwise", "bce"], default="pairwise")
    parser.add_argument("--negatives-per-positive", type=int, default=20)
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=172)
    return parser.parse_args()


def split_indices(count: int, train_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    indices = list(range(count))
    rng = random.Random(seed)
    rng.shuffle(indices)
    train_count = max(1, min(count - 1, int(round(count * train_ratio))))
    train = sorted(indices[:train_count])
    test = sorted(indices[train_count:])
    return train, test


def vector_product_features(
    user_vector: np.ndarray | None,
    item_vector: np.ndarray | None,
    dim: int,
) -> np.ndarray:
    if user_vector is None or item_vector is None:
        return np.zeros(dim, dtype=np.float32)
    product = np.asarray(user_vector[:dim] * item_vector[:dim], dtype=np.float32)
    if len(product) < dim:
        padded = np.zeros(dim, dtype=np.float32)
        padded[: len(product)] = product
        return padded
    return product


def build_feature_pack(
    prepared_cases: list[rec.PreparedCase],
    cases: list[rec.EvalCase],
    train_indices: list[int],
    lemon_user_vectors: dict[str, dict[str, float]],
    lemon_song_vectors: dict[str, dict[str, float]],
    kar_user_vectors: dict[str, np.ndarray],
    kar_item_vectors: dict[str, np.ndarray],
    kar_dim: int,
) -> FeaturePack:
    features_by_case: list[np.ndarray] = []
    labels_by_case: list[np.ndarray] = []
    candidates_by_case: list[list[str]] = []
    feature_names = [
        "cf",
        "artist",
        "popularity",
        "lemon",
        "kar",
        "cf_artist",
        "cf_popularity",
        "cf_lemon",
        "cf_kar",
        "artist_kar",
        "popularity_kar",
        "lemon_kar",
    ] + [f"kar_product_{idx}" for idx in range(kar_dim)]

    for prepared, case in zip(prepared_cases, cases):
        lemon_norm, kar_norm = lemonKAR.combined_scores_for_case(
            prepared,
            lemon_user_vectors,
            lemon_song_vectors,
            kar_user_vectors,
            kar_item_vectors,
        )
        user_kar = kar_user_vectors.get(prepared.playlist_id)
        truth = set(case.heldout)
        rows: list[np.ndarray] = []
        labels: list[float] = []
        for song_id in prepared.candidates:
            cf = float(prepared.cf_norm.get(song_id, 0.0))
            artist = float(prepared.artist_norm.get(song_id, 0.0))
            pop = float(prepared.pop_norm.get(song_id, 0.0))
            lemon_score = float(lemon_norm.get(song_id, 0.0))
            kar_score = float(kar_norm.get(song_id, 0.0))
            interactions = np.asarray(
                [
                    cf * artist,
                    cf * pop,
                    cf * lemon_score,
                    cf * kar_score,
                    artist * kar_score,
                    pop * kar_score,
                    lemon_score * kar_score,
                ],
                dtype=np.float32,
            )
            product = vector_product_features(user_kar, kar_item_vectors.get(song_id), kar_dim)
            rows.append(
                np.concatenate(
                    [
                        np.asarray([cf, artist, pop, lemon_score, kar_score], dtype=np.float32),
                        interactions,
                        product,
                    ]
                )
            )
            labels.append(1.0 if song_id in truth else 0.0)
        features_by_case.append(np.vstack(rows).astype(np.float32))
        labels_by_case.append(np.asarray(labels, dtype=np.float32))
        candidates_by_case.append(list(prepared.candidates))

    train_features = np.vstack([features_by_case[idx] for idx in train_indices])
    mean = train_features.mean(axis=0).astype(np.float32)
    std = train_features.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    normalized = [((features - mean) / std).astype(np.float32) for features in features_by_case]
    return FeaturePack(normalized, labels_by_case, candidates_by_case, mean, std, feature_names)


def select_feature_pack(feature_pack: FeaturePack, selected_names: list[str]) -> FeaturePack:
    selected = set(selected_names)
    indices = [idx for idx, name in enumerate(feature_pack.feature_names) if name in selected]
    if not indices:
        raise ValueError("No feature columns selected for adapter.")
    return FeaturePack(
        features_by_case=[features[:, indices].astype(np.float32) for features in feature_pack.features_by_case],
        labels_by_case=feature_pack.labels_by_case,
        candidates_by_case=feature_pack.candidates_by_case,
        mean=feature_pack.mean[indices],
        std=feature_pack.std[indices],
        feature_names=[feature_pack.feature_names[idx] for idx in indices],
    )


def ablation_feature_sets(feature_pack: FeaturePack) -> list[tuple[str, list[str]]]:
    cf = ["cf"]
    metadata = ["artist", "popularity", "cf_artist", "cf_popularity"]
    lemon_features = ["lemon", "cf_lemon"]
    kar_score_features = ["kar", "cf_kar", "artist_kar", "popularity_kar"]
    kar_products = [name for name in feature_pack.feature_names if name.startswith("kar_product_")]
    return [
        ("Adapter CF only", cf),
        ("Adapter CF + metadata", cf + metadata),
        ("Adapter CF + metadata + LEMON", cf + metadata + lemon_features),
        ("Adapter CF + metadata + KAR score", cf + metadata + kar_score_features),
        ("Adapter CF + metadata + KAR vector", cf + metadata + kar_products),
        ("Adapter semantic only LEMON + KAR", ["lemon", "kar", "lemon_kar"] + kar_products),
        ("Adapter CF + metadata + LEMON + KAR all", list(feature_pack.feature_names)),
    ]


def train_adapter(
    feature_pack: FeaturePack,
    train_indices: list[int],
    args: argparse.Namespace,
    label: str = "Neural adapter",
) -> AdapterMLP:
    x_train = np.vstack([feature_pack.features_by_case[idx] for idx in train_indices])
    y_train = np.concatenate([feature_pack.labels_by_case[idx] for idx in train_indices])
    input_dim = x_train.shape[1]
    positives = float(y_train.sum())
    negatives = float(len(y_train) - positives)
    pos_weight = negatives / max(1.0, positives)

    device = torch.device("cuda" if torch.cuda.is_available() and args.semantic_device == "cuda" else "cpu")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    model = AdapterMLP(input_dim, args.hidden_dim, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    rec.print_section(f"Training {label}")
    print(f"  train rows:         {len(y_train):,}")
    print(f"  positives:          {int(positives):,}")
    print(f"  pos_weight:         {pos_weight:.2f}")
    print(f"  input_dim:          {input_dim}")
    print(f"  feature_count:      {len(feature_pack.feature_names)}")
    print(f"  device:             {device}")
    print(f"  loss:               {args.loss}")

    model.train()
    if args.loss == "bce":
        dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device))
        for epoch in range(1, args.epochs + 1):
            total_loss = 0.0
            total_rows = 0
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach().cpu()) * len(batch_y)
                total_rows += len(batch_y)
            if epoch == 1 or epoch == args.epochs or epoch % 5 == 0:
                print(f"  epoch {epoch:>3}/{args.epochs} | loss={total_loss / max(1, total_rows):.5f}")
        return model

    rng = random.Random(args.seed)
    pair_pos: list[np.ndarray] = []
    pair_neg: list[np.ndarray] = []
    for idx in train_indices:
        labels = feature_pack.labels_by_case[idx]
        positives_idx = np.flatnonzero(labels > 0.0)
        negatives_idx = np.flatnonzero(labels <= 0.0)
        if len(positives_idx) == 0 or len(negatives_idx) == 0:
            continue
        for pos_idx in positives_idx:
            for _ in range(max(1, args.negatives_per_positive)):
                neg_idx = int(rng.choice(negatives_idx.tolist()))
                pair_pos.append(feature_pack.features_by_case[idx][pos_idx])
                pair_neg.append(feature_pack.features_by_case[idx][neg_idx])

    x_pos = np.vstack(pair_pos).astype(np.float32)
    x_neg = np.vstack(pair_neg).astype(np.float32)
    print(f"  training pairs:     {len(x_pos):,}")
    pair_dataset = TensorDataset(torch.from_numpy(x_pos), torch.from_numpy(x_neg))
    pair_loader = DataLoader(pair_dataset, batch_size=args.batch_size, shuffle=True)
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        total_rows = 0
        for batch_pos, batch_neg in pair_loader:
            batch_pos = batch_pos.to(device)
            batch_neg = batch_neg.to(device)
            optimizer.zero_grad(set_to_none=True)
            pos_scores = model(batch_pos)
            neg_scores = model(batch_neg)
            loss = torch.nn.functional.softplus(-(pos_scores - neg_scores)).mean()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(batch_pos)
            total_rows += len(batch_pos)
        if epoch == 1 or epoch == args.epochs or epoch % 5 == 0:
            print(f"  epoch {epoch:>3}/{args.epochs} | pairwise_loss={total_loss / max(1, total_rows):.5f}")
    return model


def adapter_rankings(
    model: AdapterMLP,
    feature_pack: FeaturePack,
    indices: list[int],
    device_name: str,
) -> list[list[str]]:
    device = torch.device("cuda" if torch.cuda.is_available() and device_name == "cuda" else "cpu")
    model.eval()
    rankings: list[list[str]] = []
    with torch.no_grad():
        for idx in indices:
            features = torch.from_numpy(feature_pack.features_by_case[idx]).to(device)
            scores = model(features).detach().cpu().numpy()
            candidates = feature_pack.candidates_by_case[idx]
            scored = sorted(zip(scores.tolist(), candidates), key=lambda item: (-item[0], item[1]))
            rankings.append([song_id for _, song_id in scored])
    return rankings


def train_and_rank_adapter(
    label: str,
    feature_pack: FeaturePack,
    train_indices: list[int],
    all_indices: list[int],
    args: argparse.Namespace,
) -> list[list[str]]:
    model = train_adapter(feature_pack, train_indices, args, label)
    return adapter_rankings(model, feature_pack, all_indices, args.semantic_device)


def subset(items: list, indices: list[int]) -> list:
    return [items[idx] for idx in indices]


def evaluate_subset(
    label: str,
    rankings: list[list[str]],
    cases: list[rec.EvalCase],
    top_k: int,
) -> tuple[str, float, float, float]:
    recall, ndcg, proxy = rec.evaluate_rankings(rankings, cases, top_k)
    return label, recall, ndcg, proxy


def main() -> None:
    args = parse_args()
    started = time.time()
    random.seed(args.seed)
    np.random.seed(args.seed)

    rec.print_section("LEMON + KAR neural adapter inputs")
    print(f"  lyrics_csv:        {args.lyrics_csv}")
    print(f"  playlist_csv:      {args.playlist_csv}")
    print(f"  song_source:       {args.song_semantic_source}")
    print(f"  train_ratio:       {args.train_ratio:.2f}")
    print(f"  epochs:            {args.epochs}")

    songs = rec.load_lyrics_csv(args.lyrics_csv)
    playlists = rec.load_playlist_csv(args.playlist_csv, songs, args.max_playlists)
    cases = rec.split_playlists(playlists, args.min_playlist_len, args.holdout_k)
    if args.max_eval_cases and len(cases) > args.max_eval_cases:
        cases = cases[: args.max_eval_cases]
    if not cases:
        raise SystemExit("No evaluation playlists after filtering.")

    train_indices, test_indices = split_indices(len(cases), args.train_ratio, args.seed)
    all_indices = list(range(len(cases)))

    eval_song_ids = {song_id for case in cases for song_id in case.observed + case.heldout}
    songs = {song_id: song for song_id, song in songs.items() if song_id in eval_song_ids}
    catalog = set(songs)
    popularity = rec.popularity_counts(cases)

    rec.print_section("Semantic feature preparation")
    if args.song_semantic_source == "qwen":
        song_profiles = lemon.build_qwen_song_profiles(
            songs=songs,
            cache_path=args.song_llm_cache,
            model_name=args.semantic_model,
            model_cache_dir=args.semantic_model_cache_dir,
            device=args.semantic_device,
            lyrics_chars=args.song_llm_lyrics_chars,
            max_new_tokens=args.song_llm_max_new_tokens,
            max_generate=args.song_llm_max_generate,
            batch_size=args.song_llm_batch_size,
        )
    else:
        song_profiles = lemon.build_song_profiles(songs, args.song_semantic_csv)
    lemon_song_vectors = {song_id: lemon.profile_to_vector(profile) for song_id, profile in song_profiles.items()}
    lemon_user_vectors, alphas = lemon.build_user_vectors(cases, lemon_song_vectors, args.lemon_recent_window)
    print(f"  LEMON vectors:      {sum(1 for v in lemon_song_vectors.values() if v):,}/{len(lemon_song_vectors):,}")
    print(f"  LEMON alpha avg:    {sum(alphas) / len(alphas):.3f}")

    rec.print_section("KAR-style knowledge encoding")
    kar_item_vectors, kar_user_vectors, explained, vocab_size = lemonKAR.build_kar_vectors(
        cases=cases,
        songs=songs,
        song_profiles=song_profiles,
        dimensions=args.kar_dim,
        lyrics_chars=args.kar_lyrics_chars,
        recent_window=args.kar_recent_window,
        max_features=args.kar_max_features,
        random_state=args.seed,
    )
    print(f"  KAR encoder:        TF-IDF + TruncatedSVD")
    print(f"  KAR dim:            {min(args.kar_dim, len(next(iter(kar_item_vectors.values()), []))):,}")
    print(f"  KAR vocabulary:     {vocab_size:,}")
    print(f"  KAR explained var:  {explained:.3f}")

    rec.print_section("Algorithm input")
    print(f"  eval playlists:     {len(cases):,}")
    print(f"  train playlists:    {len(train_indices):,}")
    print(f"  test playlists:     {len(test_indices):,}")
    print(f"  eval songs:         {len(songs):,}")
    print(f"  heldout per list:   {args.holdout_k}")

    rec.print_section("Building co-occurrence CF neighbors")
    cf_neighbors = rec.build_cf_neighbors(cases, args.cf_neighbors)
    print(f"  songs with neighbors:  {len(cf_neighbors):,}")
    print(f"  stored neighbor edges: {sum(len(items) for items in cf_neighbors.values()):,}")
    rec.print_stage1_same_artist_grid(cases, songs, cf_neighbors, popularity, args.stage1_recent_window)

    prepared_cases = rec.prepare_cases(
        cases=cases,
        songs=songs,
        cf_neighbors=cf_neighbors,
        popularity=popularity,
        playlist_semantics={},
        song_semantics={},
        pool_size=rec.STAGE2_POOL_SIZE,
        force_ratio=args.stage1_force_ratio,
        recent_window=args.stage1_recent_window,
        cold_start_threshold=args.cold_start_threshold,
        progress_interval=rec.PROGRESS_INTERVAL,
    )

    feature_pack = build_feature_pack(
        prepared_cases=prepared_cases,
        cases=cases,
        train_indices=train_indices,
        lemon_user_vectors=lemon_user_vectors,
        lemon_song_vectors=lemon_song_vectors,
        kar_user_vectors=kar_user_vectors,
        kar_item_vectors=kar_item_vectors,
        kar_dim=args.kar_dim,
    )
    baseline_cf = rec.build_rankings(prepared_cases, args.cf_weight, 0.0, 0.0)
    enhanced_baseline = rec.build_rankings(prepared_cases, args.cf_weight, args.artist_weight, args.pop_weight)
    fixed_cf_kar = lemonKAR.rank_with_features(
        prepared_cases,
        lemon_user_vectors,
        lemon_song_vectors,
        kar_user_vectors,
        kar_item_vectors,
        args.cf_weight,
        0.0,
        0.0,
        0.0,
        args.kar_weight,
    )
    fixed_enhanced_kar = lemonKAR.rank_with_features(
        prepared_cases,
        lemon_user_vectors,
        lemon_song_vectors,
        kar_user_vectors,
        kar_item_vectors,
        args.cf_weight,
        args.artist_weight,
        args.pop_weight,
        0.0,
        args.kar_weight,
    )
    adapter_rankings_by_label: list[tuple[str, list[list[str]]]] = []
    if args.ablation:
        for label, selected_names in ablation_feature_sets(feature_pack):
            selected_pack = select_feature_pack(feature_pack, selected_names)
            adapter_rankings_by_label.append(
                (label, train_and_rank_adapter(label, selected_pack, train_indices, all_indices, args))
            )
    else:
        adapter_rankings_by_label.append(
            ("Neural adapter", train_and_rank_adapter("Neural adapter", feature_pack, train_indices, all_indices, args))
        )

    rec.print_section("Adapter ranking result")
    rows: list[tuple[str, float, float, float]] = []
    for split_name, indices in [("train 80%", train_indices), ("test 20%", test_indices), ("all 799", all_indices)]:
        split_cases = subset(cases, indices)
        rows.append(evaluate_subset(f"{split_name} | Baseline CF only", subset(baseline_cf, indices), split_cases, args.top_k))
        rows.append(evaluate_subset(f"{split_name} | Fixed CF + KAR", subset(fixed_cf_kar, indices), split_cases, args.top_k))
        rows.append(evaluate_subset(f"{split_name} | Enhanced baseline", subset(enhanced_baseline, indices), split_cases, args.top_k))
        rows.append(evaluate_subset(f"{split_name} | Fixed enhanced baseline + KAR", subset(fixed_enhanced_kar, indices), split_cases, args.top_k))
        for label, rankings in adapter_rankings_by_label:
            rows.append(evaluate_subset(f"{split_name} | {label}", subset(rankings, indices), split_cases, args.top_k))
    rec.print_stage2_results(rows)
    print(f"\nElapsed: {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
