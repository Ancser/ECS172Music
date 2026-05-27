import argparse
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
KAGGLEHUB_VERSION = "0.3.13"

MPD_DATASET = "himanshuwagh/spotify-million"
SONGS_DATASET = "notshrirang/spotify-million-song-dataset"


def _cached_path(
    data_dir: Path,
    expected_name: str | None = None,
    expected_glob: str | None = None,
) -> Path | None:
    if expected_name and (data_dir / expected_name).exists():
        return data_dir / expected_name
    if expected_glob and list(data_dir.glob(expected_glob)):
        return data_dir
    return None


def _save_marker(data_dir: Path):
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / ".download_complete").write_text("ok")


def _copy_files(raw_path: Path, data_dir: Path, pattern: str) -> list[Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for source in raw_path.rglob(pattern):
        destination = data_dir / source.name
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
        copied.append(destination)
    return copied


def _format_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"


def _print_existing_files(data_dir: Path, pattern: str, label: str) -> bool:
    files = sorted(data_dir.glob(pattern))
    if not files:
        return False
    total_size = sum(path.stat().st_size for path in files if path.is_file())
    print(f"{label} already exists in {data_dir}")
    print(f"  files: {len(files):,}")
    print(f"  size:  {_format_size(total_size)}")
    return True


def _import_kagglehub():
    try:
        import kagglehub
        return kagglehub
    except ModuleNotFoundError as exc:
        if exc.name != "kagglehub":
            raise
        raise SystemExit(
            "Missing dependency: kagglehub\n"
            "Install it for this exact Python with:\n"
            f"  {sys.executable} -m pip install --user kagglehub=={KAGGLEHUB_VERSION}"
        ) from exc
    except ImportError as exc:
        raise SystemExit(
            "kagglehub is installed, but it failed to import because its dependency versions are mismatched.\n"
            f"Problem: {exc}\n\n"
            "Repair it from the terminal, then rerun the downloader:\n"
            f"  {sys.executable} -m pip install --user --upgrade --force-reinstall kagglehub=={KAGGLEHUB_VERSION}\n"
        ) from exc


def ensure_playlists(data_dir: Path = DATA_DIR) -> Path:
    """Download Spotify Million Playlist Dataset (~8 GB) on first run.
    Returns repo-local data directory containing mpd.slice.*.json files."""
    cached = _cached_path(data_dir, expected_glob="mpd.slice.*.json")
    if cached:
        _print_existing_files(data_dir, "mpd.slice.*.json", "Playlist slices")
        return cached

    kagglehub = _import_kagglehub()
    print("Downloading Spotify Million Playlist Dataset (~8 GB). This can take a while...")
    raw_path = Path(kagglehub.dataset_download(MPD_DATASET))
    copied = _copy_files(raw_path, data_dir, "mpd.slice.*.json")
    if not copied:
        raise FileNotFoundError(f"No mpd.slice.*.json files found in downloaded dataset: {raw_path}")
    _save_marker(data_dir)
    print(f"Copied {len(copied)} playlist slice files into: {data_dir}")
    _print_existing_files(data_dir, "mpd.slice.*.json", "Playlist slices")
    return data_dir


def ensure_songs(data_dir: Path = DATA_DIR) -> Path:
    """Download Spotify Million Song Dataset (lyrics CSV) on first run.
    Returns repo-local data/spotify_millsongdata.csv."""
    cached = _cached_path(data_dir, "spotify_millsongdata.csv")
    if cached:
        return cached

    kagglehub = _import_kagglehub()
    print("Downloading Spotify Million Song Dataset...")
    raw_path = Path(kagglehub.dataset_download(SONGS_DATASET))
    copied = _copy_files(raw_path, data_dir, "spotify_millsongdata.csv")
    if not copied:
        copied = _copy_files(raw_path, data_dir, "*.csv")
    if not copied:
        raise FileNotFoundError(f"No lyrics CSV found in downloaded dataset: {raw_path}")
    final_path = data_dir / "spotify_millsongdata.csv"
    if copied[0] != final_path:
        copied[0].replace(final_path)
    _save_marker(data_dir)
    print(f"Songs CSV copied to: {final_path}")
    return final_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-time Kaggle data downloader for ECS172 music project")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="Local folder where data files should live")
    parser.add_argument("--songs", action="store_true", help="Download/copy spotify_millsongdata.csv into data/")
    parser.add_argument("--playlists", action="store_true", help="Download/copy MPD mpd.slice.*.json files into data/")
    parser.add_argument("--all", action="store_true", help="Download/copy both songs and playlist data")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data_dir = args.data_dir.resolve()

    if not args.songs and not args.playlists and not args.all:
        args.playlists = True

    print(f"Data directory: {data_dir}")
    data_dir.mkdir(parents=True, exist_ok=True)

    if args.all or args.songs:
        songs_path = ensure_songs(data_dir)
        print(f"Song lyrics ready: {songs_path}")

    if args.all or args.playlists:
        playlists_path = ensure_playlists(data_dir)
        print(f"Playlist slices ready: {playlists_path}")

    print("Done.")
