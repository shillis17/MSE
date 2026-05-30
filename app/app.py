import ast
import gzip
import math
import os
import pickle
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


import boto3
import faiss
import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as T
from botocore.exceptions import BotoCoreError, ClientError
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from panns_inference import AudioTagging
from sklearn.preprocessing import normalize
from transformers import (
    ASTModel,
    AutoFeatureExtractor,
    AutoModel,
    ClapAudioModelWithProjection,
    ClapProcessor,
    Wav2Vec2FeatureExtractor,
)

TRACK_FILE = "https://music-sim-capstone-data.s3.us-east-1.amazonaws.com/raw/metadata/raw_tracks.csv"
S3_BUCKET = "music-sim-capstone-data"
DEFAULT_MODEL = "panns"

LOCAL_ARTIFACT_DIR = Path("model_artifacts")
LOCAL_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_RUNTIME_DIR = Path("runtime_artifacts")
LOCAL_RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_MODEL_DIR = LOCAL_RUNTIME_DIR / "models"
LOCAL_NEIGHBOR_DIR = LOCAL_RUNTIME_DIR / "neighbors"

UPLOAD_DEFAULT_DURATION_SECONDS = 30.0
UPLOAD_TOP_K = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def log_progress(message: str) -> None:
    print(message, flush=True)


MODEL_CONFIGS = {
    "panns": {
        "label": "PANNs",
        "neighbor_lookup_s3_key": "runtime/neighbors/panns/panns_neighbor_lookup.pkl.gz",
        "neighbor_lookup_file": LOCAL_NEIGHBOR_DIR / "panns" / "panns_neighbor_lookup.pkl.gz",
        "index_s3_key": "embeddings/panns/panns_faiss.index",
        "track_ids_s3_key": "embeddings/panns/track_ids.npy",
        "index_file": LOCAL_ARTIFACT_DIR / "panns_faiss.index",
        "track_ids_file": LOCAL_ARTIFACT_DIR / "panns_track_ids.npy",
        "target_sr": 32000,
        "upload_embedder": "panns",
    },
    "clap": {
        "label": "CLAP",
        "neighbor_lookup_s3_key": "runtime/neighbors/clap/clap_neighbor_lookup.pkl.gz",
        "neighbor_lookup_file": LOCAL_NEIGHBOR_DIR / "clap" / "clap_neighbor_lookup.pkl.gz",
        "index_s3_key": "embeddings/clap/clap_faiss.index",
        "track_ids_s3_key": "embeddings/clap/track_ids.npy",
        "index_file": LOCAL_ARTIFACT_DIR / "clap_faiss.index",
        "track_ids_file": LOCAL_ARTIFACT_DIR / "clap_track_ids.npy",
        "target_sr": 48000,
        "checkpoint": "laion/larger_clap_music",
        "checkpoint_env": "CLAP_CHECKPOINT",
        "model_s3_prefix": "runtime/models/clap/",
        "model_dir": LOCAL_MODEL_DIR / "clap",
        "model_s3_files": [
            "config.json",
            "processor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "model.safetensors",
        ],
        "upload_embedder": "clap",
    },
    "ast": {
        "label": "AST",
        "neighbor_lookup_s3_key": "runtime/neighbors/ast/ast_neighbor_lookup.pkl.gz",
        "neighbor_lookup_file": LOCAL_NEIGHBOR_DIR / "ast" / "ast_neighbor_lookup.pkl.gz",
        "index_s3_key": "embeddings/ast/ast_faiss.index",
        "track_ids_s3_key": "embeddings/ast/track_ids.npy",
        "index_file": LOCAL_ARTIFACT_DIR / "ast_faiss.index",
        "track_ids_file": LOCAL_ARTIFACT_DIR / "ast_track_ids.npy",
        "target_sr": 16000,
        "fixed_seconds": 10.0,
        "checkpoint": "MIT/ast-finetuned-audioset-10-10-0.4593",
        "checkpoint_env": "AST_CHECKPOINT",
        "model_s3_prefix": "runtime/models/ast/",
        "model_dir": LOCAL_MODEL_DIR / "ast",
        "model_s3_files": [
            "config.json",
            "preprocessor_config.json",
            "model.safetensors",
        ],
        "upload_embedder": "ast",
    },
    "mert": {
        "label": "MERT",
        "neighbor_lookup_s3_key": "runtime/neighbors/mert/mert_neighbor_lookup.pkl.gz",
        "neighbor_lookup_file": LOCAL_NEIGHBOR_DIR / "mert" / "mert_neighbor_lookup.pkl.gz",
        "index_s3_key": "embeddings/mert/mert_faiss.index",
        "track_ids_s3_key": "embeddings/mert/track_ids.npy",
        "index_file": LOCAL_ARTIFACT_DIR / "mert_faiss.index",
        "track_ids_file": LOCAL_ARTIFACT_DIR / "mert_track_ids.npy",
        "target_sr": 24000,
        "fixed_seconds": 30.0,
        "checkpoint": "m-a-p/MERT-v1-330M",
        "checkpoint_env": "MERT_CHECKPOINT",
        "model_s3_prefix": "runtime/models/mert/",
        "model_dir": LOCAL_MODEL_DIR / "mert",
        "model_s3_files": [
            "config.json",
            "preprocessor_config.json",
            "configuration_MERT.py",
            "modeling_MERT.py",
            "model.safetensors",
        ],
        "upload_embedder": "mert",
    },
    "ccmusic": {
        "label": "CCMUSIC",
        "neighbor_lookup_s3_key": "runtime/neighbors/ccmusic/ccmusic_neighbor_lookup.pkl.gz",
        "neighbor_lookup_file": LOCAL_NEIGHBOR_DIR / "ccmusic" / "ccmusic_neighbor_lookup.pkl.gz",
        "index_s3_key": "embeddings/ccmusic/ccmusic_faiss.index",
        "track_ids_s3_key": "embeddings/ccmusic/track_ids.npy",
        "index_file": LOCAL_ARTIFACT_DIR / "ccmusic_faiss.index",
        "track_ids_file": LOCAL_ARTIFACT_DIR / "ccmusic_track_ids.npy",
        "target_sr": 22050,
        "fixed_seconds": 30.0,
        "checkpoint": "ccmusic-database/music_genre",
        "checkpoint_env": "CCMUSIC_CHECKPOINT",
        "model_s3_prefix": "runtime/models/ccmusic/",
        "model_dir": LOCAL_MODEL_DIR / "ccmusic",
        "model_s3_files": [
            "vgg19_bn_cqt/save.pt",
        ],
        "upload_embedder": "ccmusic",
    },
}

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

s3_client = boto3.client("s3")

tracks_df: Optional[pd.DataFrame] = None
artist_counts_cache: List[Dict[str, Any]] = []
album_counts_cache: List[Dict[str, Any]] = []
genre_counts_cache: List[Dict[str, Any]] = []
track_row_by_id: Dict[int, Dict[str, Any]] = {}
neighbors_lookup_by_model: Dict[str, Dict[int, List[Dict[str, Any]]]] = {}

faiss_index_by_model: Dict[str, Any] = {}
track_ids_by_model: Dict[str, np.ndarray] = {}

panns_model: Optional[AudioTagging] = None
clap_model: Optional[ClapAudioModelWithProjection] = None
clap_processor: Optional[ClapProcessor] = None
ast_model: Optional[ASTModel] = None
ast_feature_extractor: Optional[AutoFeatureExtractor] = None
mert_model: Optional[AutoModel] = None
mert_processor: Optional[Wav2Vec2FeatureExtractor] = None
ccmusic_model: Optional[nn.Module] = None
ccmusic_transform: Optional[T.Compose] = None

core_data_ready = False
core_data_loading = False
core_data_error: Optional[str] = None
upload_assets_ready = False
upload_assets_loading = False
upload_assets_error: Optional[str] = None
upload_model_status: Dict[str, str] = {
    model_name: "pending"
    for model_name, config in MODEL_CONFIGS.items()
    if config.get("upload_embedder")
}

core_data_lock = threading.Lock()
core_data_event = threading.Event()
upload_assets_lock = threading.Lock()
upload_assets_event = threading.Event()

TRACK_DETAIL_FIELDS = [
    "album_id", "album_title", "artist_id", "artist_name", "artist_url",
    "artist_website", "license_image_file", "license_title", "license_url",
    "tags", "track_composer", "track_copyright_c", "track_copyright_p",
    "track_date_created", "track_date_recorded", "track_disc_number",
    "track_duration", "track_explicit", "track_explicit_notes",
    "track_genres", "track_information", "track_instrumental",
    "track_language_code", "track_lyricist", "track_number",
    "track_publisher", "track_title", "track_url",
]


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_model_config(model: str) -> Dict[str, Any]:
    config = MODEL_CONFIGS.get(model)
    if config is None:
        raise HTTPException(status_code=400, detail=f"Unknown model '{model}'.")
    return config


def get_checkpoint(model: str) -> str:
    config = get_model_config(model)
    env_name = config.get("checkpoint_env")
    if env_name:
        return os.getenv(env_name, config["checkpoint"])
    return config["checkpoint"]


def safe_literal(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        try:
            return ast.literal_eval(value)
        except Exception:
            return []
    return []


def first_nonempty(values: List[Any], default: str = "Unknown") -> str:
    for v in values:
        if pd.notna(v):
            s = str(v).strip()
            if s:
                return s
    return default


def clean_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        return value
    return value


def parse_genres(value: Any) -> List[str]:
    parsed = safe_literal(value)
    names: List[str] = []

    for item in parsed:
        if isinstance(item, dict):
            name = item.get("genre_title") or item.get("title") or item.get("name")
            if name:
                names.append(str(name))
        elif isinstance(item, str):
            names.append(item)

    seen = set()
    cleaned: List[str] = []
    for g in names:
        g = g.strip()
        if g and g not in seen:
            seen.add(g)
            cleaned.append(g)

    return cleaned


def parse_string_list(value: Any) -> List[str]:
    if pd.isna(value):
        return []

    if isinstance(value, list):
        raw_items = value
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        parsed = safe_literal(stripped)
        raw_items = parsed if parsed else [part.strip() for part in stripped.split(",")]
    else:
        raw_items = [value]

    cleaned: List[str] = []
    seen = set()

    for item in raw_items:
        if isinstance(item, dict):
            text = item.get("tag") or item.get("name") or item.get("title") or item.get("label")
        else:
            text = item

        if text is None:
            continue

        text = str(text).strip()
        if text and text.lower() != "nan" and text not in seen:
            seen.add(text)
            cleaned.append(text)

    return cleaned


def clean_track_id(track_id: Any) -> Optional[int]:
    try:
        if pd.isna(track_id):
            return None
        return int(track_id)
    except Exception:
        return None


def build_audio_url(track_id: int) -> str:
    return f"https://{S3_BUCKET}.s3.us-east-1.amazonaws.com/raw/audio/{track_id:06d}.mp3"


def build_spectrogram_url(track_id: int) -> str:
    return f"https://{S3_BUCKET}.s3.us-east-1.amazonaws.com/processed/spectrograms/{track_id:06d}_spectrogram.png"


def paginate_list(items: List[Any], page: int, limit: int, label: str = "items") -> Dict[str, Any]:
    total = len(items)
    total_pages = max(1, math.ceil(total / limit)) if limit > 0 else 1
    page = max(page, 1)
    start = (page - 1) * limit
    end = start + limit

    return {
        label: items[start:end],
        "page": page,
        "limit": limit,
        "total_results": total,
        "total_pages": total_pages,
        "has_next": page < total_pages,
        "has_prev": page > 1,
    }


def paginate_df(df: pd.DataFrame, page: int, limit: int) -> Dict[str, Any]:
    total = len(df)
    total_pages = max(1, math.ceil(total / limit)) if limit > 0 else 1
    page = max(page, 1)
    start = (page - 1) * limit
    end = start + limit

    return {
        "df": df.iloc[start:end],
        "page": page,
        "limit": limit,
        "total_results": total,
        "total_pages": total_pages,
        "has_next": page < total_pages,
        "has_prev": page > 1,
    }


def build_track_card(row: pd.Series) -> Dict[str, Any]:
    return {
        "track_id": int(clean_scalar(row["track_id"])),
        "title": clean_scalar(row["title"]),
        "artist": clean_scalar(row["artist"]),
        "album": clean_scalar(row["album"]),
        "duration": clean_scalar(row["duration"]),
        "genres": row["genres"] if isinstance(row["genres"], list) else [],
    }


def build_track_detail(row: Dict[str, Any], model: str, recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "track_id": int(clean_scalar(row.get("track_id"))),
        "title": clean_scalar(row.get("title")),
        "artist": clean_scalar(row.get("artist")),
        "album": clean_scalar(row.get("album")),
        "duration": clean_scalar(row.get("duration")),
        "genres": row.get("genres") if isinstance(row.get("genres"), list) else [],
        "audio_url": clean_scalar(row.get("audio_url")),
        "spectrogram_url": clean_scalar(row.get("spectrogram_url")),
        "model": model,
        "model_label": MODEL_CONFIGS[model]["label"],
        "album_id": clean_scalar(row.get("album_id")),
        "album_title": clean_scalar(row.get("album_title")),
        "artist_id": clean_scalar(row.get("artist_id")),
        "artist_name": clean_scalar(row.get("artist_name")),
        "artist_url": clean_scalar(row.get("artist_url")),
        "artist_website": clean_scalar(row.get("artist_website")),
        "license_image_file": clean_scalar(row.get("license_image_file")),
        "license_title": clean_scalar(row.get("license_title")),
        "license_url": clean_scalar(row.get("license_url")),
        "tags": [clean_scalar(x) for x in row.get("tags", [])] if isinstance(row.get("tags"), list) else [],
        "track_composer": clean_scalar(row.get("track_composer")),
        "track_copyright_c": clean_scalar(row.get("track_copyright_c")),
        "track_copyright_p": clean_scalar(row.get("track_copyright_p")),
        "track_date_created": clean_scalar(row.get("track_date_created")),
        "track_date_recorded": clean_scalar(row.get("track_date_recorded")),
        "track_disc_number": clean_scalar(row.get("track_disc_number")),
        "track_duration": clean_scalar(row.get("track_duration")),
        "track_explicit": clean_scalar(row.get("track_explicit")),
        "track_explicit_notes": clean_scalar(row.get("track_explicit_notes")),
        "track_genres": [clean_scalar(x) for x in row.get("track_genres", [])] if isinstance(row.get("track_genres"), list) else [],
        "track_information": clean_scalar(row.get("track_information")),
        "track_instrumental": clean_scalar(row.get("track_instrumental")),
        "track_language_code": clean_scalar(row.get("track_language_code")),
        "track_lyricist": clean_scalar(row.get("track_lyricist")),
        "track_number": clean_scalar(row.get("track_number")),
        "track_publisher": clean_scalar(row.get("track_publisher")),
        "track_title": clean_scalar(row.get("track_title")),
        "track_url": clean_scalar(row.get("track_url")),
        "recommendations": recommendations,
    }


def precompute_lists(df: pd.DataFrame) -> None:
    global artist_counts_cache, album_counts_cache, genre_counts_cache

    artist_counts = (
        df[df["artist"].fillna("").astype(str).str.strip() != ""]
        .groupby("artist")
        .size()
        .reset_index(name="track_count")
    )
    artist_counts["artist"] = artist_counts["artist"].astype(str).str.strip()
    artist_counts = artist_counts.sort_values("artist", key=lambda col: col.str.lower())

    album_counts = (
        df[df["album"].fillna("").astype(str).str.strip() != ""]
        .groupby("album")
        .size()
        .reset_index(name="track_count")
    )
    album_counts["album"] = album_counts["album"].astype(str).str.strip()
    album_counts = album_counts.sort_values("album", key=lambda col: col.str.lower())

    genre_rows = []
    for genres in df["genres"]:
        if isinstance(genres, list):
            genre_values = genres
        elif pd.isna(genres):
            genre_values = []
        else:
            genre_values = [genres]

        genre_rows.extend(str(genre).strip() for genre in genre_values if str(genre).strip())

    genre_counts_df = pd.DataFrame({"genre": genre_rows})
    if len(genre_counts_df) > 0:
        genre_counts_df = (
            genre_counts_df.groupby("genre")
            .size()
            .reset_index(name="track_count")
            .sort_values("genre", key=lambda col: col.str.lower())
        )
    else:
        genre_counts_df = pd.DataFrame(columns=["genre", "track_count"])

    artist_counts_cache = [{"name": row["artist"], "track_count": int(row["track_count"])} for _, row in artist_counts.iterrows()]
    album_counts_cache = [{"name": row["album"], "track_count": int(row["track_count"])} for _, row in album_counts.iterrows()]
    genre_counts_cache = [{"name": row["genre"], "track_count": int(row["track_count"])} for _, row in genre_counts_df.iterrows()]


def sort_cached_items(items: List[Dict[str, Any]], sort: str) -> List[Dict[str, Any]]:
    if sort == "name_asc":
        return sorted(items, key=lambda item: str(item.get("name", "")).lower())
    if sort == "name_desc":
        return sorted(items, key=lambda item: str(item.get("name", "")).lower(), reverse=True)
    if sort == "tracks_desc":
        return sorted(items, key=lambda item: (-int(item.get("track_count", 0)), str(item.get("name", "")).lower()))
    if sort == "tracks_asc":
        return sorted(items, key=lambda item: (int(item.get("track_count", 0)), str(item.get("name", "")).lower()))
    raise HTTPException(status_code=400, detail="Invalid sort value.")


def sort_tracks_df(df: pd.DataFrame, sort: str) -> pd.DataFrame:
    if sort == "title_asc":
        return df.sort_values("title", key=lambda col: col.fillna("").astype(str).str.lower())
    if sort == "title_desc":
        return df.sort_values("title", key=lambda col: col.fillna("").astype(str).str.lower(), ascending=False)
    if sort == "artist_asc":
        return df.sort_values("artist", key=lambda col: col.fillna("").astype(str).str.lower())
    if sort == "artist_desc":
        return df.sort_values("artist", key=lambda col: col.fillna("").astype(str).str.lower(), ascending=False)
    if sort == "album_asc":
        return df.sort_values("album", key=lambda col: col.fillna("").astype(str).str.lower())
    if sort == "album_desc":
        return df.sort_values("album", key=lambda col: col.fillna("").astype(str).str.lower(), ascending=False)
    if sort == "genre_asc":
        temp = df.assign(primary_genre=df["genres"].apply(lambda g: g[0] if g else ""))
        return temp.sort_values("primary_genre", key=lambda col: col.fillna("").astype(str).str.lower()).drop(columns=["primary_genre"])
    if sort == "genre_desc":
        temp = df.assign(primary_genre=df["genres"].apply(lambda g: g[0] if g else ""))
        return temp.sort_values("primary_genre", key=lambda col: col.fillna("").astype(str).str.lower(), ascending=False).drop(columns=["primary_genre"])
    raise HTTPException(status_code=400, detail="Invalid sort value.")


def ensure_local_artifact(local_path: Path, s3_key: str) -> None:
    if local_path.exists():
        log_progress(f"Upload artifacts: using cached {local_path} ({local_path.stat().st_size / 1024 / 1024:.1f} MB)")
        return

    local_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_start_time = time.perf_counter()
    log_progress(f"Upload artifacts: downloading s3://{S3_BUCKET}/{s3_key} to {local_path}")

    try:
        s3_client.download_file(S3_BUCKET, s3_key, str(local_path))
        log_progress(
            f"Upload artifacts: downloaded {local_path} "
            f"({local_path.stat().st_size / 1024 / 1024:.1f} MB) "
            f"in {time.perf_counter() - artifact_start_time:.1f}s"
        )
    except (BotoCoreError, ClientError) as e:
        raise RuntimeError(f"Failed to download s3://{S3_BUCKET}/{s3_key}: {e}") from e


def ensure_local_s3_files(local_dir: Path, s3_prefix: str, relative_files: List[str]) -> None:
    missing_files = [
        relative_file
        for relative_file in relative_files
        if not (local_dir / relative_file).exists()
    ]

    if not missing_files:
        log_progress(f"Runtime models: using cached {local_dir}")
        return

    local_dir.mkdir(parents=True, exist_ok=True)
    download_start_time = time.perf_counter()
    log_progress(
        f"Runtime models: downloading {len(missing_files)} files "
        f"from s3://{S3_BUCKET}/{s3_prefix} to {local_dir}"
    )
    downloaded_files = 0
    downloaded_bytes = 0

    try:
        for relative_file in missing_files:
            key = f"{s3_prefix}{relative_file}"
            local_path = local_dir / relative_file
            local_path.parent.mkdir(parents=True, exist_ok=True)

            response = s3_client.head_object(Bucket=S3_BUCKET, Key=key)
            size = int(response.get("ContentLength", 0))
            log_progress(f"Runtime models: downloading s3://{S3_BUCKET}/{key} ({size / 1024 / 1024:.1f} MB)")
            s3_client.download_file(S3_BUCKET, key, str(local_path))
            downloaded_files += 1
            downloaded_bytes += size
    except (BotoCoreError, ClientError) as e:
        raise RuntimeError(f"Failed to download s3://{S3_BUCKET}/{s3_prefix}: {e}") from e

    log_progress(
        f"Runtime models: downloaded {downloaded_files} files "
        f"({downloaded_bytes / 1024 / 1024:.1f} MB) "
        f"from s3://{S3_BUCKET}/{s3_prefix} in {time.perf_counter() - download_start_time:.1f}s"
    )


def ensure_runtime_model_dir(model: str) -> Path:
    config = get_model_config(model)
    local_dir = config["model_dir"]
    ensure_local_s3_files(local_dir, config["model_s3_prefix"], config["model_s3_files"])
    return local_dir


def load_neighbor_lookup(model_name: str, config: Dict[str, Any]) -> Dict[int, List[Dict[str, Any]]]:
    lookup_start_time = time.perf_counter()
    local_path = config["neighbor_lookup_file"]
    ensure_local_artifact(local_path, config["neighbor_lookup_s3_key"])
    log_progress(f"Core data: loading {model_name} neighbor lookup from {local_path}")

    with gzip.open(local_path, "rb") as f:
        lookup = pickle.load(f)

    log_progress(
        f"Core data: loaded {model_name} neighbor lookup "
        f"tracks={len(lookup)} in {time.perf_counter() - lookup_start_time:.1f}s"
    )
    return lookup


def ensure_upload_artifacts_loaded() -> None:
    artifact_load_start_time = time.perf_counter()
    log_progress("Upload artifacts: starting FAISS index and track id preload")
    for model_name, config in MODEL_CONFIGS.items():
        model_start_time = time.perf_counter()
        log_progress(f"Upload artifacts: checking {model_name}")
        ensure_local_artifact(config["index_file"], config["index_s3_key"])
        ensure_local_artifact(config["track_ids_file"], config["track_ids_s3_key"])

        if model_name not in faiss_index_by_model:
            faiss_start_time = time.perf_counter()
            log_progress(f"Upload artifacts: reading {model_name} FAISS index from {config['index_file']}")
            faiss_index_by_model[model_name] = faiss.read_index(str(config["index_file"]))
            log_progress(
                f"Upload artifacts: loaded {model_name} FAISS index "
                f"dim={faiss_index_by_model[model_name].d} vectors={faiss_index_by_model[model_name].ntotal} "
                f"in {time.perf_counter() - faiss_start_time:.1f}s"
            )

        if model_name not in track_ids_by_model:
            ids_start_time = time.perf_counter()
            log_progress(f"Upload artifacts: reading {model_name} track ids from {config['track_ids_file']}")
            track_ids_by_model[model_name] = np.load(config["track_ids_file"], allow_pickle=True)
            log_progress(
                f"Upload artifacts: loaded {model_name} track ids "
                f"count={len(track_ids_by_model[model_name])} "
                f"in {time.perf_counter() - ids_start_time:.1f}s"
            )

        log_progress(f"Upload artifacts: finished {model_name} in {time.perf_counter() - model_start_time:.1f}s")

    log_progress(f"Upload artifacts: all FAISS assets loaded in {time.perf_counter() - artifact_load_start_time:.1f}s")


def ensure_panns_model_loaded() -> None:
    global panns_model
    if panns_model is None:
        panns_model = AudioTagging(device=DEVICE)


def ensure_clap_model_loaded() -> None:
    global clap_model, clap_processor
    if clap_model is None or clap_processor is None:
        model_dir = ensure_runtime_model_dir("clap")
        clap_model = ClapAudioModelWithProjection.from_pretrained(model_dir).to(DEVICE)
        clap_processor = ClapProcessor.from_pretrained(model_dir)
        clap_model.eval()


def ensure_ast_model_loaded() -> None:
    global ast_model, ast_feature_extractor
    if ast_model is None or ast_feature_extractor is None:
        model_dir = ensure_runtime_model_dir("ast")
        ast_feature_extractor = AutoFeatureExtractor.from_pretrained(model_dir)
        ast_model = ASTModel.from_pretrained(model_dir).to(DEVICE)
        ast_model.eval()


def ensure_mert_model_loaded() -> None:
    global mert_model, mert_processor
    if mert_model is None or mert_processor is None:
        model_dir = ensure_runtime_model_dir("mert")
        mert_processor = Wav2Vec2FeatureExtractor.from_pretrained(model_dir, trust_remote_code=True)
        mert_model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).to(DEVICE)
        mert_model.eval()


def ensure_ccmusic_model_loaded() -> None:
    global ccmusic_model, ccmusic_transform

    if ccmusic_model is not None and ccmusic_transform is not None:
        return

    model_dir = ensure_runtime_model_dir("ccmusic")
    weight_file = model_dir / "vgg19_bn_cqt" / "save.pt"

    if not weight_file.exists():
        raise RuntimeError(f"Expected CCMusic VGG19-BN CQT weights at {weight_file}")

    model_obj = tv_models.vgg19_bn(weights=None)
    model_obj.classifier = nn.Sequential(
        nn.Dropout(),
        nn.Linear(25088, 3958),
        nn.ReLU(inplace=True),
        nn.Dropout(),
        nn.Linear(3958, 629),
        nn.ReLU(inplace=True),
        nn.Dropout(),
        nn.Linear(629, 100),
        nn.ReLU(inplace=True),
        nn.Linear(100, 16),
    )

    state = torch.load(weight_file, map_location=DEVICE)

    if isinstance(state, dict):
        for key in ["state_dict", "model_state_dict", "model"]:
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break

    clean_state = {}
    for k, v in state.items():
        new_k = k
        for prefix in ["module.", "model.", "vgg_model."]:
            if new_k.startswith(prefix):
                new_k = new_k[len(prefix):]
        clean_state[new_k] = v

    model_obj.load_state_dict(clean_state, strict=True)
    model_obj.classifier[-1] = Identity()
    model_obj = model_obj.to(DEVICE)
    model_obj.eval()

    ccmusic_model = model_obj
    ccmusic_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


UPLOAD_MODEL_LOADERS: Dict[str, Callable[[], None]] = {
    "panns": ensure_panns_model_loaded,
    "clap": ensure_clap_model_loaded,
    "ast": ensure_ast_model_loaded,
    "mert": ensure_mert_model_loaded,
    "ccmusic": ensure_ccmusic_model_loaded,
}


def ensure_upload_models_loaded() -> None:
    errors: List[str] = []

    for model_name, load_model in UPLOAD_MODEL_LOADERS.items():
        model_start_time = time.perf_counter()
        upload_model_status[model_name] = "loading"
        log_progress(f"Upload models: loading {model_name}")

        try:
            load_model()
            upload_model_status[model_name] = "ready"
            log_progress(f"Upload models: {model_name} ready in {time.perf_counter() - model_start_time:.1f}s")
        except Exception as e:
            upload_model_status[model_name] = f"error: {e}"
            errors.append(f"{model_name}: {e}")
            log_progress(f"Upload models: {model_name} preload failed after {time.perf_counter() - model_start_time:.1f}s: {e}")

    if errors:
        raise RuntimeError("Upload model preload failed: " + "; ".join(errors))


def _background_load_upload_assets() -> None:
    global upload_assets_ready, upload_assets_loading, upload_assets_error

    try:
        wait_for_core_data()
        ensure_upload_artifacts_loaded()
        ensure_upload_models_loaded()
        upload_assets_ready = True
        upload_assets_error = None
        log_progress("Upload-search assets and models ready.")
    except Exception as e:
        upload_assets_ready = False
        upload_assets_error = str(e)
        log_progress(f"Upload asset preload failed: {e}")
    finally:
        upload_assets_loading = False
        upload_assets_event.set()


def start_upload_assets_background_load() -> None:
    global upload_assets_loading, upload_assets_error

    with upload_assets_lock:
        if upload_assets_ready or upload_assets_loading:
            return

        upload_assets_loading = True
        upload_assets_error = None
        upload_assets_event.clear()

        thread = threading.Thread(target=_background_load_upload_assets, daemon=True)
        thread.start()


def wait_for_upload_assets(timeout: float = 1200.0) -> None:
    wait_for_core_data(timeout=timeout)

    if upload_assets_ready:
        return

    start_upload_assets_background_load()

    if upload_assets_loading:
        raise HTTPException(status_code=503, detail="Upload search assets are still loading. Please try again shortly.")

    raise HTTPException(status_code=503, detail=f"Upload search assets failed to load: {upload_assets_error or 'unknown error'}")


def _build_core_data() -> None:
    global tracks_df, track_row_by_id, neighbors_lookup_by_model
    global artist_counts_cache, album_counts_cache, genre_counts_cache

    start_time = time.perf_counter()
    log_progress(f"Core data: loading tracks from {TRACK_FILE}")
    df = pd.read_csv(TRACK_FILE, low_memory=False)
    log_progress(f"Core data: loaded tracks rows={len(df)} columns={list(df.columns)} in {time.perf_counter() - start_time:.1f}s")

    df["track_id"] = df["track_id"].astype(int)
    if "title" not in df.columns and "track_title" in df.columns:
        df["title"] = df["track_title"]
    if "artist" not in df.columns and "artist_name" in df.columns:
        df["artist"] = df["artist_name"]
    if "album" not in df.columns and "album_title" in df.columns:
        df["album"] = df["album_title"]
    if "duration" not in df.columns and "track_duration" in df.columns:
        df["duration"] = df["track_duration"]
    if "track_genres" in df.columns:
        df["genres"] = df["track_genres"].apply(parse_genres)
    if "audio_url" not in df.columns:
        df["audio_url"] = df["track_id"].apply(build_audio_url)
    if "spectrogram_url" not in df.columns:
        df["spectrogram_url"] = df["track_id"].apply(build_spectrogram_url)

    needed_columns = ["track_id", "title", "artist", "album", "duration", "genres", "audio_url", "spectrogram_url"]
    detail_columns = [field for field in TRACK_DETAIL_FIELDS if field in df.columns]
    built_tracks_df = df[needed_columns + [field for field in detail_columns if field not in needed_columns]].copy()
    grouped_neighbors_by_model: Dict[str, Dict[int, List[Dict[str, Any]]]] = {}
    log_progress(f"Core data: using track metadata rows={len(built_tracks_df)} in {time.perf_counter() - start_time:.1f}s")

    for model_name, config in MODEL_CONFIGS.items():
        grouped_neighbors_by_model[model_name] = load_neighbor_lookup(model_name, config)

    tracks_df = built_tracks_df
    track_row_by_id = {int(row["track_id"]): row.to_dict() for _, row in built_tracks_df.iterrows()}
    neighbors_lookup_by_model = grouped_neighbors_by_model
    log_progress(f"Core data: built track lookup rows={len(track_row_by_id)} in {time.perf_counter() - start_time:.1f}s")

    precompute_lists(built_tracks_df)
    log_progress(f"Core data: precomputed browse lists in {time.perf_counter() - start_time:.1f}s")

    log_progress(f"Loaded tracks: {len(tracks_df)}")
    log_progress(f"Loaded artists: {len(artist_counts_cache)}")
    log_progress(f"Loaded albums: {len(album_counts_cache)}")
    log_progress(f"Loaded genres: {len(genre_counts_cache)}")
    log_progress(f"Loaded models: {list(neighbors_lookup_by_model.keys())}")
    log_progress(f"Upload-search device: {DEVICE}")
    log_progress(f"Core data: finished in {time.perf_counter() - start_time:.1f}s")


def _background_load_core_data() -> None:
    global core_data_ready, core_data_loading, core_data_error

    try:
        _build_core_data()
        core_data_ready = True
        core_data_error = None
        log_progress("Core track data ready.")
        log_progress("Starting upload-search asset preload in background...")
        start_upload_assets_background_load()
    except Exception as e:
        core_data_ready = False
        core_data_error = str(e)
        log_progress(f"Core data preload failed: {e}")
    finally:
        core_data_loading = False
        core_data_event.set()


def start_core_data_background_load() -> None:
    global core_data_loading, core_data_error

    with core_data_lock:
        if core_data_ready or core_data_loading:
            return

        core_data_loading = True
        core_data_error = None
        core_data_event.clear()

        thread = threading.Thread(target=_background_load_core_data, daemon=True)
        thread.start()


def wait_for_core_data(timeout: float = 1200.0) -> None:
    if core_data_ready:
        return

    start_core_data_background_load()

    finished = core_data_event.wait(timeout=timeout)
    if not finished:
        raise HTTPException(status_code=503, detail="Core track data is still loading. Please try again shortly.")

    if not core_data_ready:
        raise HTTPException(status_code=503, detail=f"Core track data failed to load: {core_data_error or 'unknown error'}")


def get_filtered_tracks_df(
    genre: Optional[str] = None,
    artist: Optional[str] = None,
    album: Optional[str] = None,
    search: Optional[str] = None,
) -> pd.DataFrame:
    wait_for_core_data()
    df = tracks_df

    if genre:
        genre_lower = genre.strip().lower()
        df = df[df["genres"].apply(lambda gs: any(str(g).lower() == genre_lower for g in gs))]

    if artist:
        artist_lower = artist.strip().lower()
        df = df[df["artist"].fillna("").astype(str).str.lower() == artist_lower]

    if album:
        album_lower = album.strip().lower()
        df = df[df["album"].fillna("").astype(str).str.lower() == album_lower]

    if search:
        s = search.strip().lower()
        df = df[
            df["title"].fillna("").astype(str).str.lower().str.contains(s, na=False, regex=False)
            | df["artist"].fillna("").astype(str).str.lower().str.contains(s, na=False, regex=False)
            | df["album"].fillna("").astype(str).str.lower().str.contains(s, na=False, regex=False)
        ]

    return df


def prepare_query_waveform(source_path: str, target_sr: int, start_seconds: float, duration_seconds: float) -> np.ndarray:
    y, _ = librosa.load(source_path, sr=target_sr, mono=True)

    start_seconds = max(0.0, float(start_seconds))
    duration_seconds = max(1.0, float(duration_seconds))

    start_sample = int(start_seconds * target_sr)
    end_sample = int((start_seconds + duration_seconds) * target_sr)
    clip = y[start_sample:end_sample]

    if clip.size == 0:
        raise HTTPException(status_code=400, detail="Selected clip is empty.")

    fixed_samples = int(target_sr * duration_seconds)

    if len(clip) > fixed_samples:
        clip = clip[:fixed_samples]
    elif len(clip) < fixed_samples:
        clip = np.pad(clip, (0, fixed_samples - len(clip)), mode="constant")

    return clip.astype(np.float32)


def get_upload_duration(model: str, requested_duration_seconds: float) -> float:
    config = get_model_config(model)
    return float(config.get("fixed_seconds") or requested_duration_seconds)


def embed_uploaded_clip_with_panns(source_path: str, start_seconds: float, duration_seconds: float, model: str) -> np.ndarray:
    ensure_panns_model_loaded()
    config = get_model_config(model)

    waveform = prepare_query_waveform(
        source_path,
        int(config["target_sr"]),
        start_seconds,
        get_upload_duration(model, duration_seconds),
    )

    batch = waveform[np.newaxis, :]
    _, batch_embeddings = panns_model.inference(batch)
    embeddings = np.asarray(batch_embeddings, dtype=np.float32)
    return normalize(embeddings, norm="l2").astype(np.float32)


def embed_uploaded_clip_with_clap(source_path: str, start_seconds: float, duration_seconds: float, model: str) -> np.ndarray:
    ensure_clap_model_loaded()
    config = get_model_config(model)

    waveform = prepare_query_waveform(
        source_path,
        int(config["target_sr"]),
        start_seconds,
        get_upload_duration(model, duration_seconds),
    )

    inputs = clap_processor(
        audio=[waveform],
        sampling_rate=int(config["target_sr"]),
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = clap_model(**inputs)
        embeddings = outputs.audio_embeds.detach().cpu().numpy().astype(np.float32)

    return normalize(embeddings, norm="l2").astype(np.float32)


def embed_uploaded_clip_with_ast(source_path: str, start_seconds: float, duration_seconds: float, model: str) -> np.ndarray:
    ensure_ast_model_loaded()
    config = get_model_config(model)

    waveform = prepare_query_waveform(
        source_path,
        int(config["target_sr"]),
        start_seconds,
        get_upload_duration(model, duration_seconds),
    )

    inputs = ast_feature_extractor(
        [waveform],
        sampling_rate=int(config["target_sr"]),
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = ast_model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy().astype(np.float32)

    return normalize(embeddings, norm="l2").astype(np.float32)


def embed_uploaded_clip_with_mert(source_path: str, start_seconds: float, duration_seconds: float, model: str) -> np.ndarray:
    ensure_mert_model_loaded()
    config = get_model_config(model)

    waveform = prepare_query_waveform(
        source_path,
        int(config["target_sr"]),
        start_seconds,
        get_upload_duration(model, duration_seconds),
    )

    inputs = mert_processor(
        [waveform],
        sampling_rate=int(config["target_sr"]),
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = mert_model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy().astype(np.float32)

    return normalize(embeddings, norm="l2").astype(np.float32)


def waveform_to_ccmusic_image(waveform: np.ndarray, sr: int) -> Image.Image:
    cqt = librosa.cqt(
        y=waveform,
        sr=sr,
        hop_length=512,
        n_bins=84,
        bins_per_octave=12,
    )
    cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
    cqt_db = np.nan_to_num(cqt_db, nan=-80.0, posinf=0.0, neginf=-80.0)

    cqt_min = float(cqt_db.min())
    cqt_max = float(cqt_db.max())
    cqt_norm = (cqt_db - cqt_min) / max(cqt_max - cqt_min, 1e-6)

    cqt_img = (cqt_norm * 255).astype(np.uint8)
    return Image.fromarray(cqt_img, mode="L").convert("RGB")


def embed_uploaded_clip_with_ccmusic(source_path: str, start_seconds: float, duration_seconds: float, model: str) -> np.ndarray:
    ensure_ccmusic_model_loaded()
    config = get_model_config(model)

    waveform = prepare_query_waveform(
        source_path,
        int(config["target_sr"]),
        start_seconds,
        get_upload_duration(model, duration_seconds),
    )

    image = waveform_to_ccmusic_image(waveform, int(config["target_sr"]))
    tensor = ccmusic_transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        embeddings = ccmusic_model(tensor).detach().cpu().numpy().astype(np.float32)

    return normalize(embeddings, norm="l2").astype(np.float32)


UPLOAD_EMBEDDERS: Dict[str, Callable[[str, float, float, str], np.ndarray]] = {
    "panns": embed_uploaded_clip_with_panns,
    "clap": embed_uploaded_clip_with_clap,
    "ast": embed_uploaded_clip_with_ast,
    "mert": embed_uploaded_clip_with_mert,
    "ccmusic": embed_uploaded_clip_with_ccmusic,
}


def search_uploaded_clip_with_model(
    model: str,
    source_path: str,
    start_seconds: float,
    duration_seconds: float,
    top_k: int = UPLOAD_TOP_K,
) -> List[Dict[str, Any]]:
    config = get_model_config(model)
    embedder_name = config.get("upload_embedder")

    embedder = UPLOAD_EMBEDDERS.get(embedder_name)
    if embedder is None:
        raise HTTPException(status_code=501, detail=f"Upload embedder '{embedder_name}' is not registered.")

    embeddings = embedder(source_path, start_seconds, duration_seconds, model)
    expected_dim = faiss_index_by_model[model].d
    actual_dim = embeddings.shape[1] if embeddings.ndim == 2 else None

    if actual_dim != expected_dim:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Embedding dimension mismatch for model '{model}': "
                f"query embedding has {actual_dim}, FAISS index expects {expected_dim}."
            ),
        )

    scores, neighbors = faiss_index_by_model[model].search(embeddings, top_k)

    hits: List[Dict[str, Any]] = []
    for score, idx in zip(scores[0], neighbors[0]):
        if idx < 0:
            continue

        track_id = int(track_ids_by_model[model][idx])
        hits.append({"track_id": track_id, "score": float(score)})

    return hits


def hydrate_search_results(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    for i, hit in enumerate(hits):
        track_id = int(hit["track_id"])
        row = track_row_by_id.get(track_id)
        if row is None:
            continue

        results.append({
            "track_id": int(clean_scalar(row["track_id"])),
            "title": clean_scalar(row["title"]),
            "artist": clean_scalar(row["artist"]),
            "album": clean_scalar(row["album"]),
            "duration": clean_scalar(row["duration"]),
            "genres": row["genres"] if isinstance(row["genres"], list) else [],
            "score": float(hit["score"]) if hit.get("score") is not None else None,
            "rank": i + 1,
        })

    return results


@app.on_event("startup")
def startup() -> None:
    log_progress("Starting core data preload in background...")
    start_core_data_background_load()


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "core_data_ready": core_data_ready,
        "core_data_loading": core_data_loading,
        "core_data_error": core_data_error,
        "upload_assets_ready": upload_assets_ready,
        "upload_assets_loading": upload_assets_loading,
        "upload_assets_error": upload_assets_error,
        "upload_model_status": upload_model_status,
        "device": DEVICE,
    }


@app.get("/ready")
def ready() -> Dict[str, Any]:
    if not core_data_ready or not upload_assets_ready:
        raise HTTPException(status_code=503, detail={
            "core_data_ready": core_data_ready,
            "core_data_loading": core_data_loading,
            "core_data_error": core_data_error,
            "upload_assets_ready": upload_assets_ready,
            "upload_assets_loading": upload_assets_loading,
            "upload_assets_error": upload_assets_error,
            "upload_model_status": upload_model_status,
        })

    return {
        "status": "ready",
        "core_data_ready": core_data_ready,
        "upload_assets_ready": upload_assets_ready,
        "upload_model_status": upload_model_status,
    }


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "message": "Music Similarity API is running",
        "default_model": DEFAULT_MODEL,
        "available_models": [
            {
                "name": model_name,
                "label": config["label"],
                "upload_search_supported": config.get("upload_embedder") in UPLOAD_EMBEDDERS,
                "upload_model_status": upload_model_status.get(model_name, "not_applicable"),
            }
            for model_name, config in MODEL_CONFIGS.items()
        ],
        "core_data_ready": core_data_ready,
        "core_data_loading": core_data_loading,
        "core_data_error": core_data_error,
        "upload_assets_ready": upload_assets_ready,
        "upload_assets_loading": upload_assets_loading,
        "upload_assets_error": upload_assets_error,
        "upload_model_status": upload_model_status,
    }


@app.get("/models")
def get_models() -> Dict[str, Any]:
    return {
        "default_model": DEFAULT_MODEL,
        "models": [
            {
                "name": model_name,
                "label": config["label"],
                "upload_search_supported": config.get("upload_embedder") in UPLOAD_EMBEDDERS,
                "upload_model_status": upload_model_status.get(model_name, "not_applicable"),
            }
            for model_name, config in MODEL_CONFIGS.items()
        ],
    }


@app.get("/genres")
def get_genres(
    page: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=500),
    search: Optional[str] = None,
    sort: str = Query("name_asc"),
) -> Dict[str, Any]:
    wait_for_core_data()
    items = genre_counts_cache

    if search:
        s = search.strip().lower()
        items = [item for item in items if s in str(item["name"]).lower()]

    return paginate_list(sort_cached_items(items, sort), page, limit, label="genres")


@app.get("/artists")
def get_artists(
    page: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=500),
    search: Optional[str] = None,
    sort: str = Query("name_asc"),
) -> Dict[str, Any]:
    wait_for_core_data()
    items = artist_counts_cache

    if search:
        s = search.strip().lower()
        items = [item for item in items if s in str(item["name"]).lower()]

    return paginate_list(sort_cached_items(items, sort), page, limit, label="artists")


@app.get("/albums")
def get_albums(
    page: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=500),
    search: Optional[str] = None,
    sort: str = Query("name_asc"),
) -> Dict[str, Any]:
    wait_for_core_data()
    items = album_counts_cache

    if search:
        s = search.strip().lower()
        items = [item for item in items if s in str(item["name"]).lower()]

    return paginate_list(sort_cached_items(items, sort), page, limit, label="albums")


@app.get("/tracks")
def get_tracks(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=200),
    sort: str = Query("title_asc"),
    genre: Optional[str] = None,
    artist: Optional[str] = None,
    album: Optional[str] = None,
    search: Optional[str] = None,
) -> Dict[str, Any]:
    df = get_filtered_tracks_df(genre=genre, artist=artist, album=album, search=search)
    df = sort_tracks_df(df, sort)

    paged = paginate_df(df, page, limit)
    result_tracks = [build_track_card(row) for _, row in paged["df"].iterrows()]

    return {
        "tracks": result_tracks,
        "page": paged["page"],
        "limit": paged["limit"],
        "total_results": paged["total_results"],
        "total_pages": paged["total_pages"],
        "has_next": paged["has_next"],
        "has_prev": paged["has_prev"],
    }


@app.get("/artists/{artist_name}")
def get_artist_detail(artist_name: str) -> Dict[str, Any]:
    df = get_filtered_tracks_df(artist=artist_name)
    if df.empty:
        raise HTTPException(status_code=404, detail="Artist not found.")

    df = sort_tracks_df(df, "album_asc").sort_values(
        by=["album", "title"],
        key=lambda col: col.fillna("").astype(str).str.lower(),
    )

    tracks = [build_track_card(row) for _, row in df.iterrows()]

    albums = sorted(
        {str(row["album"]).strip() for _, row in df.iterrows() if str(row["album"]).strip()},
        key=lambda value: value.lower(),
    )

    genres = sorted(
        {genre for genres_list in df["genres"] for genre in genres_list if str(genre).strip()},
        key=lambda value: value.lower(),
    )

    return {
        "name": artist_name,
        "track_count": len(tracks),
        "album_count": len(albums),
        "albums": albums,
        "genres": genres,
        "tracks": tracks,
    }


@app.get("/albums/{album_name}")
def get_album_detail(album_name: str) -> Dict[str, Any]:
    df = get_filtered_tracks_df(album=album_name)
    if df.empty:
        raise HTTPException(status_code=404, detail="Album not found.")

    df = sort_tracks_df(df, "title_asc")
    tracks = [build_track_card(row) for _, row in df.iterrows()]

    artists = sorted(
        {str(row["artist"]).strip() for _, row in df.iterrows() if str(row["artist"]).strip()},
        key=lambda value: value.lower(),
    )

    genres = sorted(
        {genre for genres_list in df["genres"] for genre in genres_list if str(genre).strip()},
        key=lambda value: value.lower(),
    )

    total_seconds = 0.0
    for _, row in df.iterrows():
        try:
            total_seconds += float(row["duration"])
        except Exception:
            pass

    return {
        "name": album_name,
        "track_count": len(tracks),
        "artist_count": len(artists),
        "artists": artists,
        "genres": genres,
        "total_duration_seconds": total_seconds,
        "tracks": tracks,
    }


@app.get("/tracks/{track_id}")
def get_track(track_id: int, model: str = DEFAULT_MODEL) -> Dict[str, Any]:
    wait_for_core_data()
    get_model_config(model)

    track_row = track_row_by_id.get(track_id)
    if track_row is None:
        raise HTTPException(status_code=404, detail="Track not found.")

    model_neighbors = neighbors_lookup_by_model.get(model)
    if model_neighbors is None:
        raise HTTPException(status_code=500, detail=f"Neighbors not loaded for model '{model}'.")

    related = model_neighbors.get(track_id, [])
    recommendations: List[Dict[str, Any]] = []

    for nbr in related:
        nbr_id = nbr["neighbor_track_id"]
        nbr_row = track_row_by_id.get(nbr_id)
        if nbr_row is None:
            continue

        recommendations.append({
            "track_id": int(clean_scalar(nbr_row["track_id"])),
            "title": clean_scalar(nbr_row["title"]),
            "artist": clean_scalar(nbr_row["artist"]),
            "album": clean_scalar(nbr_row["album"]),
            "duration": clean_scalar(nbr_row["duration"]),
            "genres": nbr_row["genres"] if isinstance(nbr_row["genres"], list) else [],
            "score": nbr["score"],
            "rank": nbr["rank"],
        })

    return build_track_detail(track_row, model, recommendations)


@app.post("/upload-search")
async def upload_search(
    file: UploadFile = File(...),
    start_seconds: float = Form(...),
    duration_seconds: float = Form(UPLOAD_DEFAULT_DURATION_SECONDS),
    model: str = Form(DEFAULT_MODEL),
) -> Dict[str, Any]:
    wait_for_upload_assets()
    config = get_model_config(model)

    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")

    suffix = Path(file.filename).suffix.lower()
    allowed = {".mp3", ".wav", ".m4a", ".flac", ".ogg"}

    if suffix not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    duration_seconds = float(duration_seconds)
    start_seconds = float(start_seconds)

    if duration_seconds <= 0:
        raise HTTPException(status_code=400, detail="duration_seconds must be positive.")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        temp_input_path = tmp.name
        contents = await file.read()
        tmp.write(contents)

    try:
        hits = search_uploaded_clip_with_model(
            model=model,
            source_path=temp_input_path,
            start_seconds=start_seconds,
            duration_seconds=duration_seconds,
            top_k=UPLOAD_TOP_K,
        )

        return {
            "query": {
                "filename": file.filename,
                "start_seconds": start_seconds,
                "duration_seconds": duration_seconds,
                "model": model,
                "model_label": config["label"],
            },
            "results": hydrate_search_results(hits),
        }

    finally:
        try:
            os.remove(temp_input_path)
        except Exception:
            pass
