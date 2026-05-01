import ast
import math
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import boto3
import faiss
import librosa
import numpy as np
import pandas as pd
import torch
from botocore.exceptions import BotoCoreError, ClientError
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from panns_inference import AudioTagging
from sklearn.preprocessing import normalize
from transformers import ClapAudioModelWithProjection, ClapProcessor

TRACK_FILE = "https://music-sim-capstone-data.s3.us-east-1.amazonaws.com/raw/metadata/raw_tracks.csv"
S3_BUCKET = "music-sim-capstone-data"
DEFAULT_MODEL = "panns"

MODEL_CONFIGS = {
    "panns": {
        "neighbor_file": "https://music-sim-capstone-data.s3.us-east-1.amazonaws.com/neighbors/panns/panns_top20_neighbors.csv",
        "label": "PANNs",
    },
    "clap": {
        "neighbor_file": "https://music-sim-capstone-data.s3.us-east-1.amazonaws.com/neighbors/clap/clap_top20_neighbors.csv",
        "label": "CLAP",
    },
}

PANNS_INDEX_S3_KEY = "embeddings/panns/panns_faiss.index"
PANNS_TRACK_IDS_S3_KEY = "embeddings/panns/track_ids.npy"

CLAP_INDEX_S3_KEY = "embeddings/clap/clap_faiss.index"
CLAP_TRACK_IDS_S3_KEY = "embeddings/clap/track_ids.npy"
CLAP_MEAN_S3_KEY = "embeddings/clap/clap_mean.npy"

LOCAL_ARTIFACT_DIR = Path("model_artifacts")
LOCAL_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

CLAP_MEAN_FILE = LOCAL_ARTIFACT_DIR / "clap_mean.npy"

PANNS_INDEX_FILE = LOCAL_ARTIFACT_DIR / "panns_faiss.index"
PANNS_TRACK_IDS_FILE = LOCAL_ARTIFACT_DIR / "panns_track_ids.npy"

CLAP_INDEX_FILE = LOCAL_ARTIFACT_DIR / "clap_faiss.index"
CLAP_TRACK_IDS_FILE = LOCAL_ARTIFACT_DIR / "clap_track_ids.npy"

CLAP_CHECKPOINT = os.getenv("CLAP_CHECKPOINT", "laion/larger_clap_music")

PANNS_TARGET_SR = 32000
CLAP_TARGET_SR = 48000
UPLOAD_DEFAULT_DURATION_SECONDS = 30.0
UPLOAD_TOP_K = 20

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
clap_mean: Optional[np.ndarray] = None

core_data_ready = False
core_data_loading = False
core_data_error: Optional[str] = None

upload_assets_ready = False
upload_assets_loading = False
upload_assets_error: Optional[str] = None

core_data_lock = threading.Lock()
core_data_event = threading.Event()

upload_assets_lock = threading.Lock()
upload_assets_event = threading.Event()

TRACK_DETAIL_FIELDS = [
    "album_id",
    "album_title",
    "artist_id",
    "artist_name",
    "artist_url",
    "artist_website",
    "license_image_file",
    "license_title",
    "license_url",
    "tags",
    "track_composer",
    "track_copyright_c",
    "track_copyright_p",
    "track_date_created",
    "track_date_recorded",
    "track_disc_number",
    "track_duration",
    "track_explicit",
    "track_explicit_notes",
    "track_genres",
    "track_information",
    "track_instrumental",
    "track_language_code",
    "track_lyricist",
    "track_number",
    "track_publisher",
    "track_title",
    "track_url",
]


def safe_literal(value: Any) -> List[Any]:
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return value
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
        if parsed:
            raw_items = parsed
        else:
            raw_items = [part.strip() for part in stripped.split(",")]
    else:
        raw_items = [value]

    cleaned: List[str] = []
    seen = set()

    for item in raw_items:
        if isinstance(item, dict):
            text = (
                item.get("tag")
                or item.get("name")
                or item.get("title")
                or item.get("label")
            )
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
    padded = f"{track_id:06d}"
    return f"https://{S3_BUCKET}.s3.us-east-1.amazonaws.com/raw/audio/{padded}.mp3"


def build_spectrogram_url(track_id: int) -> str:
    padded = f"{track_id:06d}"
    return f"https://{S3_BUCKET}.s3.us-east-1.amazonaws.com/processed/spectrograms/{padded}_spectrogram.png"


def paginate_list(items: List[Any], page: int, limit: int, label: str = "items") -> Dict[str, Any]:
    total = len(items)
    total_pages = max(1, math.ceil(total / limit)) if limit > 0 else 1

    if page < 1:
        page = 1

    start = (page - 1) * limit
    end = start + limit
    paged_items = items[start:end]

    return {
        label: paged_items,
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

    if page < 1:
        page = 1

    start = (page - 1) * limit
    end = start + limit
    page_df = df.iloc[start:end]

    return {
        "df": page_df,
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

    genre_rows: List[str] = []
    for genres in df["genres"]:
        for genre in genres:
            genre_name = str(genre).strip()
            if genre_name:
                genre_rows.append(genre_name)

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

    artist_counts_cache = [
        {"name": row["artist"], "track_count": int(row["track_count"])}
        for _, row in artist_counts.iterrows()
    ]
    album_counts_cache = [
        {"name": row["album"], "track_count": int(row["track_count"])}
        for _, row in album_counts.iterrows()
    ]
    genre_counts_cache = [
        {"name": row["genre"], "track_count": int(row["track_count"])}
        for _, row in genre_counts_df.iterrows()
    ]


def sort_cached_items(items: List[Dict[str, Any]], sort: str) -> List[Dict[str, Any]]:
    if sort == "name_asc":
        return sorted(items, key=lambda item: str(item.get("name", "")).lower())

    if sort == "name_desc":
        return sorted(items, key=lambda item: str(item.get("name", "")).lower(), reverse=True)

    if sort == "tracks_desc":
        return sorted(
            items,
            key=lambda item: (-int(item.get("track_count", 0)), str(item.get("name", "")).lower()),
        )

    if sort == "tracks_asc":
        return sorted(
            items,
            key=lambda item: (int(item.get("track_count", 0)), str(item.get("name", "")).lower()),
        )

    raise HTTPException(status_code=400, detail="Invalid sort value.")


def sort_tracks_df(df: pd.DataFrame, sort: str) -> pd.DataFrame:
    if sort == "title_asc":
        return df.sort_values("title", key=lambda col: col.fillna("").astype(str).str.lower())

    if sort == "title_desc":
        return df.sort_values(
            "title",
            key=lambda col: col.fillna("").astype(str).str.lower(),
            ascending=False,
        )

    if sort == "artist_asc":
        return df.sort_values("artist", key=lambda col: col.fillna("").astype(str).str.lower())

    if sort == "artist_desc":
        return df.sort_values(
            "artist",
            key=lambda col: col.fillna("").astype(str).str.lower(),
            ascending=False,
        )

    if sort == "album_asc":
        return df.sort_values("album", key=lambda col: col.fillna("").astype(str).str.lower())

    if sort == "album_desc":
        return df.sort_values(
            "album",
            key=lambda col: col.fillna("").astype(str).str.lower(),
            ascending=False,
        )

    if sort == "genre_asc":
        temp = df.assign(primary_genre=df["genres"].apply(lambda g: g[0] if g else ""))
        return temp.sort_values(
            "primary_genre",
            key=lambda col: col.fillna("").astype(str).str.lower(),
        ).drop(columns=["primary_genre"])

    if sort == "genre_desc":
        temp = df.assign(primary_genre=df["genres"].apply(lambda g: g[0] if g else ""))
        return temp.sort_values(
            "primary_genre",
            key=lambda col: col.fillna("").astype(str).str.lower(),
            ascending=False,
        ).drop(columns=["primary_genre"])

    raise HTTPException(status_code=400, detail="Invalid sort value.")


def ensure_local_artifact(local_path: Path, s3_key: str) -> None:
    if local_path.exists():
        return

    local_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        s3_client.download_file(S3_BUCKET, s3_key, str(local_path))
    except (BotoCoreError, ClientError) as e:
        raise RuntimeError(f"Failed to download s3://{S3_BUCKET}/{s3_key}: {e}") from e


def ensure_upload_artifacts_loaded() -> None:
    global panns_model, clap_model, clap_processor, clap_mean

    ensure_local_artifact(PANNS_INDEX_FILE, PANNS_INDEX_S3_KEY)
    ensure_local_artifact(PANNS_TRACK_IDS_FILE, PANNS_TRACK_IDS_S3_KEY)
    ensure_local_artifact(CLAP_INDEX_FILE, CLAP_INDEX_S3_KEY)
    ensure_local_artifact(CLAP_TRACK_IDS_FILE, CLAP_TRACK_IDS_S3_KEY)
    ensure_local_artifact(CLAP_MEAN_FILE, CLAP_MEAN_S3_KEY)
 
    if clap_mean is None:
        clap_mean = np.load(CLAP_MEAN_FILE).astype(np.float32)

    if "panns" not in faiss_index_by_model:
        faiss_index_by_model["panns"] = faiss.read_index(str(PANNS_INDEX_FILE))
        track_ids_by_model["panns"] = np.load(PANNS_TRACK_IDS_FILE, allow_pickle=True)

    if "clap" not in faiss_index_by_model:
        faiss_index_by_model["clap"] = faiss.read_index(str(CLAP_INDEX_FILE))
        track_ids_by_model["clap"] = np.load(CLAP_TRACK_IDS_FILE, allow_pickle=True)

    if panns_model is None:
        panns_model = AudioTagging(device=DEVICE)

    if clap_model is None or clap_processor is None:
        clap_model = ClapAudioModelWithProjection.from_pretrained(CLAP_CHECKPOINT).to(DEVICE)
        clap_processor = ClapProcessor.from_pretrained(CLAP_CHECKPOINT)
        clap_model.eval()

    


def _background_load_upload_assets() -> None:
    global upload_assets_ready, upload_assets_loading, upload_assets_error

    try:
        wait_for_core_data()
        ensure_upload_artifacts_loaded()
        upload_assets_ready = True
        upload_assets_error = None
        print("Upload-search assets ready.")
    except Exception as e:
        upload_assets_ready = False
        upload_assets_error = str(e)
        print(f"Upload asset preload failed: {e}")
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

        thread = threading.Thread(
            target=_background_load_upload_assets,
            daemon=True,
        )
        thread.start()


def wait_for_upload_assets(timeout: float = 1200.0) -> None:
    wait_for_core_data(timeout=timeout)

    if upload_assets_ready:
        return

    start_upload_assets_background_load()

    finished = upload_assets_event.wait(timeout=timeout)
    if not finished:
        raise HTTPException(
            status_code=503,
            detail="Upload search assets are still loading. Please try again shortly.",
        )

    if not upload_assets_ready:
        raise HTTPException(
            status_code=503,
            detail=f"Upload search assets failed to load: {upload_assets_error or 'unknown error'}",
        )


def _build_core_data() -> None:
    global tracks_df, track_row_by_id, neighbors_lookup_by_model
    global artist_counts_cache, album_counts_cache, genre_counts_cache

    df = pd.read_csv(TRACK_FILE, low_memory=False)

    if "track_id" not in df.columns:
        if "Unnamed: 0" in df.columns:
            df = df.rename(columns={"Unnamed: 0": "track_id"})
        elif "id" in df.columns:
            df = df.rename(columns={"id": "track_id"})
        else:
            raise RuntimeError("No track_id column found in tracks csv.")

    df["track_id"] = df["track_id"].apply(clean_track_id)
    df = df[df["track_id"].notna()].copy()
    df["track_id"] = df["track_id"].astype(int)

    for field in TRACK_DETAIL_FIELDS:
        if field not in df.columns:
            df[field] = None

    df["title"] = df.apply(
        lambda row: first_nonempty(
            [row.get("track_title"), row.get("title"), row.get("name")],
            default="Unknown Title",
        ),
        axis=1,
    )

    df["artist"] = df.apply(
        lambda row: first_nonempty(
            [row.get("artist_name"), row.get("artist"), row.get("track_artist")],
            default="Unknown Artist",
        ),
        axis=1,
    )

    df["album"] = df.apply(
        lambda row: first_nonempty(
            [row.get("album_title"), row.get("album")],
            default="Unknown Album",
        ),
        axis=1,
    )

    scalar_fields = [
        "track_title",
        "artist_name",
        "album_title",
        "artist_url",
        "artist_website",
        "license_image_file",
        "license_title",
        "license_url",
        "track_composer",
        "track_copyright_c",
        "track_copyright_p",
        "track_date_created",
        "track_date_recorded",
        "track_explicit_notes",
        "track_information",
        "track_language_code",
        "track_lyricist",
        "track_publisher",
        "track_url",
        "album_id",
        "artist_id",
        "track_disc_number",
        "track_duration",
        "track_explicit",
        "track_instrumental",
        "track_number",
    ]

    for field in scalar_fields:
        df[field] = df[field].apply(clean_scalar)

    df["track_genres"] = df["track_genres"].apply(parse_genres)
    df["genres"] = df["track_genres"]
    df["tags"] = df["tags"].apply(parse_string_list)
    df["duration"] = df["track_duration"]

    df["audio_url"] = df["track_id"].apply(build_audio_url)
    df["spectrogram_url"] = df["track_id"].apply(build_spectrogram_url)

    keep_columns = [
        "track_id",
        "title",
        "artist",
        "album",
        "duration",
        "genres",
        "audio_url",
        "spectrogram_url",
    ] + TRACK_DETAIL_FIELDS

    built_tracks_df = df[keep_columns].copy()

    valid_track_ids = set()
    grouped_neighbors_by_model: Dict[str, Dict[int, List[Dict[str, Any]]]] = {}

    for model_name, config in MODEL_CONFIGS.items():
        neighbors_df = pd.read_csv(config["neighbor_file"])
        neighbors_df["track_id"] = neighbors_df["track_id"].apply(clean_track_id)
        neighbors_df["neighbor_track_id"] = neighbors_df["neighbor_track_id"].apply(clean_track_id)
        neighbors_df = neighbors_df.dropna(subset=["track_id", "neighbor_track_id"]).copy()
        neighbors_df["track_id"] = neighbors_df["track_id"].astype(int)
        neighbors_df["neighbor_track_id"] = neighbors_df["neighbor_track_id"].astype(int)

        valid_track_ids.update(neighbors_df["track_id"].unique())

        neighbors_df = neighbors_df.sort_values(["track_id", "rank"])
        model_lookup: Dict[int, List[Dict[str, Any]]] = {}

        for track_id_value, group in neighbors_df.groupby("track_id", sort=False):
            model_lookup[int(track_id_value)] = [
                {
                    "neighbor_track_id": int(row["neighbor_track_id"]),
                    "score": float(row["score"]) if not pd.isna(row["score"]) else None,
                    "rank": int(row["rank"]) if not pd.isna(row["rank"]) else None,
                }
                for _, row in group.iterrows()
            ]

        grouped_neighbors_by_model[model_name] = model_lookup

    built_tracks_df = built_tracks_df[built_tracks_df["track_id"].isin(valid_track_ids)].copy()

    built_track_row_by_id = {
        int(row["track_id"]): row.to_dict()
        for _, row in built_tracks_df.iterrows()
    }

    precompute_lists(built_tracks_df)

    tracks_df = built_tracks_df
    track_row_by_id = built_track_row_by_id
    neighbors_lookup_by_model = grouped_neighbors_by_model

    print("Loaded tracks:", len(tracks_df))
    print("Loaded artists:", len(artist_counts_cache))
    print("Loaded albums:", len(album_counts_cache))
    print("Loaded genres:", len(genre_counts_cache))
    print("Loaded models:", list(neighbors_lookup_by_model.keys()))
    print("Upload-search device:", DEVICE)


def _background_load_core_data() -> None:
    global core_data_ready, core_data_loading, core_data_error

    try:
        _build_core_data()
        core_data_ready = True
        core_data_error = None
        print("Core track data ready.")
        print("Starting upload-search asset preload in background...")
        start_upload_assets_background_load()
    except Exception as e:
        core_data_ready = False
        core_data_error = str(e)
        print(f"Core data preload failed: {e}")
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

        thread = threading.Thread(
            target=_background_load_core_data,
            daemon=True,
        )
        thread.start()


def wait_for_core_data(timeout: float = 1200.0) -> None:
    if core_data_ready:
        return

    start_core_data_background_load()

    finished = core_data_event.wait(timeout=timeout)
    if not finished:
        raise HTTPException(
            status_code=503,
            detail="Core track data is still loading. Please try again shortly.",
        )

    if not core_data_ready:
        raise HTTPException(
            status_code=503,
            detail=f"Core track data failed to load: {core_data_error or 'unknown error'}",
        )


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
            df["title"].fillna("").astype(str).str.lower().str.contains(s, na=False)
            | df["artist"].fillna("").astype(str).str.lower().str.contains(s, na=False)
            | df["album"].fillna("").astype(str).str.lower().str.contains(s, na=False)
        ]

    return df


def prepare_query_waveform(
    source_path: str,
    target_sr: int,
    start_seconds: float,
    duration_seconds: float,
) -> np.ndarray:
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


def search_uploaded_clip_with_panns(
    source_path: str,
    start_seconds: float,
    duration_seconds: float,
    top_k: int = UPLOAD_TOP_K,
) -> List[Dict[str, Any]]:
    waveform = prepare_query_waveform(
        source_path=source_path,
        target_sr=PANNS_TARGET_SR,
        start_seconds=start_seconds,
        duration_seconds=duration_seconds,
    )

    batch = waveform[np.newaxis, :]
    _, batch_embeddings = panns_model.inference(batch)
    batch_embeddings = np.asarray(batch_embeddings, dtype=np.float32)
    batch_embeddings = normalize(batch_embeddings, norm="l2").astype(np.float32)

    scores, neighbors = faiss_index_by_model["panns"].search(batch_embeddings, top_k)

    hits: List[Dict[str, Any]] = []
    for score, idx in zip(scores[0], neighbors[0]):
        if idx < 0:
            continue

        track_id = int(track_ids_by_model["panns"][idx])
        hits.append({
            "track_id": track_id,
            "score": float(score),
        })

    return hits


def search_uploaded_clip_with_clap(
    source_path: str,
    start_seconds: float,
    duration_seconds: float,
    top_k: int = UPLOAD_TOP_K,
) -> List[Dict[str, Any]]:
    
    if clap_mean is None:
        raise HTTPException(status_code=500, detail="CLAP mean not loaded.")
    
    waveform = prepare_query_waveform(
        source_path=source_path,
        target_sr=CLAP_TARGET_SR,
        start_seconds=start_seconds,
        duration_seconds=duration_seconds,
    )

    inputs = clap_processor(
        audio=[waveform],
        sampling_rate=CLAP_TARGET_SR,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = clap_model(**inputs)
        embeddings = outputs.audio_embeds.detach().cpu().numpy().astype(np.float32)

    embeddings = embeddings - clap_mean
    embeddings = normalize(embeddings, norm="l2").astype(np.float32)
    scores, neighbors = faiss_index_by_model["clap"].search(embeddings, top_k)

    hits: List[Dict[str, Any]] = []
    for score, idx in zip(scores[0], neighbors[0]):
        if idx < 0:
            continue

        track_id = int(track_ids_by_model["clap"][idx])
        hits.append({
            "track_id": track_id,
            "score": float(score),
        })

    return hits


def hydrate_search_results(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    for i, hit in enumerate(hits):
        track_id = int(hit["track_id"])
        row = track_row_by_id.get(track_id)
        if row is None:
            continue

        results.append(
            {
                "track_id": int(clean_scalar(row["track_id"])),
                "title": clean_scalar(row["title"]),
                "artist": clean_scalar(row["artist"]),
                "album": clean_scalar(row["album"]),
                "duration": clean_scalar(row["duration"]),
                "genres": row["genres"] if isinstance(row["genres"], list) else [],
                "score": float(hit["score"]) if hit.get("score") is not None else None,
                "rank": i + 1,
            }
        )

    return results


@app.on_event("startup")
def startup() -> None:
    print("Starting core data preload in background...")
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
        "device": DEVICE,
    }


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "message": "Music Similarity API is running",
        "default_model": DEFAULT_MODEL,
        "available_models": list(MODEL_CONFIGS.keys()),
        "core_data_ready": core_data_ready,
        "core_data_loading": core_data_loading,
        "core_data_error": core_data_error,
        "upload_assets_ready": upload_assets_ready,
        "upload_assets_loading": upload_assets_loading,
        "upload_assets_error": upload_assets_error,
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

    items = sort_cached_items(items, sort)
    return paginate_list(items, page, limit, label="genres")


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

    items = sort_cached_items(items, sort)
    return paginate_list(items, page, limit, label="artists")


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

    items = sort_cached_items(items, sort)
    return paginate_list(items, page, limit, label="albums")


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
    df = get_filtered_tracks_df(
        genre=genre,
        artist=artist,
        album=album,
        search=search,
    )
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
        {
            str(row["album"]).strip()
            for _, row in df.iterrows()
            if str(row["album"]).strip()
        },
        key=lambda value: value.lower(),
    )

    genres = sorted(
        {
            genre
            for genres_list in df["genres"]
            for genre in genres_list
            if str(genre).strip()
        },
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
        {
            str(row["artist"]).strip()
            for _, row in df.iterrows()
            if str(row["artist"]).strip()
        },
        key=lambda value: value.lower(),
    )

    genres = sorted(
        {
            genre
            for genres_list in df["genres"]
            for genre in genres_list
            if str(genre).strip()
        },
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

    if model not in MODEL_CONFIGS:
        raise HTTPException(status_code=400, detail=f"Unknown model '{model}'.")

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

        recommendations.append(
            {
                "track_id": int(clean_scalar(nbr_row["track_id"])),
                "title": clean_scalar(nbr_row["title"]),
                "artist": clean_scalar(nbr_row["artist"]),
                "album": clean_scalar(nbr_row["album"]),
                "duration": clean_scalar(nbr_row["duration"]),
                "genres": nbr_row["genres"] if isinstance(nbr_row["genres"], list) else [],
                "score": nbr["score"],
                "rank": nbr["rank"],
            }
        )

    return build_track_detail(track_row, model, recommendations)


@app.post("/upload-search")
async def upload_search(
    file: UploadFile = File(...),
    start_seconds: float = Form(...),
    duration_seconds: float = Form(UPLOAD_DEFAULT_DURATION_SECONDS),
) -> Dict[str, Any]:
    wait_for_upload_assets()

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
        panns_hits = search_uploaded_clip_with_panns(
            source_path=temp_input_path,
            start_seconds=start_seconds,
            duration_seconds=duration_seconds,
            top_k=UPLOAD_TOP_K,
        )
        clap_hits = search_uploaded_clip_with_clap(
            source_path=temp_input_path,
            start_seconds=start_seconds,
            duration_seconds=duration_seconds,
            top_k=UPLOAD_TOP_K,
        )

        panns_results = hydrate_search_results(panns_hits)
        clap_results = hydrate_search_results(clap_hits)

        return {
            "query": {
                "filename": file.filename,
                "start_seconds": start_seconds,
                "duration_seconds": duration_seconds,
            },
            "results": {
                "panns": panns_results,
                "clap": clap_results,
            },
        }

    finally:
        try:
            os.remove(temp_input_path)
        except Exception:
            pass