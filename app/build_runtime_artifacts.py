import argparse
import gzip
import json
import pickle
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
from urllib.parse import urlparse

import boto3
import pandas as pd


S3_BUCKET = "music-sim-capstone-data"

NEIGHBOR_FILES = {
    "panns": f"https://{S3_BUCKET}.s3.us-east-1.amazonaws.com/neighbors/panns/panns_top20_neighbors.csv",
    "clap": f"https://{S3_BUCKET}.s3.us-east-1.amazonaws.com/neighbors/clap/clap_top20_neighbors.csv",
    "ast": f"https://{S3_BUCKET}.s3.us-east-1.amazonaws.com/neighbors/ast/ast_top20_neighbors.csv",
    "mert": f"https://{S3_BUCKET}.s3.us-east-1.amazonaws.com/neighbors/mert/mert_top20_neighbors.csv",
    "ccmusic": f"https://{S3_BUCKET}.s3.us-east-1.amazonaws.com/neighbors/ccmusic/ccmusic_top20_neighbors.csv",
}

MODEL_REPOS = {
    "clap": "laion/larger_clap_music",
    "ast": "MIT/ast-finetuned-audioset-10-10-0.4593",
    "mert": "m-a-p/MERT-v1-330M",
    "ccmusic": "ccmusic-database/music_genre",
}


def log(message: str) -> None:
    print(message, flush=True)


def parse_models(value: str, valid_models: Iterable[str]) -> List[str]:
    if value == "all":
        return list(valid_models)

    models = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = sorted(set(models) - set(valid_models))
    if unknown:
        raise ValueError(f"Unknown model(s): {', '.join(unknown)}")

    return models


def build_neighbor_lookup(csv_path_or_url: str, output_file: Path, chunksize: int) -> Dict[str, Any]:
    start_time = time.perf_counter()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    lookup: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    total_rows = 0

    log(f"Neighbors: reading {csv_path_or_url}")
    reader = pd.read_csv(
        csv_path_or_url,
        usecols=["track_id", "neighbor_track_id", "rank", "score"],
        dtype={
            "track_id": "int32",
            "neighbor_track_id": "int32",
            "rank": "int16",
            "score": "float32",
        },
        chunksize=chunksize,
    )

    for chunk_number, chunk in enumerate(reader, start=1):
        total_rows += len(chunk)
        for row in chunk.itertuples(index=False):
            lookup[int(row.track_id)].append({
                "neighbor_track_id": int(row.neighbor_track_id),
                "score": float(row.score),
                "rank": int(row.rank),
            })

        log(f"Neighbors: processed chunk {chunk_number} rows={total_rows}")

    normal_lookup = dict(lookup)
    with gzip.open(output_file, "wb", compresslevel=5) as f:
        pickle.dump(normal_lookup, f, protocol=pickle.HIGHEST_PROTOCOL)

    elapsed = time.perf_counter() - start_time
    size_mb = output_file.stat().st_size / 1024 / 1024
    log(
        f"Neighbors: wrote {output_file} "
        f"tracks={len(normal_lookup)} rows={total_rows} size={size_mb:.1f} MB in {elapsed:.1f}s"
    )

    return {
        "path": str(output_file),
        "tracks": len(normal_lookup),
        "rows": total_rows,
        "size_mb": round(size_mb, 1),
        "seconds": round(elapsed, 1),
    }


def save_clap_model(output_dir: Path, repo_id: str) -> Dict[str, Any]:
    from transformers import ClapAudioModelWithProjection, ClapProcessor

    output_dir.mkdir(parents=True, exist_ok=True)
    log(f"Models: downloading CLAP from {repo_id}")
    model = ClapAudioModelWithProjection.from_pretrained(repo_id)
    processor = ClapProcessor.from_pretrained(repo_id)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    return {"path": str(output_dir), "repo_id": repo_id}


def save_ast_model(output_dir: Path, repo_id: str) -> Dict[str, Any]:
    from transformers import ASTModel, AutoFeatureExtractor

    output_dir.mkdir(parents=True, exist_ok=True)
    log(f"Models: downloading AST from {repo_id}")
    model = ASTModel.from_pretrained(repo_id)
    feature_extractor = AutoFeatureExtractor.from_pretrained(repo_id)
    model.save_pretrained(output_dir)
    feature_extractor.save_pretrained(output_dir)
    return {"path": str(output_dir), "repo_id": repo_id}


def save_mert_model(output_dir: Path, repo_id: str) -> Dict[str, Any]:
    from huggingface_hub import snapshot_download

    output_dir.mkdir(parents=True, exist_ok=True)
    log(f"Models: downloading MERT snapshot from {repo_id}")
    snapshot_download(repo_id=repo_id, local_dir=output_dir)
    return {"path": str(output_dir), "repo_id": repo_id}


def save_ccmusic_snapshot(output_dir: Path, repo_id: str) -> Dict[str, Any]:
    from huggingface_hub import snapshot_download

    output_dir.mkdir(parents=True, exist_ok=True)
    log(f"Models: downloading CCMUSIC snapshot from {repo_id}")
    snapshot_download(repo_id=repo_id, local_dir=output_dir)
    return {"path": str(output_dir), "repo_id": repo_id}


def save_model(model_name: str, output_root: Path) -> Dict[str, Any]:
    repo_id = MODEL_REPOS[model_name]
    output_dir = output_root / model_name
    start_time = time.perf_counter()

    if output_dir.exists():
        shutil.rmtree(output_dir)

    if model_name == "clap":
        result = save_clap_model(output_dir, repo_id)
    elif model_name == "ast":
        result = save_ast_model(output_dir, repo_id)
    elif model_name == "mert":
        result = save_mert_model(output_dir, repo_id)
    elif model_name == "ccmusic":
        result = save_ccmusic_snapshot(output_dir, repo_id)
    else:
        raise ValueError(f"Model save is not configured for {model_name}")

    elapsed = time.perf_counter() - start_time
    result["seconds"] = round(elapsed, 1)
    log(f"Models: saved {model_name} to {output_dir} in {elapsed:.1f}s")
    return result


def parse_s3_uri(uri: str) -> Tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError("S3 URI must look like s3://bucket/prefix")
    return parsed.netloc, parsed.path.lstrip("/")


def upload_directory_to_s3(local_dir: Path, s3_uri: str) -> None:
    bucket, prefix = parse_s3_uri(s3_uri)
    s3_client = boto3.client("s3")

    for path in local_dir.rglob("*"):
        if not path.is_file():
            continue

        relative_key = path.relative_to(local_dir).as_posix()
        key = f"{prefix.rstrip('/')}/{relative_key}" if prefix else relative_key
        log(f"Upload: {path} -> s3://{bucket}/{key}")
        s3_client.upload_file(str(path), bucket, key)


def write_manifest(output_dir: Path, manifest: Dict[str, Any]) -> None:
    manifest_file = output_dir / "manifest.json"
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log(f"Manifest: wrote {manifest_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build runtime artifacts for App Runner: neighbor lookup pickles and saved model folders."
    )
    parser.add_argument("--output-dir", default="runtime_artifacts", help="Directory to write generated artifacts.")
    parser.add_argument(
        "--neighbors",
        default="all",
        help="Comma-separated neighbor models to build, 'all', or 'none'.",
    )
    parser.add_argument(
        "--models",
        default="none",
        help="Comma-separated Hugging Face model folders to save, 'all', or 'none'. PANNs is intentionally excluded.",
    )
    parser.add_argument("--chunksize", type=int, default=250_000, help="Rows per CSV chunk for neighbor builds.")
    parser.add_argument("--upload-s3-prefix", help="Optional S3 URI to upload the output directory after building.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "neighbors": {},
        "models": {},
    }

    if args.neighbors != "none":
        neighbor_models = parse_models(args.neighbors, NEIGHBOR_FILES.keys())
        for model_name in neighbor_models:
            output_file = output_dir / "neighbors" / model_name / f"{model_name}_neighbor_lookup.pkl.gz"
            manifest["neighbors"][model_name] = build_neighbor_lookup(
                NEIGHBOR_FILES[model_name],
                output_file,
                args.chunksize,
            )

    if args.models != "none":
        model_names = parse_models(args.models, MODEL_REPOS.keys())
        for model_name in model_names:
            manifest["models"][model_name] = save_model(model_name, output_dir / "models")

    write_manifest(output_dir, manifest)

    if args.upload_s3_prefix:
        upload_directory_to_s3(output_dir, args.upload_s3_prefix)


if __name__ == "__main__":
    main()
