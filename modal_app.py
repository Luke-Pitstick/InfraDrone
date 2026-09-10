"""Submit a recording to Modal and retrieve durable GPU inference artifacts.

Run from the engine directory with `python -m modal run modal_app.py --help`.
Only inference source, the selected weights, and recording sidecars are uploaded.
"""

from datetime import datetime, timezone
import hashlib
import json
from math import isfinite
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from uuid import UUID, uuid4

import modal

ROOT = Path(__file__).resolve().parent
app = modal.App("road-survey")
models = modal.Volume.from_name("road-survey-models", create_if_missing=True)
recordings = modal.Volume.from_name("road-survey-recordings", create_if_missing=True)
results = modal.Volume.from_name("road-survey-results", create_if_missing=True)

# Inference dependencies pinned to this project's uv.lock; training tools excluded.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.11.0", "torchvision==0.26.0", "ultralytics==8.4.41",
        "av==18.1.0", "numpy==2.4.4", "scipy==1.17.1", "scikit-image==0.26.0",
        "opencv-python==4.13.0.92", "pillow==12.2.0", "pyyaml==6.0.3",
    )
    .env({"YOLO_CONFIG_DIR": "/tmp/ultralytics", "MPLCONFIGDIR": "/tmp/matplotlib"})
    .add_local_dir(ROOT / "src" / "engine", remote_path="/root/src/engine",
                   ignore=["**/__pycache__/**", "**/*.pyc"])
)


def checked_job_id(value: str) -> str:
    return str(UUID(value))


def file_hash(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


@app.cls(image=image, gpu="L4", timeout=1200, max_containers=1, scaledown_window=60,
         retries=0, volumes={"/models": models, "/recordings": recordings, "/results": results})
class SurveyWorker:
    model_hash: str = modal.parameter()
    confidence: str = modal.parameter(default="0.25")
    triangulation: bool = modal.parameter(default=False)
    motion_scale_source: str = modal.parameter(default="camera_height")

    @modal.enter()
    def load_model(self):
        import torch
        from src.engine.video_pipeline import VideoPipeline

        if len(self.model_hash) != 64 or any(c not in "0123456789abcdef" for c in self.model_hash):
            raise ValueError("Invalid model hash")
        models.reload()
        path = Path("/models") / f"{self.model_hash}.pt"
        if file_hash(path) != self.model_hash:
            raise ValueError("Uploaded model checksum does not match")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in the GPU worker")
        self.gpu_name = torch.cuda.get_device_name(0)
        self.pipeline = VideoPipeline(path, device="cuda:0", confidence_threshold=float(self.confidence),
                                      triangulation=self.triangulation, motion_scale_source=self.motion_scale_source)

    @modal.method()
    def analyze(self, job_id: str, sample_fps: float = 2.0) -> dict:
        from src.engine.recording_io import load_video

        job_id = checked_job_id(job_id)
        if not isfinite(sample_fps) or sample_fps <= 0:
            raise ValueError("sample_fps must be finite and positive")
        recordings.reload()
        results.reload()
        job_dir = Path("/results") / job_id
        job_dir.mkdir(exist_ok=False)
        status_path = job_dir / "job.json"
        status = {"job_id": job_id, "status": "running", "model_sha256": self.model_hash,
                  "gpu": self.gpu_name, "started_at": datetime.now(timezone.utc).isoformat()}
        status_path.write_text(json.dumps(status, indent=2))
        results.commit()
        try:
            video = load_video(Path("/recordings") / job_id)
            manifest_path = self.pipeline.write(video, job_dir / "output", sample_fps=sample_fps)
            manifest = json.loads(manifest_path.read_text())
            status.update(status="complete", damage_count=manifest["damage_count"],
                          manifest_path=f"{job_id}/output/manifest.json")
        except Exception as error:
            status.update(status="failed", error=str(error))
            raise
        finally:
            status["finished_at"] = datetime.now(timezone.utc).isoformat()
            status_path.write_text(json.dumps(status, indent=2))
            results.commit()
        return status


def download_job(job_id: str, output_dir: Path) -> Path:
    """Stream files from the results volume; never overwrite an existing download."""
    job_id = checked_job_id(job_id)
    parent = Path(output_dir).resolve()
    target = parent / job_id
    if target.exists():
        raise FileExistsError(target)
    status = json.loads(b"".join(results.read_file(f"/{job_id}/job.json")))
    if status["status"] not in ("complete", "failed"):
        raise RuntimeError("Job is still running; fetch its results after it finishes")
    parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{job_id}-", dir=parent) as temporary:
        staging = Path(temporary) / "job"
        staging.mkdir()
        for entry in results.iterdir(f"/{job_id}", recursive=True):
            if entry.type != modal.volume.FileEntryType.FILE:
                continue
            relative = PurePosixPath(entry.path.lstrip("/")).relative_to(job_id)
            if ".." in relative.parts:
                raise ValueError("Unexpected result path")
            destination = staging.joinpath(*relative.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as output:
                for chunk in results.read_file(entry.path):
                    output.write(chunk)
        if not (staging / "job.json").is_file():
            raise RuntimeError("Results volume did not return job.json")
        staging.rename(target)
    return target


@app.local_entrypoint()
def main(recording: str, weights: str = "src/ml/models/weights/segmentation/best.pt",
         output_dir: str = "outputs", sample_fps: float = 2.0, confidence: float = 0.25,
         triangulation: bool = False, motion_scale_source: str = "camera_height"):
    """Validate/upload a recording, run GPU inference, and download its results."""
    from src.engine.recording_io import load_video

    if not isfinite(sample_fps) or sample_fps <= 0:
        raise ValueError("sample_fps must be finite and positive")
    if not isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("confidence must be between 0 and 1")
    if motion_scale_source not in ("camera_height", "gps"):
        raise ValueError("motion_scale_source must be camera_height or gps")
    video = load_video(Path(recording))
    if triangulation and (video.calibration is None or video.calibration.road_roi is None):
        raise ValueError("Triangulation requires calibration with a road ROI")
    weights_path = Path(weights).resolve()
    model_hash = file_hash(weights_path)
    job_id = str(uuid4())
    print(f"Job ID: {job_id}", flush=True)
    with models.batch_upload(force=True) as upload:
        upload.put_file(weights_path, f"/{model_hash}.pt")
    with recordings.batch_upload() as upload:
        upload.put_file(video.footage_path, f"/{job_id}/{video.footage_path.name}")
        for name in ("metadata.json", "gps.csv", "calibration.json"):
            sidecar = video.footage_path.parent / name
            if sidecar.is_file():
                upload.put_file(sidecar, f"/{job_id}/{name}")
    print("Uploaded recording and weights; starting GPU inference.", flush=True)
    status = SurveyWorker(model_hash=model_hash, confidence=str(confidence), triangulation=triangulation,
                          motion_scale_source=motion_scale_source).analyze.remote(job_id, sample_fps)
    print(json.dumps(status, indent=2), flush=True)
    print(f"Downloaded results: {download_job(job_id, Path(output_dir))}", flush=True)


@app.local_entrypoint()
def fetch(job_id: str, output_dir: str = "outputs"):
    """Retrieve an existing job without running inference again."""
    print(download_job(job_id, Path(output_dir)))
