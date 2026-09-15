"""Upload a result video through Modal without downloading R2 credentials."""

import os
from pathlib import Path

import modal

app = modal.App("infradrone-video-upload")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("boto3==1.43.91")
    .add_local_file(Path(__file__).with_name("video_storage.py"), "/root/video_storage.py")
)
if modal.is_local():
    source = Path(os.environ["INFRADRONE_VIDEO_PATH"]).expanduser().resolve(strict=True)
    if source.suffix.lower() not in (".mp4", ".webm") or not source.is_file():
        raise ValueError("INFRADRONE_VIDEO_PATH must point to an MP4 or WebM file")
    remote_path = "/root/input" + source.suffix.lower()
    image = image.add_local_file(source, remote_path)


@app.function(image=image, cpu=0.25, memory=256, gpu=None, timeout=1800,
              max_containers=1, secrets=[modal.Secret.from_name("infradrone-r2-write")])
def upload(job_id: str, path: str):
    from video_storage import upload_video

    return upload_video(path, job_id)


@app.local_entrypoint()
def main(job_id: str):
    from uuid import UUID

    print(upload.remote(str(UUID(job_id)), remote_path))
