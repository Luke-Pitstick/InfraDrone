"""Minimal CPU HTTP API. Deploy with MODAL_PROFILE=lukepitstick06 modal deploy modal_api.py."""

import modal
from pathlib import Path

app = modal.App("infradrone-api")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("fastapi==0.136.1", "boto3==1.43.91")
    .add_local_file(Path(__file__).with_name("video_storage.py"), "/root/video_storage.py")
)


@app.function(
    image=image,
    cpu=0.25,
    memory=256,
    gpu=None,
    min_containers=0,
    max_containers=2,
    scaledown_window=60,
    timeout=30,
    secrets=[modal.Secret.from_name("infradrone-r2-read")],
)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def api():
    import logging
    from fastapi import FastAPI, HTTPException, Query, Response
    from fastapi.middleware.cors import CORSMiddleware
    from botocore.exceptions import BotoCoreError, ClientError
    import video_storage

    web = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    # Public read-only API; R2 itself stays private and credentials stay server-side.
    web.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["GET"],
        allow_headers=["Accept", "Content-Type"],
    )

    @web.get("/health")
    async def health():
        return {"status": "ok"}

    @web.get("/")
    async def root():
        return {"service": "infradrone-api", "health": "/health", "videos": "/videos"}

    def storage_error(error):
        if isinstance(error, ClientError):
            code = error.response["Error"]["Code"]
            if code in ("404", "NoSuchKey", "NotFound"):
                return HTTPException(404, "Video not found")
            if code in ("InvalidArgument", "InvalidToken"):
                return HTTPException(400, "Invalid storage query")
        logging.getLogger(__name__).exception("R2 request failed")
        return HTTPException(503, "Video storage unavailable")

    # Synchronous boto3 calls run in FastAPI's thread pool.
    @web.get("/videos")
    def videos(response: Response, limit: int = Query(50, ge=1, le=100),
               cursor: str | None = Query(None, max_length=4096),
               prefix: str = Query("", max_length=512)):
        response.headers["Cache-Control"] = "no-store"
        try:
            return video_storage.list_videos(limit, cursor, prefix)
        except (ClientError, BotoCoreError) as error:
            raise storage_error(error) from error

    @web.get("/videos/playback")
    def video_playback(response: Response, key: str = Query(..., max_length=1024)):
        response.headers["Cache-Control"] = "no-store"
        try:
            return video_storage.playback(key)
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
        except (ClientError, BotoCoreError) as error:
            raise storage_error(error) from error

    return web
