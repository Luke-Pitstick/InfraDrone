# CPU HTTP API and result videos

## Live deployment

Deployed and verified on 2026-09-10:

- API: https://lukepitstick06--infradrone-api-api.modal.run
- Bucket: `infradrone-video-results` (Western North America, Standard storage).
- Cloudflare account: `6fca4fb9d7e6285e06efeac87ebb0320`.
- Modal secrets: `infradrone-r2-read` for the API and `infradrone-r2-write` for producers.

Both credentials are scoped to this bucket. Public bucket access is disabled.
The API provides public read-only browsing and one-hour signed playback URLs;
anyone able to call the API can view its videos. There is no sign-in requirement.
R2 CORS allows GET/HEAD, the Range header, and exposes range/length/ETag headers.

The secrets contain `R2_ENDPOINT_URL`, `R2_BUCKET_NAME`, `R2_ACCESS_KEY_ID`, and
`R2_SECRET_ACCESS_KEY`. Credential values are not stored in project files.

Verified a synthetic MP4 upload through the CPU uploader, live API listing,
signed playback, complete-file checksum, cross-origin headers, 206 byte-range
seeking, and missing-video 404. The synthetic test object was removed afterward.

## Deployment

From `InfraDrone-Engine`, deploy using the existing environment and Modal profile:

```sh
MODAL_PROFILE=lukepitstick06 .venv/bin/python -m modal deploy modal_api.py
```

`modal_api.py` defines the separate `infradrone-api` Modal app. It uses Python 3.12,
FastAPI and boto3, 0.25 CPU and 256 MiB memory per container, explicitly no GPU, and at most
two containers. It scales to zero when idle, so the first request may cold start.
Deploying this file does not import the inference app or upload models/datasets.

- `GET /health` returns `{"status":"ok"}` (process health, no dependency checks).
- `GET /` returns the service name and health path.
- `GET /videos?limit=50&prefix=<job-uuid>/&cursor=<next_cursor>` lists video keys,
  sizes and modification timestamps. Limit is 1-100; prefix is relative to
  `videos/`. Keep paging until `next_cursor` is null, even if a page is empty.
- `GET /videos/playback?key=<URL-encoded-key>` returns a signed R2 GET URL,
  content type, size, and `expires_in: 3600`. Missing objects return 404;
  storage failures return 503. Responses containing URLs are not cached.

## Frontend

```js
const api = "https://lukepitstick06--infradrone-api-api.modal.run";
const listResponse = await fetch(`${api}/videos`);
if (!listResponse.ok) throw new Error("Could not list videos");
const { videos, next_cursor } = await listResponse.json();
if (videos.length) {
  const response = await fetch(`${api}/videos/playback?key=${encodeURIComponent(videos[0].key)}`);
  if (!response.ok) throw new Error("Could not load video");
  const { url } = await response.json();
  document.querySelector("video").src = url;
}
```

Use a `<video controls>` element. R2 serves the bytes and range requests for
seeking directly, so video traffic does not pass through Modal. Fetch a fresh
playback URL after expiry. MP4 should use browser-compatible codecs such as H.264
with AAC audio and `faststart`; the storage helper does not transcode videos.

## Uploading results

Recommended: use the authenticated Modal CLI to upload without retrieving credentials:

```sh
MODAL_PROFILE=lukepitstick06 INFRADRONE_VIDEO_PATH=/absolute/path/result.mp4 \
  .venv/bin/python -m modal run modal_upload.py --job-id <job-uuid>
```

This runs a short-lived CPU-only uploader with the write secret. It prints the
stored object key, which becomes available immediately through `/videos`.


`video_storage.upload_video(path, job_id)` uploads a local MP4/WebM with the correct
content type under `videos/<job-uuid>/<unique-id>.<extension>` and returns its key.
It uses multipart transfers for large files. Supply the same four R2 environment
variables to the producer, using the separate write credential. Never place the
write credential in frontend code or attach it to the public API.

```sh
uv run --with boto3==1.43.91 python video_storage.py result.mp4 --job-id <job-uuid>
```

The existing GPU pipeline produces JSON and masks, not a rendered result video.
Call the upload helper once a producer has created its MP4/WebM; this change does
not run GPU inference, render videos, or upload existing recordings automatically.

## Tests

```sh
uv run --with boto3==1.43.91 --with pytest --with httpx python -m pytest test_video_api.py -q
```

Tests stub R2 to cover pagination, MIME metadata, URL signing, missing objects,
storage failures, input validation, CORS, and rejection of public uploads.

Routes are public. CORS permits GET requests from any frontend origin without
credentials. Restrict origins and add appropriate authentication before adding
private data or mutation endpoints. Interactive docs and OpenAPI are disabled.
