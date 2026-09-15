"""Local command and artifact-transfer tests; no Modal network or GPU calls."""

from contextlib import closing
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from uuid import uuid4

import modal_app


class ModalCommandTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.job_id = str(uuid4())

    def result_volume(self, status="complete"):
        files = {
            f"{self.job_id}/job.json": json.dumps({"job_id": self.job_id, "status": status}).encode(),
            f"{self.job_id}/output/manifest.json": b'{"status":"complete"}',
            f"{self.job_id}/output/masks/example.npz": b"binary artifact",
        }
        volume = Mock()
        volume.read_file.side_effect = lambda path: iter([files[path.lstrip("/")][:5], files[path.lstrip("/")][5:]])
        volume.iterdir.return_value = [SimpleNamespace(path=path, type=modal_app.modal.volume.FileEntryType.FILE)
                                       for path in files]
        return volume

    def test_streamed_download_and_no_overwrite(self):
        with patch.object(modal_app, "results", self.result_volume()):
            path = modal_app.download_job(self.job_id, self.root)
            self.assertEqual((path / "output/masks/example.npz").read_bytes(), b"binary artifact")
            self.assertEqual(json.loads((path / "job.json").read_text())["status"], "complete")
            with self.assertRaises(FileExistsError):
                modal_app.download_job(self.job_id, self.root)

    def test_reject_running_job_and_unsafe_paths(self):
        with patch.object(modal_app, "results", self.result_volume("running")):
            with self.assertRaisesRegex(RuntimeError, "still running"):
                modal_app.download_job(self.job_id, self.root)
        volume = self.result_volume()
        volume.iterdir.return_value = [SimpleNamespace(path=f"{self.job_id}/../escape", type=modal_app.modal.volume.FileEntryType.FILE)]
        with patch.object(modal_app, "results", volume):
            with self.assertRaises(ValueError):
                modal_app.download_job(self.job_id, self.root)
        self.assertFalse((self.root / self.job_id).exists())
        self.assertFalse((self.root / "escape").exists())
        with self.assertRaises(ValueError):
            modal_app.download_job("../invalid", self.root)

    def test_failed_download_leaves_no_partial_destination(self):
        volume = self.result_volume()
        volume.iterdir.side_effect = ConnectionError("connection lost")
        with patch.object(modal_app, "results", volume):
            with self.assertRaises(ConnectionError):
                modal_app.download_job(self.job_id, self.root)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_submission_uploads_only_selected_recording_and_weights(self):
        video = self.root / "video.mp4"
        video.write_bytes(b"footage")
        weights = self.root / "best.pt"
        weights.write_bytes(b"checkpoint")
        (self.root / "metadata.json").write_text('{}')
        (self.root / "unrelated.txt").write_text("do not upload")
        model_upload, recording_upload = Mock(), Mock()
        # batch_upload is a synchronous context manager.
        from unittest.mock import MagicMock
        model_context = MagicMock()
        model_context.__enter__.return_value = model_upload
        recording_context = MagicMock()
        recording_context.__enter__.return_value = recording_upload
        with patch("src.engine.recording_io.load_video", return_value=SimpleNamespace(footage_path=video, route_id="road", calibration=SimpleNamespace(road_roi=(0, 0, 10, 10)))), \
             patch.object(modal_app, "models") as models, patch.object(modal_app, "recordings") as recordings, \
             patch.object(modal_app.modal.Cls, "from_name") as lookup, patch.object(modal_app, "download_job") as download:
            models.batch_upload.return_value = model_context
            recordings.batch_upload.return_value = recording_context
            worker = lookup.return_value
            worker.return_value.analyze.remote.return_value = {"status": "complete"}
            modal_app.main(recording=str(self.root), weights=str(weights), output_dir=str(self.root / "out"))
            uploaded = [call.args[0].name for call in recording_upload.put_file.call_args_list]
            self.assertEqual(uploaded, ["video.mp4", "metadata.json"])
            self.assertEqual(model_upload.put_file.call_args.args[1], f"/{hashlib.sha256(b'checkpoint').hexdigest()}.pt")
            worker.return_value.analyze.remote.assert_called_once()
            download.assert_called_once()

    def test_database_published_only_after_success(self):
        import sqlite3
        from src.engine.temporal_store import TemporalStore
        for fail in (True, False):
            with self.subTest(fail=fail):
                root = self.root / str(fail)
                (root / "models").mkdir(parents=True)
                (root / "temporal").mkdir()
                checkpoint = b"checkpoint"
                model_hash = hashlib.sha256(checkpoint).hexdigest()
                (root / "models" / f"{model_hash}.pt").write_bytes(checkpoint)
                database = root / "temporal/surveys.sqlite"
                TemporalStore(database).close()
                original = database.read_bytes()
                video = SimpleNamespace(id="survey", route_id="road", calibration=SimpleNamespace(road_roi=(0, 0, 10, 10)))
                def write(video, output, *, sample_fps, temporal):
                    temporal.store.db.execute('CREATE TABLE publication_test (id INTEGER)')
                    temporal.store.db.commit()
                    if fail:
                        raise RuntimeError("inference failed")
                    output.mkdir()
                    manifest = output / "manifest.json"
                    manifest.write_text('{"damage_count": 0}')
                    return manifest
                def rooted(path):
                    path = Path(path)
                    return root / str(path).lstrip('/') if path.is_absolute() and str(path).split('/')[1] in ('models', 'results', 'recordings', 'temporal') else path
                worker = SimpleNamespace(gpu_name="test", features=SimpleNamespace(extract=Mock(), match=Mock()))
                with patch.object(modal_app, "Path", side_effect=rooted), \
                     patch.object(modal_app, "models"), patch.object(modal_app, "results"), \
                     patch.object(modal_app, "recordings"), patch.object(modal_app, "temporal_data") as volume, \
                     patch("src.engine.recording_io.load_video", return_value=video), \
                     patch("src.engine.video_pipeline.VideoPipeline") as pipeline:
                    (root / "results").mkdir()
                    pipeline.return_value.write.side_effect = write
                    run = modal_app.SurveyWorker._get_user_cls().analyze._get_raw_f()
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, "inference failed"):
                            run(worker, self.job_id, model_hash)
                        self.assertEqual(database.read_bytes(), original)
                        volume.commit.assert_not_called()
                    else:
                        self.assertEqual(run(worker, self.job_id, model_hash)["status"], "complete")
                        volume.commit.assert_called_once()
                        with closing(sqlite3.connect(database)) as db:
                            self.assertIsNotNone(db.execute("SELECT name FROM sqlite_master WHERE name='publication_test'").fetchone())

    def test_invalid_options_do_not_contact_modal(self):
        with patch.object(modal_app, "models") as models:
            for options in ({"sample_fps": 0}, {"confidence": 2}, {"motion_scale_source": "invalid"}):
                with self.subTest(options=options), self.assertRaises(ValueError):
                    modal_app.main(recording="missing", **options)
            models.batch_upload.assert_not_called()


if __name__ == "__main__":
    unittest.main()
