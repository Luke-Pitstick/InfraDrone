"""Local temporal evidence, stored transactionally without a database service."""

from dataclasses import asdict
from io import BytesIO
import json
from pathlib import Path
import sqlite3

import numpy as np


def pack(**arrays) -> bytes:
    buffer = BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


def unpack(data: bytes) -> dict:
    with np.load(BytesIO(data), allow_pickle=False) as arrays:
        return dict(arrays)


class TemporalStore:
    """One SQLite file contains frame features, original masks, and observations.

    Frame ingestion is atomic and keyed by video ID/frame index, so replaying a
    completed frame cannot duplicate defects. Use a fresh video ID for reanalysis.
    """

    def __init__(self, path: str | Path):
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA foreign_keys = ON')
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS frames (
                video_id TEXT NOT NULL, frame_index INTEGER NOT NULL,
                route_id TEXT NOT NULL, latitude REAL, longitude REAL, accuracy REAL,
                calibration TEXT NOT NULL, features BLOB NOT NULL, video TEXT NOT NULL,
                PRIMARY KEY(video_id, frame_index));
            CREATE INDEX IF NOT EXISTS frames_route ON frames(route_id, latitude);
            CREATE TABLE IF NOT EXISTS defects (id TEXT PRIMARY KEY);
            CREATE TABLE IF NOT EXISTS observations (
                id TEXT PRIMARY KEY, video_id TEXT NOT NULL, frame_index INTEGER NOT NULL,
                defect_id TEXT REFERENCES defects(id), status TEXT NOT NULL,
                score REAL, payload TEXT NOT NULL, masks BLOB NOT NULL,
                FOREIGN KEY(video_id, frame_index) REFERENCES frames(video_id, frame_index));
            CREATE INDEX IF NOT EXISTS observations_frame ON observations(video_id, frame_index);
        ''')

    def close(self):
        self.db.close()

    def observations(self, video_id, frame_index):
        return self.db.execute('SELECT * FROM observations WHERE video_id=? AND frame_index=?',
                               (video_id, frame_index)).fetchall()

    def has_frame(self, video_id, frame_index):
        return self.db.execute('SELECT 1 FROM frames WHERE video_id=? AND frame_index=?',
                               (video_id, frame_index)).fetchone() is not None

    def candidates(self, video, location):
        rows = self.db.execute('SELECT * FROM frames WHERE route_id=?', (video.route_id,)).fetchall()
        candidates = []
        for row in rows:
            # Unknown GPS retains candidates; it must not imply a new road section.
            if location is not None and row['latitude'] is not None:
                lat = np.radians((location.latitude + row['latitude']) / 2)
                distance = 111320 * np.hypot(location.latitude-row['latitude'],
                                            (location.longitude-row['longitude'])*np.cos(lat))
                radius = 2*video.calibration.max_range_m + (location.accuracy_m or 50) + (row['accuracy'] or 50)
                if distance > radius:
                    continue
            candidates.append(row)
        return candidates

    def save(self, video, result, features, associations):
        with self.db:
            location = result.location
            self.db.execute('INSERT INTO frames VALUES (?,?,?,?,?,?,?,?,?)', (
                video.id, result.frame.index, video.route_id,
                location.latitude if location else None, location.longitude if location else None,
                location.accuracy_m if location else None, json.dumps(asdict(video.calibration)),
                pack(points=features.points, descriptors=features.descriptors, image_size=features.image_size), json.dumps(video.to_dict())))
            for damage, association in zip(result.damages, associations):
                if association.defect_id is not None:
                    self.db.execute('INSERT OR IGNORE INTO defects VALUES (?)', (association.defect_id,))
                self.db.execute('INSERT INTO observations VALUES (?,?,?,?,?,?,?,?)', (
                    str(damage.id), video.id, result.frame.index, association.defect_id,
                    association.status, association.score, json.dumps(damage.to_dict(), allow_nan=False),
                    pack(mask=damage.mask, skeleton=damage.skeleton),
                ))
