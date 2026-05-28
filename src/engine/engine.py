"""
Top-level engine facade for detection and segmentation pipelines.
"""

import yaml
import numpy as np
from pathlib import Path
import sqlite3
from datetime import datetime
import cv2

from .detection import DetectionEngine
from .segmentation import SegmentationEngine
from .models import Damage, ScalarMeasurement, AngleMeasurement, Frame
from .constants import UnitTypes

"""
Database Schema:
- frames (id: str, filepath: Path, timestamp: datetime, coordinates: tuple[int, int], processed: bool, elevation: ScalarMeasurement, azimuth: AngleMeasurement, pitch: AngleMeasurement, roll: AngleMeasurement, yaw: AngleMeasurement, heading: AngleMeasurement)
"""

sqlite_schema = """
CREATE TABLE IF NOT EXISTS job_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filepath TEXT NOT NULL,
    timestamp DATETIME NOT NULL,
    latitude FLOAT NOT NULL,
    longitude FLOAT NOT NULL,
    processed BOOLEAN DEFAULT FALSE,
    elevation FLOAT NOT NULL,
    azimuth FLOAT NOT NULL,
    pitch FLOAT NOT NULL,
    roll FLOAT NOT NULL,
    yaw FLOAT NOT NULL,
    heading FLOAT NOT NULL
);
"""

class FrameListener:
    """
    Listener for frames from a directory and sqlite database that keeps track of already processed frames and metadata.
    """

    def __init__(self, frame_directory: Path, db_path: Path, scalar_unit: UnitTypes, angle_unit: UnitTypes) -> None:
        """
        Initialize the frame listener.

        Args:
            frame_directory: Path to the directory containing the frames.
            db_path: Path to the database file.

        Attributes:
            frame_directory: Path to the directory containing the frames.
            db_path: Path to the database file.
            conn: Connection to the database.
            cursor: Cursor for the database.
        """
        # Initialize units
        self.scalar_unit = scalar_unit
        self.angle_unit = angle_unit
        
        # Initialize database
        self.frame_directory = frame_directory
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()

    def get_next_frame(self) -> Frame | None:
        """Get the next frame from the directory.

        Returns:
            Frame object containing the frame data and metadata.
        """
        self.cursor.execute("SELECT * FROM job_queue WHERE processed = FALSE ORDER BY timestamp ASC LIMIT 1")
        result = self.cursor.fetchone()
        
        # Build Frame object
        if result is None:
            return None
        
        id = result[0]
        filepath = Path(result[1])
        timestamp = datetime.fromisoformat(result[2])
        latitude = result[3]
        longitude = result[4]
        processed = bool(result[5])
        elevation = ScalarMeasurement(value=result[6], unit=self.scalar_unit)
        azimuth = AngleMeasurement(value=result[7], unit=self.angle_unit)
        pitch = AngleMeasurement(value=result[8], unit=self.angle_unit)
        roll = AngleMeasurement(value=result[9], unit=self.angle_unit)
        yaw = AngleMeasurement(value=result[10], unit=self.angle_unit)
        heading = AngleMeasurement(value=result[11], unit=self.angle_unit)
        
        coordinates = (latitude, longitude)
        
        frame = Frame(
            id=id,
            filepath=filepath,
            timestamp=timestamp,
            coordinates=coordinates,
            processed=processed,
            elevation=elevation,
            azimuth=azimuth,
            pitch=pitch,
            roll=roll,
            yaw=yaw,
            heading=heading
        )     
        
        return frame
    
    def update_frame(self, frame: Frame) -> None:
        """Update a frame in the database.
        
        Args:
            frame: Frame object to update.
        """
        self.cursor.execute("UPDATE job_queue SET processed = ? WHERE id = ?", (frame.processed, frame.id))
        self.conn.commit()

class Engine:
    """Coordinate detection and segmentation for end-to-end road-damage inference.

    Loads model paths and thresholds from a YAML config, runs detection on full
    frames, then segments each detected region to produce measured ``Damage``
    records.
    """

    def __init__(self, config_path: Path) -> None:
        """Load config and construct detection and segmentation engines.

        Args:
            config_path: Path to a YAML file with a top-level ``config`` mapping
                (see ``src/engine/config.yaml``). Must include
                ``detection_model_path`` and ``segmentation_model_path``.

        Attributes:
            cfg: Parsed config mapping from the YAML file.
            backend_client: Client for the backend API.
            listener: Listener for frames from a directory and sqlite database that keeps track of already processed frames and metadata.
            detection_engine: YOLO detector for cracks and potholes.
            segmentation_engine: YOLO segmenter for mask and branch analysis.
        """
        # Load config
        with config_path.open() as f:
            raw = yaml.safe_load(f)
        self.cfg = raw["config"]
        
        
        # Initialize sub-engines
        self.detection_engine = DetectionEngine(
            model_path=self.cfg["detection_model_path"]
        )
        self.segmentation_engine = SegmentationEngine(
            model_path=self.cfg["segmentation_model_path"]
        )
        
        # TODO: Initialize backend client
        
        # Initialize listener
        self.listener = FrameListener(
            frame_directory=self.cfg["frame_directory"],
            db_path=self.cfg["db_path"],
            scalar_unit=self.cfg["scalar_unit"],
            angle_unit=self.cfg["angle_unit"]
        )

    # TODO: Make this asynchronous
    def process_frame(self, image: np.ndarray) -> list[Damage]:
        """Detect damage regions, then segment and measure each crop.

        Runs detection on the full frame, crops each bounding box, and passes
        each crop through the segmentation pipeline.

        Args:
            image: Raw BGR input image.

        Returns:
            Measured damage records from all detection crops. Empty if no
            detections are found.
        """
        damages = []

        # Detect damage
        detections = self.detection_engine.process_frame(image)

        # Segment damage
        for detection in detections:
            segments = self.segmentation_engine.process_frame(detection.image)
            damages.extend(segments)

        return damages

    # TODO: Make this asynchronous
    def push_to_backend(self, damage: Damage) -> None:
        """Persist a damage record to the database.

        Args:
            damage: Analyzed damage instance to store.
            db: Database client or connection used by the application.

        Note:
            Not yet implemented.
        """
        pass

    def find_coordinates(self, damage: Damage) -> tuple[int, int]:
        """Return the image coordinates for a damage instance.

        Intended to map a ``Damage`` (e.g. from its mask centroid) back to
        pixel coordinates in the source frame for geolocation or UI overlay.

        Args:
            damage: Damage record whose position should be resolved.

        Returns:
            ``(x, y)`` pixel coordinates in the source image.

        Note:
            Not yet implemented.
        """
        pass

    def calculate_severity(self, damage: Damage) -> float:
        """Compute a severity score from damage measurements and context.

        Args:
            damage: Damage record with dimensions, subtype, and stress range.

        Returns:
            Severity score used to rank or prioritize repairs.

        Note:
            Not yet implemented.
        """
        pass

    def calculate_confidence(self, damage: Damage) -> float:
        pass

    # TODO: Make this asynchronous
    def run(self) -> None:
        """Run the engine in a loop, processing frames and notifying a listener.
        """
        while True:
            frame = self.listener.get_next_frame()
            if frame is None:
                continue
            
            # Read image
            try:
                image = cv2.imread(str(frame.filepath))
                image = np.array(image)
            
            except Exception as e:
                print(f"Error reading image: {e}")
                continue
            
            damages = self.process_frame(image)
            for damage in damages:
                self.push_to_backend(damage)
            
            # Update frame in database
            frame.processed = True
            self.listener.update_frame(frame)
            

if __name__ == "__main__":
    CONFIG_PATH = Path("config.yaml")
    
    engine = Engine(config_path=CONFIG_PATH)
    engine.run()