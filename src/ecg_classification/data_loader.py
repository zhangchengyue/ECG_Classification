"""data_loader

Module for downloading, converting, and loading data from Icentia11k.

For dataset details refer to:
https://physionet.org/content/icentia11k-continuous-ecg/1.0/
"""

from dataclasses import dataclass
import logging
from pathlib import Path
import random
import time
from typing import Optional, Self
import tarfile
import urllib.error
import urllib.request

import pandas as pd
import numpy as np
import numpy.typing as npt
import wfdb

from ecg_classification.utils import cartesian

log = logging.getLogger()
logging.basicConfig(level=logging.DEBUG, format="%(name)s %(levelname)s - %(message)s")


class DownloadManager:
    """Manages downloads for dataset"""

    DATASET_URL = "https://physionet.org/files/icentia11k-continuous-ecg/1.0"
    ECG_FILE_EXTENSIONS = ("atr", "hea", "dat")
    TOTAL_SEGMENTS = 50

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def fetch_patient_segment(self, patient_id: int, segment_id: int) -> None:
        """Fetches patient data from Icentia11k dataset.
        
        The data is hosted at PhysioNet, here: https://physionet.org/files/icentia11k-continuous-ecg/1.0
        Refer to Tan et al. https://arxiv.org/pdf/1910.09570 for more information.
        """

        patient_folder = f"p{patient_id:05d}"
        patient_folder_url = self._get_patient_folder_url(patient_folder)

        # Check if already downloaded
        if (self.output_dir/f"{patient_folder}.tar.gz").exists():
            log.info(f"{patient_folder} already fetched")
            return

        write_path = self.output_dir/patient_folder
        write_path.mkdir(parents=True, exist_ok=True)

        # Define file urls to fetch
        files_to_download: list[Path] = []
        for ext in self.ECG_FILE_EXTENSIONS:
            files_to_download.append(Path(f"{patient_folder}_s{segment_id:02d}.{ext}"))

        # Download files
        for file in files_to_download:
            try:
                self.download_file(
                    f"{patient_folder_url}/{file}",
                    write_path/file
                )
            except urllib.error.URLError as e:
                log.error(f"{e}. Could not download {file}")
                continue

        # Archive & compress
        with tarfile.open(self.get_tar_gz_path(patient_id, segment_id), "w:gz") as tar:
            tar.add(write_path)

        # Remove files
        for file in files_to_download:
            if not (write_path/file).exists():
                continue
            (write_path/file).unlink()
        write_path.rmdir()

    def download_file(self, url: str, out: Path, sleep: int = 3) -> bytes:
        """Downloads a file using HTTPS.
        Sleeps for `sleep` seconds to prevent too many requests in a short time.
        """
        log.info(f"Downloading {url}")
        with urllib.request.urlopen(url) as response:
            contents = response.read()
            out.write_bytes(contents)
        time.sleep(sleep)


    def _get_patient_folder_url(self, patient_folder: str) -> str:
        return "/".join([self.DATASET_URL, patient_folder[:3], patient_folder])
    
    def get_patient_folder_path(self, patient_id: int, segment_id: int) -> Path:
        """Gets local path to patient ECG recording
        
        Naming convention for folders is the same as in PhysioNet
        """
        patient_folder = f"p{patient_id:05d}"
        path = self.output_dir/Path(patient_folder, f"{patient_folder}_s{segment_id:02d}")
        return path
    
    def get_tar_gz_path(self, patient_id: int, segment_id: int) -> Path:
        patient_folder = f"p{patient_id:05d}"
        return self.output_dir/f"{patient_folder}_{segment_id:02d}.tar.gz"

class ECGLabelEncoder:
    """Encodes labels for ECG recordings"""

    beat_categories = [
        "N", # Normal: Normal beat
        "S", # ESSV (PAC): Premature or ectopic supraventricular beat
        "V", # ESV (PVC): Premature ventricular contraction
    ]
    rhythm_categories = [
        "(N", # NSR (normal sinusal rhythm): Normal sinusal rhythm
        "(AFIB", # AFib: Atrial fibrillation
        "(AFL", # AFlutter: Atrial flutter
    ]

    def __init__(self):
        pass

    def encode_presence_absence(self, data: npt.NDArray|list[str], categories: list[str]) -> npt.NDArray:
        categories = {class_label:idx for idx, class_label in enumerate(categories)}
        classes = np.unique(data)
        encoded = np.zeros(len(categories), dtype=np.int8)
        for c in classes:
            if c in categories:
                encoded[categories[c]] = 1
        return encoded

    def reclassify_beats_in_frame(self, frame_labels: npt.NDArray) -> npt.NDArray:
        """Returns beat classification for a frame.
        
        A given frame contains either all Normal beats, or at least one Abnormal beat.
        The two classes are:
        0 - All normal beats
        1 - At least one abnormal beat
        """
        encoding = self.encode_presence_absence(frame_labels, self.beat_categories)
        # Remember, first element in categories is normal, rest are subtypes of abnormal beats
        has_abnormal_beats = np.bitwise_xor.reduce(encoding[1:])
        return np.array([int(not has_abnormal_beats), has_abnormal_beats])
        
    def reclassify_rhythm_in_frame(self, frame_labels: npt.NDArray) -> npt.NDArray:
        """Returns rhythm classification for a frame.
        
        A given frame contains either all Normal beats, or at least one Abnormal beat.
        The two classes are:
        0 - All normal rhythm
        1 - At least one abnormal rhythm
        """
        encoding = self.encode_presence_absence(frame_labels, self.rhythm_categories)
        # Remember, first element in categories is normal, rest are subtypes of abnormal rhythms
        has_abnormal_rhythm = np.bitwise_xor.reduce(encoding[1:])
        return np.array([int(not has_abnormal_rhythm), has_abnormal_rhythm])

@dataclass
class ECGData:
    """Stores ECG recordings"""

    frames: npt.NDArray
    frame_number: npt.NDArray
    beat_classes: npt.NDArray
    rhythm_classes: npt.NDArray
    patient_ids: npt.NDArray
    segment_ids: npt.NDArray

    @classmethod
    def from_npz(cls: Self, file: Path) -> Self:
        """Creates an instance of ECGData from a npz file"""
        d = np.load(file)
        return cls(
            frames=d["signal"], frame_number=d["frame_num"], beat_classes=d["beat"], rhythm_classes=d["rhythm"],
            patient_ids=d["patient"], segment_ids=d["segment"]
        )

    def to_npz(self, file: Path) -> None:
        """Saves ECGData to npz"""
        np.savez_compressed(
            file,
            signal=self.frames,
            frame_num=self.frame_number,
            beat=self.beat_classes,
            rhythm=self.rhythm_classes,
            patient=self.patient_ids,
            segment=self.segment_ids,
        )

    def combine(self, other: Self) -> Self:
        """Combines two instances of ECGData into one"""
        return ECGData(
            frames=np.vstack([self.frames, other.frames]),
            frame_number=np.hstack([self.frame_number, other.frame_number]),
            beat_classes=np.vstack([self.beat_classes, other.beat_classes]),
            rhythm_classes=np.vstack([self.rhythm_classes, other.rhythm_classes]),
            patient_ids=np.hstack([self.patient_ids, other.patient_ids]),
            segment_ids=np.hstack([self.segment_ids, other.segment_ids]),
        )

class Icentia11k:
    """Interface to work with the Icentia11k dataset"""

    def __init__(self, dir: Path, frame_length: int):
        self.dir = dir
        self.download_manager = DownloadManager(output_dir=self.dir)
        self.ecg_encoder = ECGLabelEncoder()
        # Tan et al. 2019 used a frame length of 2^11 + 1 = 2049 (https://arxiv.org/pdf/1910.09570)
        self.frame_length = frame_length

    def create(self, patient_segments: npt.NDArray, overwrite: bool = False) -> None:
        """
        Args:
            patient_segments - a 2D array; each row is a [patient, segment] pair
            overwrite - whether or not to overwrite the local dataset file
        """

        file = self.dir/"data.npz"
        if not overwrite and file.exists():
            ecg_data: ECGData = ECGData.from_npz(file)

            # Filter patients and segments that already exist in the dataset
            stored = set((patient, segment) for patient, segment in cartesian(np.unique(ecg_data.patient_ids), np.unique(ecg_data.segment_ids)))
            query = set((patient, segment) for patient, segment in patient_segments)
            patient_segments = np.array([[patient, segment] for patient, segment in query - stored])

            # If there are patients and segments to add
            if len(patient_segments) >= 1:
                ecg_data = ecg_data.combine(self.get_aggregate_frames_and_labels(patient_segments))
        else:
            ecg_data = self.get_aggregate_frames_and_labels(patient_segments)

        ecg_data.to_npz(file)

        summary = self.summarize_dataset(ecg_data)
        log.info(f"Dataset Summary\n{summary}")
        summary.to_parquet(self.dir/Path("summary.parquet.gzip"), compression="gzip")

    def is_valid_patient_segment_id(self, patient_id, segment: int) -> bool:
        """Checks if given patient and segment IDs are valid.
        
        Patient IDs should be in [9_000, 10_999],
            which corresponds to the fully-labelled ECG recordings from Tan et al. 2019.
        Segment IDs should be in [0, 49] because each patient has only 50 ~70min segments of ECG recordings.

        Refer to Tan et al. 2019 (https://www.cinc.org/2021/Program/accepted/229_Preprint.pdf) for details.
        """
        return 9_000 <= patient_id <= 10_999 and 0 <= segment <= 49

    def get_recording(self,
            patient_id: int, segment_id: int,
            start: int = 0, length: Optional[int] = None,
            download_if_missing: bool = True
        ) -> tuple[wfdb.Record | wfdb.MultiRecord, wfdb.Annotation]:
        """Returns the recording and annotation data in `segment_dir`.
        
        Args:
            start - the starting time/sample. Ignored if length is `None`.
            length - the length of the segment to visualize. If `None`, it is set to full segment length
            download - download if missing
        """
        if length and length <= 0:
            raise ValueError("Expected length greater than 0")
        if start < 0:
            raise ValueError("Expected start greater than or equal to 0")
        
        if not self.is_valid_patient_segment_id(patient_id, segment_id):
            raise ValueError(f"Expected valid (patient_id, segment_id), got {(patient_id, segment_id)}")

        if not self.download_manager.get_tar_gz_path(patient_id, segment_id).exists() and download_if_missing:
            self.download_manager.fetch_patient_segment(patient_id, segment_id)

        segment_dir = str(self.download_manager.get_patient_folder_path(patient_id, segment_id))
        
        with tarfile.open(self.download_manager.get_tar_gz_path(patient_id, segment_id), "r:gz") as tar:
            tar.extractall()

            log.debug(segment_dir)
            rec: wfdb.Record | wfdb.MultiRecord
            if not length:
                rec = wfdb.rdrecord(segment_dir)
                ann = wfdb.rdann(segment_dir, "atr", sampfrom=start, shift_samps=True)
            else:
                rec = wfdb.rdrecord(segment_dir, sampfrom=start, sampto=start+length)
                ann = wfdb.rdann(segment_dir, "atr", sampfrom=start, sampto=start+length, shift_samps=True)

            for tarinfo in tar:
                if tarinfo.isreg():
                    Path(tarinfo.path).unlink()
            self.download_manager.get_patient_folder_path(patient_id, segment_id).parent.rmdir()

        return rec, ann
        
    def get_labelled_segment_frames(self, patient_id: int, segment_id: int) -> ECGData:
        """Returns frames of ECG signal, and the corresponding beat and rhythm labels per frame
        
        Example, frame 1 contains only Normal beats, with Normal Sinusal Rhythms
        Example, frame 2 contains a Premature Atrial Contraction, with Atrial Fibrillation rhythm
        """
        if not self.is_valid_patient_segment_id(patient_id, segment_id):
            raise ValueError(f"Expected valid (patient_id, segment_id), got {(patient_id, segment_id)}")

        rec, ann = self.get_recording(patient_id, segment_id)

        # Chunk the signal into frames
        signal_data: npt.NDArray = rec.p_signal
        n_frames = len(signal_data) // self.frame_length

        # Slice signal if frames don't evenly divide it
        signal_data = signal_data[:n_frames * self.frame_length]
        signal_data = signal_data.reshape((n_frames, self.frame_length))

        # Chunk the annotations by frames
        frame_boundary_idxs = self.frame_length * np.arange(1, n_frames+1)
        label_frame_boundary_idxs = np.searchsorted(ann.sample, frame_boundary_idxs)

        start = 0
        beat_classes = []
        rhythm_classes = []
        for i in label_frame_boundary_idxs:
            # Beat labels
            beats_in_frame: npt.NDArray = ann.symbol[start:i]
            beat_classes.append(self.ecg_encoder.reclassify_beats_in_frame(beats_in_frame))

            # Rhythm labels
            rhythm_in_frame: npt.NDArray = ann.aux_note[start:i]
            rhythm_classes.append(self.ecg_encoder.reclassify_rhythm_in_frame(rhythm_in_frame))

            start = i

        assert len(beat_classes) == n_frames, "Expected as many beat labels as frames"
        assert len(rhythm_classes) == n_frames, "Expected as many rhythm labels as frames"

        return ECGData(
            frames=signal_data,
            frame_number=np.arange(n_frames),
            beat_classes=np.array(beat_classes, dtype=np.int8),
            rhythm_classes=np.array(rhythm_classes, dtype=np.int8),
            patient_ids=np.repeat(patient_id, n_frames),
            segment_ids=np.repeat(segment_id, n_frames).astype(np.int8),
        )
    
    def get_aggregate_frames_and_labels(self, patient_segments: npt.NDArray) -> ECGData:
        """Aggregate frames and labels for multiple patients and segments"""
        # Stack all the ECG data (frames + labels)
        ecg_data = self.get_labelled_segment_frames(*patient_segments[0])
        for patient, segment in patient_segments[1:]:
            ecg_data = ecg_data.combine(self.get_labelled_segment_frames(patient, segment))
        return ecg_data

    def summarize_dataset(self, ecg_data: ECGData) -> pd.DataFrame:
        """Describes class distribution in npz file""" 
        df = pd.DataFrame({
            "patient_id": ecg_data.patient_ids,
            "segment_id": ecg_data.segment_ids,
            "frame_number": ecg_data.frame_number,
            "beat_normal": ecg_data.beat_classes[:, 0].T[0],
            "beat_abnormal": ecg_data.beat_classes[:, 1].T[0],
            "rhythm_normal": ecg_data.rhythm_classes[:, 0].T[0],
            "rhythm_abnormal": ecg_data.rhythm_classes[:, 1].T[0],
        })
        return df
    
if __name__ == "__main__":
    rng = np.random.default_rng(seed=1234)

    dataset = Icentia11k(Path("./data/icentia11k"), frame_length=800)

    patient_ids = rng.integers(low=9_000, high=10_999+1, size=3)
    segment_ids = rng.integers(low=0, high=29+1, size=1)
    patient_segments = cartesian(patient_ids, segment_ids)
    
    dataset.create(patient_segments, overwrite=False)