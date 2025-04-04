"""data_loader

Module for downloading, converting, and loading data from Icentia11k.

For dataset details refer to:
https://physionet.org/content/icentia11k-continuous-ecg/1.0/
"""

import logging
from pathlib import Path
import random
import time
from typing import Optional
import urllib.request

import pandas as pd
import numpy as np
import numpy.typing as npt
import wfdb

log = logging.getLogger()
logging.basicConfig(level=logging.DEBUG, format="%(name)s %(levelname)s - %(message)s")

random.seed(1453)

class DownloadManager:
    """Manages downloads for dataset"""

    DATASET_URL = "https://physionet.org/files/icentia11k-continuous-ecg/1.0"
    ECG_FILE_EXTENSIONS = ("atr", "hea", "dat")
    TOTAL_SEGMENTS = 50

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def fetch_patient_data(self, patient_id: int, segments: Optional[list[int]] = None, n_rand_segments: Optional[int] = None) -> None:
        """Fetches patient data from Icentia11k dataset
        
        The data is hosted at PhysioNet (https://physionet.org/files/icentia11k-continuous-ecg/1.0)
        Refer to Tan et al. https://arxiv.org/pdf/1910.09570 for more information.
        """

        patient_folder = f"p{patient_id:05d}"
        patient_folder_url = self._get_patient_folder_url(patient_folder)

        write_path = self.output_dir/patient_folder
        write_path.mkdir(parents=True, exist_ok=True)

        # Define segments to fetch
        if n_rand_segments:
            segments = [random.randint(0, self.TOTAL_SEGMENTS - 1) for _ in range(n_rand_segments)]
        assert segments, "Segments cannot be None. Either set `segments` or `n_rand_segments`"

        # Define file urls to fetch
        files_to_download: list[Path] = []
        for seg in segments:
            for ext in self.ECG_FILE_EXTENSIONS:
                files_to_download.append(Path(f"{patient_folder}_s{seg:02d}.{ext}"))

        # Download files
        for file in files_to_download:
            if (write_path/file).exists():
                log.info(f"{file} already fetched")
                continue
            self.download_file(
                f"{patient_folder_url}/{file}",
                write_path/file
            )

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
        print(encoding)
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


class Icentia11k:
    """Interface to work with the Icentia11k dataset"""

    def __init__(self, dir: Path, frame_length: int):
        self.dir = dir
        self.download_manager = DownloadManager(output_dir=self.dir)
        self.ecg_encoder = ECGLabelEncoder()
        # Tan et al. 2019 used a frame length of 2^11 + 1 = 2049 (https://arxiv.org/pdf/1910.09570)
        self.frame_length = frame_length

    def download(self, patient_id: int, segments: list[int]) -> None:
        """Downloads patient ECG recordings from Icentia11k dataset"""
        self.download_manager.fetch_patient_data(patient_id, segments)

    def get_patient_folder_path(self, patient_id: int, segment: int) -> Path:
        """Gets local path to patient ECG recording
        
        Naming convention for folders is the same as in PhysioNet
        """
        patient_folder = f"p{patient_id:05d}"
        path = self.dir/Path(patient_folder, f"{patient_folder}_s{segment:02d}")
        return path
    
    def get_recording(self,
            patient_id: int, segment: int,
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

        segment_dir = self.get_patient_folder_path(patient_id, segment)
        if not segment_dir.exists() and download_if_missing:
            self.download_manager.fetch_patient_data(patient_id, segments=[segment])
        segment_dir = str(segment_dir)

        rec: wfdb.Record | wfdb.MultiRecord
        if not length:
            rec = wfdb.rdrecord(segment_dir)
            ann = wfdb.rdann(segment_dir, "atr", sampfrom=start, shift_samps=True)
        else:
            rec = wfdb.rdrecord(segment_dir, sampfrom=start, sampto=start+length)
            ann = wfdb.rdann(segment_dir, "atr", sampfrom=start, sampto=start+length, shift_samps=True)
        return rec, ann
        
    def get_frames_and_labels(self, patient_id: int, segment: int) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """Returns frames of ECG signal, and the corresponding beat and rhythm labels per frame
        
        Example, frame 1 contains only Normal beats, with Normal Sinusal Rhythms
        Example, frame 2 contains a Premature Atrial Contraction, with Atrial Fibrillation rhythm
        """

        rec, ann = self.get_recording(patient_id, segment)

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
        return signal_data, np.array(beat_classes, dtype=np.int8), np.array(rhythm_classes, dtype=np.int8)
    
    # TODO: Figure out how to obtain balanced split of the data
    def create_supervised_training_data(self, patient_ids: list[int], segments: list[int]) -> None:
        """Creates a set of training data"""
        if not segments:
            raise ValueError("Expected at least one segment")
        
        frames = []
        beat_classes = []
        rhythm_classes = []

        for patient_id in patient_ids:
            for seg in segments:
                frame, beat, rhythm = self.get_frames_and_labels(patient_id, seg)
                frames.append(frame)
                beat_classes.append(beat)
                rhythm_classes.append(rhythm)

        frames = np.vstack(frames)
        beat_classes = np.vstack(beat_classes, dtype=np.int8)
        rhythm_classes = np.vstack(rhythm_classes, dtype=np.int8)

        np.savez(
            self.dir/"train.npz",
            signal=np.expand_dims(frames, axis=-1),
            rhythm=rhythm_classes,
            beat=beat_classes,
        )

    def describe_ecg_npz(self, npz_ecg_data: Path) -> dict[str, int]:
        """Describes class distribution in npz file"""
        ecg_data = np.load(npz_ecg_data)

        description = {}

        description["num_frames"] = ecg_data["signal"].shape[0]
        description["beat_class_counts"] = np.sum(ecg_data["rhythm"], axis=0)
        description["rhythm_class_counts"] = np.sum(ecg_data["rhythm"], axis=0)
        return description
    
if __name__ == "__main__":
    downloader = DownloadManager(output_dir=Path("./data/icentia11k"))
    
    for patient_id in [8, 108, 900, 1008, 1100, 9000]:
        downloader.fetch_patient_data(patient_id, n_rand_segments=3)