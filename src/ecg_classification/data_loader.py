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
from typing import Optional
import urllib.request

import pandas as pd
import numpy as np
import numpy.typing as npt
import wfdb

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
    
    def get_patient_folder_path(self, patient_id: int, segment: int) -> Path:
        """Gets local path to patient ECG recording
        
        Naming convention for folders is the same as in PhysioNet
        """
        patient_folder = f"p{patient_id:05d}"
        path = self.output_dir/Path(patient_folder, f"{patient_folder}_s{segment:02d}")
        return path

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
    frames: npt.NDArray
    beat_classes: npt.NDArray
    rhythm_classes: npt.NDArray
    patient_ids: npt.NDArray
    segment_ids: npt.NDArray

class TrainTestSplit:
    pass

class Summarizer:
    pass

class Icentia11k:
    """Interface to work with the Icentia11k dataset"""

    # Tan et al. 2019 remove labels patient ids < 9000. They keep labels for 'test set', i.e. patient id >= 9000
    train_patient_ids = (9_000, 9_999)
    test_patient_ids = (10_000, 10_999)
    # TODO: Train/test split might be better as 0.7 train and 0.3 test
    # TODO: Stratify splits by class distribution, not by patient id's

    segment_id_range = (0, 49)

    # TODO: Remove seed parameters, I wanna avoid any hidden random states
    def __init__(self, dir: Path, frame_length: int, seed: int = 2025):
        self.dir = dir
        self.download_manager = DownloadManager(output_dir=self.dir)
        self.ecg_encoder = ECGLabelEncoder()
        # Tan et al. 2019 used a frame length of 2^11 + 1 = 2049 (https://arxiv.org/pdf/1910.09570)
        self.frame_length = frame_length

        self.rng = np.random.default_rng(seed=seed)

    def create(self, patient_ids: list[int], segments: list[int], overwrite: bool = False) -> None:

        if not overwrite:
            # Open existing dataset and append
            pass

        # Get frames and labels
        frames_and_labels = self.get_aggregate_frames_and_labels(patient_ids, segments)

        # Compute class distributions and save to ./summary.txt at the segment level
        summary = self.summarize_dataset(frames_and_labels)

    def is_valid_patient_segment_id(self, patient_id, segment: int) -> bool:
        """Checks if given patient and segment IDs are valid.
        
        Patient IDs should be in [9_000, 10_999],
            which corresponds to the fully-labelled ECG recordings from Tan et al. 2019.
        Segment IDs should be in [0, 49] because each patient has only 50 ~70min segments of ECG recordings.

        Refer to Tan et al. 2019 (https://www.cinc.org/2021/Program/accepted/229_Preprint.pdf) for details.
        """
        return 9_000 <= patient_id <= 10_999 and 0 <= segment <= 49

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

        segment_dir = self.download_manager.get_patient_folder_path(patient_id, segment)
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
            beat_classes=np.array(beat_classes, dtype=np.int8),
            rhythm_classes=np.array(rhythm_classes, dtype=np.int8),
            patient_ids=np.repeat(patient_id, n_frames),
            segment_ids=np.repeat(segment_id, n_frames, np.int8),
        )
    
    def get_aggregate_frames_and_labels(self, patient_ids: list[int], segments: list[int]) -> ECGData:
        """Aggregate frames and labels for multiple patients and segments"""
        if not segments:
            raise ValueError("Expected at least one segment")
        
        frames = []
        beat_classes = []
        rhythm_classes = []
        patients = []
        segments = []

        for patient_id in patient_ids:
            for seg in segments:
                ecg_data = self.get_labelled_segment_frames(patient_id, seg)
                frames.append(ecg_data.frames)
                beat_classes.append(ecg_data.beat_classes)
                rhythm_classes.append(ecg_data.rhythm_classes)
                patients.append(ecg_data.patient_ids)
                segments.append(ecg_data.segment_ids)

        frames = np.vstack(frames)
        beat_classes = np.vstack(beat_classes, dtype=np.int8)
        rhythm_classes = np.vstack(rhythm_classes, dtype=np.int8)
        patients = np.vstack(patients)
        segments = np.vstack(segments)

        return ECGData(frames, beat_classes, rhythm_classes, patients, segments)

    # TODO: Swap with Scikit-Learn's StratifiedKFoldSplit
    def _create_supervised_data_split(self, split_name: str, size: int, patient_id_range: tuple[int, int], segments: list[int]) -> None:
        """Creates a split of ECG data"""
        patient_ids = self.rng.integers(low=patient_id_range[0], high=patient_id_range[1]+1, size=size)
        ecg_data = self.get_aggregate_frames_and_labels(patient_ids, segments)
        np.savez_compressed(
            self.dir/f"{split_name}.npz",
            signal=np.expand_dims(ecg_data.frames, axis=-1),
            rhythm=ecg_data.rhythm_classes,
            beat=ecg_data.beat_classes,
            patient=ecg_data.patient_ids,
        )
    
    # TODO: Create train and test in a stratified manner ...
    # TODO: Change train_size & test_size to accept proportions instead of absolute amounts
    def create_supervised_train_test_data(self, train_size: int, test_size: int, segments: list[int]) -> None:
        """Creates a training and test sets.
        
        To prevent data leakage, it's important that different patients are used for train and test sets.
        
        Args:
            train_size - Number of patients to include in training set
            test_size - Number of patients to include in test set
            segments - id of specific segments to include, regardless of patient id
        """
        self._create_supervised_data_split("train", train_size, self.train_patient_ids, segments)
        self._create_supervised_data_split("test", test_size, self.test_patient_ids, segments)

    def summarize_dataset(self, ecg_data: dict[str, npt.NDArray]) -> pd.DataFrame:
        """Describes class distribution in npz file""" 
        df = pd.DataFrame({
            "patient_id": ecg_data["patient"],
            "segment_id": ecg_data["segment"],
            "beat_normal": ecg_data["beat"][:, 0].T,
            "beat_abnormal": ecg_data["beat"][:, 1].T,
            "rhythm_normal": ecg_data["rhythm"][:, 0].T,
            "rhythm_abnormal": ecg_data["rhythm"][:, 1].T,
        })
        print(df)

        # description["num_frames"] = ecg_data["signal"].shape[0]
        # description["beat_class_counts"] = np.sum(ecg_data["beat"], axis=0)
        # description["beat_class_proportion"] = np.round(description["beat_class_counts"] / description["num_frames"], 2)
        # description["rhythm_class_counts"] = np.sum(ecg_data["rhythm"], axis=0)
        # description["rhythm_class_proportion"] = np.round(description["rhythm_class_counts"] / description["num_frames"], 2)
        # description["patient_ids"] = np.unique(ecg_data["patient"])
        return df
    
    # TODO: Summarize distribution of classes for all labelled patients 9,000 - 10,999
    
if __name__ == "__main__":
    rng = np.random.default_rng(seed=2025)

    dataset = Icentia11k(Path("./data/icentia11k"), frame_length=800)
    
    dataset.create(
        patient_ids=rng.integers(low=9_000, high=10_999+1, size=3),
        segments=rng.integers(low=0, high=49+1, size=1),
        overwrite=False,
    )

    # TODO: Rewrite the data loader so that it only stores:
    #   1. train.npz
    #   2. test.npz
    #   3. metadata.csv
    #   4. summary.txt

    # Each .npz file should contain an array of frames, beat labels, test labels, patient id, segment id

    # Every call to download another patient should NOT overwrite the train.npz/test.npz, but should append to it
    #   (If you wish to cache previous versions of train and test, simply copy and rename the data/ folder and save it somewhere else)


