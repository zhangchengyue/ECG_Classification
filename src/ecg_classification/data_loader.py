"""data_loader

Module for downloading, converting, and loading data from Icentia11k.

For dataset details refer to:
https://physionet.org/content/icentia11k-continuous-ecg/1.0/
"""

from dataclasses import dataclass
import logging
from pathlib import Path
import shutil
import time
from typing import Optional, Self
import tarfile
import tempfile
import urllib.error
import urllib.request

import pandas as pd
import numpy as np
import numpy.typing as npt
import wfdb

from ecg_classification.utils import cartesian

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(name)s:%(funcName)s:%(lineno)s %(levelname)s - %(message)s")

class MissingECGSegmentError(Exception):
    """Error for when a patient segment is missing from the dataset"""
    pass

class DownloadManager:
    """Downloads data from Icentia11k dataset"""

    DATASET_URL = "https://physionet.org/files/icentia11k-continuous-ecg/1.0"
    ECG_FILE_EXTENSIONS = ("atr", "hea", "dat")

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def fetch_patient_segment(self, patient_id: int, segment_id: int) -> None:
        """Fetches patient data from Icentia11k dataset.
        
        The data is hosted at PhysioNet, here: https://physionet.org/files/icentia11k-continuous-ecg/1.0
        Refer to Tan et al. https://arxiv.org/pdf/1910.09570 for more information.
        If the patient segment doesn't exist, then this doesn't fetch anything and raises an error.

        Args:
            patient_id: id number for patient
            segment_id: id number for patient's segment

        Raises:
            MissingSegmentError - raised if the requested patient segment does not exist in the dataset or is incomplete,
                i.e. does not contain all three ECG files (.atr, .hea, and .dat)
        """

        if self.is_patient_segment_file_exists(patient_id, segment_id):
            return

        with tempfile.TemporaryDirectory() as tempdir:
            base_dir = Path(tempdir, self.patient_name(patient_id))
            base_dir.mkdir(parents=True, exist_ok=True)

            if self.patient_archive_path(patient_id).exists():
                with tarfile.open(self.patient_archive_path(patient_id), "r:gz") as tar:
                    tar.extractall(path=tempdir)

            # Download segment files
            for file in self.patient_segment_files(patient_id, segment_id):
                try:
                    url = f"{self.patient_url(patient_id)}/{file.name}"
                    log.info(f"Downloading {url}")
                    with urllib.request.urlopen(url) as response:
                        contents = response.read()
                        (base_dir/file.name).write_bytes(contents)
                    # A cooldown to reduce load on the web server
                    time.sleep(3)
                except urllib.error.URLError as e:
                    raise MissingECGSegmentError(f"Could not download {self.patient_segment_name(patient_id, segment_id)}")

            shutil.make_archive(
                self.output_dir/self.patient_name(patient_id),
                "gztar",
                root_dir=tempdir,
                base_dir=base_dir.name,
            )

    def is_patient_segment_file_exists(self, patient_id: int, segment_id: int) -> bool:
        """Checks if patient segment has already been downloaded"""
        if not self.patient_archive_path(patient_id).exists():
            return False
        
        with tarfile.open(self.patient_archive_path(patient_id), "r:gz") as tar:
            for file in self.patient_segment_files(patient_id, segment_id):
                fname = f"{file.parent.name}/{file.name}"
                if fname in tar.getnames():
                    return True
        return False
    
    def patient_name(self, patient_id: int) -> str:
        return f"p{patient_id:05d}"
    
    def patient_url(self, patient_id: int) -> str:
        patient_dir = self.patient_name(patient_id)
        return "/".join([self.DATASET_URL, patient_dir[:3], patient_dir])
    
    def patient_archive_path(self, patient_id: int) -> Path:
        return self.output_dir/f"{self.patient_name(patient_id)}.tar.gz"
    
    def patient_segment_name(self, patient_id: int, segment_id: int) -> str:
        return f"{self.patient_name(patient_id)}_s{segment_id:02d}"
    
    def patient_segment_path(self, patient_id: int, segment_id: int) -> Path:
        return self.output_dir/Path(self.patient_name(patient_id), self.patient_segment_name(patient_id, segment_id))
    
    def patient_segment_files(self, patient_id: int, segment_id: int) -> list[Path]:
        seg_dir = self.patient_segment_path(patient_id, segment_id)
        return [Path(f"{seg_dir}.{ext}") for ext in self.ECG_FILE_EXTENSIONS]
    
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

    def reclassify_beats_in_frame(self, frame_labels: npt.NDArray) -> int:
        """Returns beat classification for a frame.
        
        A given frame contains either all Normal beats, or at least one Abnormal beat.
        The two classes are:
        0 - All normal beats
        1 - At least one abnormal beat
        """
        encoding = self.encode_presence_absence(frame_labels, self.beat_categories)
        # Remember, first element in categories is normal, rest are subtypes of abnormal beats
        has_abnormal_beats = np.bitwise_xor.reduce(encoding[1:])
        return has_abnormal_beats
        
    def reclassify_rhythm_in_frame(self, frame_labels: npt.NDArray) -> int:
        """Returns rhythm classification for a frame.
        
        A given frame contains either all Normal beats, or at least one Abnormal beat.
        The two classes are:
        0 - All normal rhythm
        1 - At least one abnormal rhythm
        """
        encoding = self.encode_presence_absence(frame_labels, self.rhythm_categories)
        # Remember, first element in categories is normal, rest are subtypes of abnormal rhythms
        has_abnormal_rhythm = np.bitwise_xor.reduce(encoding[1:])
        return has_abnormal_rhythm

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
    def new_empty(cls: Self) -> 'ECGData':
        """Creates empty ECGData. Instead of initializing `None` for a variable
        intended to potentially hold ECGData, you should use this method to preserve the type."""
        return cls(frames=np.array([]), frame_number=np.array([]),
            beat_classes=np.array([]), rhythm_classes=np.array([]),
            patient_ids=np.array([]), segment_ids=np.array([]))

    @classmethod
    def from_npz(cls: Self, file: Path) -> 'ECGData':
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
        if self.is_empty() and not other.is_empty():
            return other
        elif not self.is_empty() and other.is_empty():
            return self
        elif self.is_empty() and other.is_empty():
            # Do I need to create a new empty object? Could I just return self? Not really important tho
            return ECGData.new_empty()
        return ECGData(
            frames=np.vstack([self.frames, other.frames]),
            frame_number=np.hstack([self.frame_number, other.frame_number]),
            beat_classes=np.hstack([self.beat_classes, other.beat_classes]),
            rhythm_classes=np.hstack([self.rhythm_classes, other.rhythm_classes]),
            patient_ids=np.hstack([self.patient_ids, other.patient_ids]),
            segment_ids=np.hstack([self.segment_ids, other.segment_ids]),
        )
    
    def _get_arrays(self) -> list[npt.NDArray]:
        return [self.frames, self.frame_number, self.beat_classes, self.rhythm_classes, self.patient_ids, self.segment_ids]
    
    def is_empty(self) -> bool:
        return all(p.size == 0 for p in self._get_arrays())
    
    def to_df(self) -> pd.DataFrame:
        df = pd.DataFrame({
            "patient_id": self.patient_ids,
            "segment_id": self.segment_ids,
            "frame_number": self.frame_number,
            "beat_class": self.beat_classes,
            "rhythm_class": self.rhythm_classes,
        })
        return df
    
    def __eq__(self, other: Self):
        return all([np.all(a == b) for a, b, in zip(self._get_arrays(), other._get_arrays())])

class Icentia11k:
    """Interface to work with the Icentia11k dataset"""

    def __init__(self, dir: Path, frame_length: int):
        self.dir = dir
        self.download_manager = DownloadManager(output_dir=self.dir)
        self.ecg_encoder = ECGLabelEncoder()
        # Tan et al. 2019 used a frame length of 2^11 + 1 = 2049 (https://arxiv.org/pdf/1910.09570)
        self.frame_length = frame_length
        # TODO: add functionality for sample_rate
        self.sample_rate = None

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

        if not ecg_data.is_empty():
            ecg_data.to_npz(file)
            ecg_data.to_df().to_parquet(self.dir/Path("summary.parquet.gzip"), compression="gzip")
        else:
            log.info("ECG data was empty")

    def get_aggregate_frames_and_labels(self, patient_segments: npt.NDArray) -> ECGData:
        """Aggregate frames and labels for multiple patients and segments"""
        ecg_data: ECGData = ECGData.new_empty()
        for patient_id, segment_id in patient_segments:
            try:
                self.check_valid_patient_segment_id(patient_id, segment_id)
                if not self.download_manager.is_patient_segment_file_exists(patient_id, segment_id):
                    self.download_manager.fetch_patient_segment(patient_id, segment_id)
                d = self.get_labelled_segment_frames(patient_id, segment_id)
                ecg_data = ecg_data.combine(d)
            except MissingECGSegmentError as e:
                log.error(e)
                continue
        return ecg_data
    
    def print_summary(self) -> None:
        df = pd.read_parquet(self.dir/"summary.parquet.gzip")
        print("\nDataset Summary\n-------")
        # Patients
        print(len(df), "frames across", df["patient_id"].nunique(), "patients")
        # Segments
        print(round(df.groupby("patient_id")["segment_id"].nunique().mean(), 4), "segments per patient\n")
        # Class distribution
        print("Class distribution (%)\n--------")
        cls_distribution = (df[["beat_class", "rhythm_class"]].sum() / len(df)).round(4) * 100.0
        cls_distribution.rename(index={"beat_class": "abnormal beats", "rhythm_class": "abnormal rhythm"}, inplace=True)
        print(cls_distribution)
    
    def get_labelled_segment_frames(self, patient_id: int, segment_id: int) -> ECGData:
        """Returns frames of ECG signal, and the corresponding beat and rhythm labels per frame
        
        Example, frame 1 contains only Normal beats, with Normal Sinusal Rhythms
        Example, frame 2 contains a Premature Atrial Contraction, with Atrial Fibrillation rhythm
        """

        self.check_valid_patient_segment_id(patient_id, segment_id)

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

    def check_valid_patient_segment_id(self, patient_id, segment_id: int) -> None:
        """Checks if given patient and segment IDs are valid.
        
        Patient IDs should be in [0, 10_999],
            which corresponds to the fully-labelled ECG recordings from Tan et al. 2019.
        Segment IDs should be in [0, 49] because each patient has only 50 ~70min segments of ECG recordings.

        Refer to Tan et al. 2019 (https://www.cinc.org/2021/Program/accepted/229_Preprint.pdf) for details.
        """
        if not (0 <= patient_id <= 10_999 and 0 <= segment_id <= 49):
            raise ValueError(f"Expected valid (patient_id, segment_id), got {(patient_id, segment_id)}")

    def get_recording(self, patient_id: int, segment_id: int, start: int = 0, length: Optional[int] = None
        ) -> tuple[wfdb.Record | wfdb.MultiRecord, wfdb.Annotation]:
        """Returns the recording and annotation data in `segment_dir`.
        
        Args:
            start - the starting time/sample. Ignored if length is `None`.
            length - the length of the segment to visualize. If `None`, it is set to full segment length
        """
        if length and length <= 0:
            raise ValueError("Expected length greater than 0")
        if start < 0:
            raise ValueError("Expected start greater than or equal to 0")
        
        self.check_valid_patient_segment_id(patient_id, segment_id)

        with tempfile.TemporaryDirectory() as tempdir:

            with tarfile.open(self.download_manager.patient_archive_path(patient_id), "r:gz") as tar:
                tar.extractall(path=tempdir)
                segment_dir = str(Path(
                    tempdir, self.download_manager.patient_name(patient_id),
                    self.download_manager.patient_segment_name(patient_id, segment_id)))

                rec: wfdb.Record | wfdb.MultiRecord
                if not length:
                    rec = wfdb.rdrecord(segment_dir)
                    ann = wfdb.rdann(segment_dir, "atr", sampfrom=start, shift_samps=True)
                else:
                    rec = wfdb.rdrecord(segment_dir, sampfrom=start, sampto=start+length)
                    ann = wfdb.rdann(segment_dir, "atr", sampfrom=start, sampto=start+length, shift_samps=True)

        return rec, ann

if __name__ == "__main__":
    rng = np.random.default_rng(seed=2025)

    # TODO: Add a sample rate parameter!
    dataset = Icentia11k(Path("./data/icentia11k"), frame_length=800)
    patient_segments = np.array([
        [9000, 0], # Example normal beat & rhythm
        [900, 0], # Example segment with abnormal beat 
        [1100, 0]]) # Example segment with abnormal rhythm
    dataset.create(patient_segments)
    dataset.print_summary()
