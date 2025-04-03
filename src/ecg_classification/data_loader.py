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
        """Downloads data of patient `patient_id`.
        Will only fetch the segments specified by `segments`, OR will randomly fetch `n_rand_segments` segments.
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
    
class Icentia11k:
    """Manages retrieval of items from the Icentia11k dataset"""

    # TODO: Figure out how to obtain balanced split of the data

    # TODO: Impl method to count labels in currently downloaded dataset, like https://github.com/shawntan/icentia-ecg/blob/master/physionet/count_everything.py

    label_mapping = {"btype": {0: ('Q', ''),       # Undefined: Unclassifiable beat
                           1: ('N', ''),       # Normal: Normal beat
                           2: ('S', ''),       # ESSV (PAC): Premature or ectopic supraventricular beat
                           3: ('a', ''),       # Aberrated: Aberrated atrial premature beat
                           4: ('V', '')},      # ESV (PVC): Premature ventricular contraction
                 "rtype": {0: ('', ''),        # Null/Undefined
                           1: ('', ''),        # End
                           2: ('', ''),        # Noise
                           3: ('+', "(N"),     # NSR (normal sinusal rhythm): Normal sinusal rhythm
                           4: ('+', "(AFIB"),  # AFib: Atrial fibrillation
                           5: ('+', "(AFL"),   # AFlutter: Atrial flutter
                           6: (None, None)}}   # Used to split a rhythm when a beat annotation is not
                                               # linked to a rhythm type

    beat_mapping = {'N': 0, 'S': 1, 'V': 2}
    rhythm_mapping = {'(N': 0, '(AFIB': 1, '(AFL': 2}

    def __init__(self, dir: Path, frame_length: int):
        self.dir = dir
        self.download_manager = DownloadManager(output_dir=self.dir)
        # Tan et al. 2019 used a frame length of 2^11 + 1 = 2049 (https://arxiv.org/pdf/1910.09570)
        self.frame_length = frame_length

    def download(self, patient_id: int, segments: list[int]) -> None:
        """Download patient files"""
        self.download_manager.fetch_patient_data(patient_id, segments)

    def _get_path(self, patient_id: int, segment: int) -> Path:
        patient_folder = f"p{patient_id:05d}"
        path = self.dir/Path(patient_folder, f"{patient_folder}_s{segment:02d}")
        return path
    
    def get_recording(self,
            patient_id: int, segment: int,
            start: int = 0, length: Optional[int] = None
        ) -> tuple[wfdb.Record | wfdb.MultiRecord, wfdb.Annotation]:
        """Returns the recording and annotation data in `segment_dir`.
        
        :start: - the starting time/sample. Ignored if length is `None`.
        :length: - the length of the segment to visualize. If `None`, it is set to full segment length
        :download: - download if missing
        """
        if length and length <= 0:
            raise ValueError("Expected length greater than 0")
        if start < 0:
            raise ValueError("Expected start greater than or equal to 0")

        if not self._get_path(patient_id, segment).exists():
            self.download(patient_id, [segment])
        segment_dir = str(self._get_path(patient_id, segment))

        rec: wfdb.Record | wfdb.MultiRecord
        if not length:
            rec = wfdb.rdrecord(segment_dir)
            ann = wfdb.rdann(segment_dir, "atr", sampfrom=start, shift_samps=True)
        else:
            rec = wfdb.rdrecord(segment_dir, sampfrom=start, sampto=start+length)
            ann = wfdb.rdann(segment_dir, "atr", sampfrom=start, sampto=start+length, shift_samps=True)
        return rec, ann
    
    def reclassify_beat_for_frame(self, beat_labels_in_frame: npt.NDArray|list[str]) -> npt.NDArray:
        """Relabels the class for a given set of beat labels in a single frame
        
        A single frame can contain multiple labelled beats. So, these need to be combined in some way such that the frame
        as a whole has only one class for a given classification task
        """

        classes = np.unique(beat_labels_in_frame)
        encoded = np.array([0, 0, 0]) # N, S, V
        for c in classes:
            if c in self.beat_mapping:
                encoded[self.beat_mapping[c]] = 1
        return encoded
    
    def reclassify_rhythm_for_frame(self, rhythm_labels_in_frame: npt.NDArray) -> npt.NDArray:
        """Reclassifies the rhythm for a frame, given a set of rhythm labels"""
        # TODO: Encode the expected classes as multi-label so we can calculate cross-entropy loss
        classes = np.unique(rhythm_labels_in_frame)
        log.debug(classes)
        encoded = np.array([0, 0, 0]) # N, AFIB, AFL
        for c in classes:
            if c in self.rhythm_mapping:
                encoded[self.rhythm_mapping[c]] = 1
        frame_has_abnormal_rhythm = encoded[1] ^ encoded[2]
        encoded = np.array([not int(frame_has_abnormal_rhythm), frame_has_abnormal_rhythm])
        return encoded
    
    def get_frames_and_labels(self, patient_id: int, segment: int, drop_empty: bool = True) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """Returns a matrix where each row is a frame, and a dataframe with the corresponding labels
        
        Example, frame 1 contains only Normal beats, with Normal Sinusal Rhythms
        Example, frame 2 contains a Premature Atrial Contraction, with Atrial Fibrillation rhythm

        :patient_id:
        :segment: 
        :drop_empty: decide whether to drop empty frames
        """

        # TODO: Check if pickle file already created

        # Labels dataframe contains patient id, segment id, frame number, beat type, rhythm type, start idx & end idx of frame relative to recording
        rec, ann = self.get_recording(patient_id, segment)

        # Chunk the signal into frames
        signal_data: npt.NDArray = rec.p_signal
        n_frames = len(signal_data) // self.frame_length
        # For some reason, the frame length from the paper doesn't evenly divide a standard segment length
        #   So I decided to chop off the end of the signal_data
        signal_data = signal_data[:n_frames * self.frame_length]
        signal_data = signal_data.reshape((n_frames, self.frame_length))

        # Chunk the annotations into frames
        beat_classes = []
        rhythm_classes = []

        frame_boundary_idxs = self.frame_length * np.arange(1, n_frames+1)
        label_frame_boundary_idxs = np.searchsorted(ann.sample, frame_boundary_idxs)

        start = 0
        for i in label_frame_boundary_idxs:
            # Beats
            beat_labels: npt.NDArray = ann.symbol[start:i]
            beat_classes.append(self.reclassify_beat_for_frame(beat_labels))

            # TODO: Get rhythm labels
            rhythm_labels: npt.NDArray = ann.aux_note[start:i]
            rhythm_classes.append(self.reclassify_rhythm_for_frame(rhythm_labels))

            start = i
        assert len(beat_classes) == n_frames, "Expected as many beat labels as frames"
        return signal_data, np.array(beat_classes), np.array(rhythm_classes)
    
    def create_supervised_training_data(self, patient_ids: list[int], segments: list[int]) -> None:
        """Creates a set of training data"""
        if not segments:
            raise ValueError("Expected at least one segment")
        frames = []
        rhythm_classes = []
        for patient_id in patient_ids:
            for seg in segments:
                frame, beat, rhythm = self.get_frames_and_labels(patient_id, seg)
                frames.append(frame)
                rhythm_classes.append(rhythm)
        # TODO: Consider creating another array that indicates which segment a frame came from

        frames = np.vstack(frames)
        rhythm_classes = np.vstack(rhythm_classes)
        filepath = self.dir/"icentia11k_npz"
        filepath.mkdir(parents=True, exist_ok=True)

        np.savez(
            filepath/f"train.npz",
            signal=np.expand_dims(frames, axis=-1),
            rhythm=rhythm_classes,
            qa_label=np.zeros((len(frames), 3)),
            patient_id=patient_id,
        )

    def load_all_npz(self, folder: Path) -> dict[str, npt.NDArray]:
        """Loads all npz files in a directory and concatenates them"""
        frames = []
        rhythm_classes = []
        for file in folder.glob("*.npz"):
            contents = np.load(file)
            frames.append(contents["signal"])
            rhythm_classes.append(contents["rhythm"])
            
        frames = np.vstack(frames)
        rhythm_classes = np.vstack(rhythm_classes)
        return {"signal": np.expand_dims(frames, axis=-1), "rhythm": rhythm_classes, "qa_labels": np.zeros((frames.shape[0], 3))}

    def count_classes(self, npz_array_data: Path) -> dict[str, int]:
        """Prints class imbalance for a npz of signals and classes"""
        array_data = np.load(npz_array_data)
        rhythm_labels: npt.NDArray = array_data["rhythm"]
        class_counts = rhythm_labels.sum(axis=0)
        return {"Normal": class_counts[0], "Abnormal": class_counts[1]}
    
if __name__ == "__main__":
    downloader = DownloadManager(output_dir=Path("./data/icentia11k"))
    
    for patient_id in [8, 108, 900, 1008, 1100, 9000]:
        downloader.fetch_patient_data(patient_id, n_rand_segments=3)