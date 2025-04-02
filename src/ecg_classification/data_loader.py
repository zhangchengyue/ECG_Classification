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

    def __init__(self, dir: Path):
        self.dir = dir
        self.download_manager = DownloadManager(output_dir=self.dir)

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
        if length <= 0:
            raise ValueError("Expected length greater than 0")
        if start < 0:
            raise ValueError("Expected start greater than or equal to 0")

        segment_dir = str(self._get_path(patient_id, segment))

        rec: wfdb.Record | wfdb.MultiRecord
        if not length:
            rec = wfdb.rdrecord(segment_dir)
        else:
            rec = wfdb.rdrecord(segment_dir, sampfrom=start, sampto=start+length)
        ann = wfdb.rdann(segment_dir, "atr", sampfrom=start, sampto=start+length, shift_samps=True)
        return rec, ann 

if __name__ == "__main__":
    downloader = DownloadManager(output_dir=Path("./data/icentia11k"))
    
    for patient_id in [8, 108, 900, 1008, 1100, 9000]:
        downloader.fetch_patient_data(patient_id, n_rand_segments=3)