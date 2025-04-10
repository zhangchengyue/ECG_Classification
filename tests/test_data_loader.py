from pathlib import Path

import numpy as np
import pytest

from ecg_classification.data_loader import ECGLabelEncoder, Icentia11k, DownloadManager, ECGData


class TestDownloadManager:

    def test_patient_name(self):
        downloader = DownloadManager(Path("."))
        assert "p00900" == downloader.patient_name(patient_id=900)
        assert "p10999" == downloader.patient_name(patient_id=10_999)

    def test_patient_url(self):
        downloader = DownloadManager(Path("."))

        assert "https://physionet.org/files/icentia11k-continuous-ecg/1.0/p00/p00900" == downloader.patient_url(patient_id=900)
        assert "https://physionet.org/files/icentia11k-continuous-ecg/1.0/p10/p10999" == downloader.patient_url(patient_id=10_999)

    def test_patient_archive_path(self, tmp_path: Path):
        downloader = DownloadManager(tmp_path)
        assert tmp_path/"p00900.tar.gz" == downloader.patient_archive_path(900)

    def test_patient_segment_name(self):
        downloader = DownloadManager(Path("."))
        assert "p00900_s00" == downloader.patient_segment_name(900, 0)

    def test_patient_segment_path(self, tmp_path: Path):
        downloader = DownloadManager(tmp_path)
        assert tmp_path/"p00900/p00900_s00" == downloader.patient_segment_path(900, 0)

    def test_patient_segment_files(self, tmp_path: Path):
        downloader = DownloadManager(tmp_path)
        expected = [
            tmp_path/"p00900/p00900_s00.atr",
            tmp_path/"p00900/p00900_s00.hea",
            tmp_path/"p00900/p00900_s00.dat",

        ]
        assert expected == downloader.patient_segment_files(900, 0)
    
    def test_is_patient_segment_file_exists(self, tmp_path: Path):
        import tarfile

        downloader = DownloadManager(tmp_path)

        # Create a temporary tar.gz file
        with tarfile.open(tmp_path/"p00900.tar.gz", "w:gz") as tar:
            for file in ["p00900_s00.atr", "p00900_s00.hea", "p00900_s00.dat"]:
                p = Path(tmp_path, "p00900", file)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.touch()
                tar.add(p)

        assert downloader.is_patient_segment_file_exists(900, 0)
        assert not downloader.is_patient_segment_file_exists(900, 4)

    def test_download_files_does_not_write_file_on_url_error(self):
        import urllib
        
        downloader = DownloadManager(Path("."))
        file = downloader.output_dir/Path("out.txt")

        with pytest.raises(urllib.error.URLError):
            downloader.download_file("https://unreachable.gov", file)

        assert not file.exists()

class TestECGLabelEncoder:

    def test_encode_presence_absence(self):
        encoder = ECGLabelEncoder()

        categories = ['A', 'B', 'C']
        cases = [
            (['A', 'A', 'A'], [1, 0, 0]),
            (['C', 'B'], [0, 1, 1]),
            (['A', 'B', 'C', 'C'], [1, 1, 1]),
            (['X', 'Y', 'Z'], [0, 0, 0])
        ]

        for data, expected in cases:
            actual = encoder.encode_presence_absence(data, categories)
            assert np.all(actual == np.array(expected))

    def test_encode_presence_absence_with_empty_data(self):
        encoder = ECGLabelEncoder()
        expected = [0, 0, 0]
        actual = encoder.encode_presence_absence(np.array([]), ['A', 'B', 'C'])
        assert np.all(actual == np.array(expected))

    def test_beat_label_encoding(self):
        encoder = ECGLabelEncoder()

        # Normal
        normal_beat_frame = np.array(['N'] * 10)
        assert np.all(encoder.reclassify_beats_in_frame(normal_beat_frame) == np.array([1, 0]))

        # Abnormal
        abnormal_beat_frame = np.array(['N'] * 10 + ['S'])
        assert np.all(encoder.reclassify_beats_in_frame(abnormal_beat_frame) == np.array([0, 1]))

    def test_rhythm_label_encoding(self):
        encoder = ECGLabelEncoder()

        # Normal
        normal_rhythm_frame = np.array(['(N'] * 10)
        assert np.all(encoder.reclassify_rhythm_in_frame(normal_rhythm_frame) == np.array([1, 0]))

        # Abnormal
        abnormal_rhythm_frame = np.array(['(N'] * 10 + ['(AFIB'])
        assert np.all(encoder.reclassify_rhythm_in_frame(abnormal_rhythm_frame) == np.array([0, 1]))

    def test_is_valid_patient_segment(self):
        dataset = Icentia11k(dir=Path("./data/icentia11k"), frame_length=800)

        assert dataset.is_valid_patient_segment_id(patient_id=9_000, segment=0), "Lower patient & segment bounds"
        assert dataset.is_valid_patient_segment_id(patient_id=10_999, segment=49), "Upper patient & segment bounds"
        assert not dataset.is_valid_patient_segment_id(patient_id=9_000, segment=50), "Invalid segment ID only"
        assert not dataset.is_valid_patient_segment_id(patient_id=11_000, segment=3), "Invalid patient ID only"
        assert not dataset.is_valid_patient_segment_id(patient_id=0, segment=100), "Invalid patient & segment ID"

class TestECGData:

    def test_empty_ecgdata_is_empty(self):
        data: ECGData = ECGData.new_empty()
        assert data.is_empty()

        a = np.array([1,2,3])
        data = ECGData(a, a, a, a, a, a)
        assert not data.is_empty()

    def test_combine_ecg_data_with_empty_data(self):
        data_1: ECGData = ECGData.new_empty()
        a = np.array([1,2,3])
        data_2 = ECGData(a, a, a, a, a, a)

        assert data_1.combine(data_2) == data_2
        assert data_2.combine(data_1) == data_2