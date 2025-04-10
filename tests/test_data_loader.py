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
                p = Path("p00900", file)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.touch()
                tar.add(p)

        assert downloader.is_patient_segment_file_exists(900, 0)
        assert not downloader.is_patient_segment_file_exists(900, 4)

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
        assert encoder.reclassify_beats_in_frame(normal_beat_frame) == 0

        # Abnormal
        abnormal_beat_frame = np.array(['N'] * 10 + ['S'])
        assert encoder.reclassify_beats_in_frame(abnormal_beat_frame) == 1

    def test_rhythm_label_encoding(self):
        encoder = ECGLabelEncoder()

        # Normal
        normal_rhythm_frame = np.array(['(N'] * 10)
        assert encoder.reclassify_rhythm_in_frame(normal_rhythm_frame) == 0

        # Abnormal
        abnormal_rhythm_frame = np.array(['(N'] * 10 + ['(AFIB'])
        assert encoder.reclassify_rhythm_in_frame(abnormal_rhythm_frame) == 1

    def test_check_valid_patient_segment_id(self):
        dataset = Icentia11k(dir=Path("./data/icentia11k"), frame_length=800)

        assert dataset.check_valid_patient_segment_id(patient_id=9_000, segment_id=0) is None
        assert dataset.check_valid_patient_segment_id(patient_id=10_999, segment_id=49) is None

        with pytest.raises(ValueError): 
            # Invalid segment ID only
            dataset.check_valid_patient_segment_id(patient_id=9_000, segment_id=50)
            # Invalid patient ID only
            dataset.check_valid_patient_segment_id(patient_id=11_000, segment_id=3)
            # Invalid patient & segment ID
            dataset.check_valid_patient_segment_id(patient_id=0, segment_id=100)

class TestECGData:

    def test_combine(self):
        a = ECGData(
            frames=np.array([[1, 2, 3], [4, 5, 6]]),
            frame_number=np.array([1, 2]),
            beat_classes=np.array([0, 1]),
            rhythm_classes=np.array([0, 0]),
            patient_ids=np.array([900, 901]),
            segment_ids=np.array([0, 1]),
        )
        b = a.combine(a)
        assert b == ECGData(
            frames=np.array([[1, 2, 3], [4, 5, 6], [1, 2, 3], [4, 5, 6]]),
            frame_number=np.array([1, 2, 1, 2]),
            beat_classes=np.array([0, 1, 0, 1]),
            rhythm_classes=np.array([0, 0, 0, 0]),
            patient_ids=np.array([900, 901, 900, 901]),
            segment_ids=np.array([0, 1, 0, 1]),
        )

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