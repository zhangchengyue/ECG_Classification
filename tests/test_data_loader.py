from pathlib import Path

import numpy as np
import pytest

from ecg_classification.data_loader import ECGLabelEncoder, Icentia11k, DownloadManager, ECGData


def test_encode_presence_absence():
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

def test_encode_presence_absence_with_empty_data():
    encoder = ECGLabelEncoder()
    expected = [0, 0, 0]
    actual = encoder.encode_presence_absence(np.array([]), ['A', 'B', 'C'])
    assert np.all(actual == np.array(expected))

def test_beat_label_encoding():
    encoder = ECGLabelEncoder()

    # Normal
    normal_beat_frame = np.array(['N'] * 10)
    assert np.all(encoder.reclassify_beats_in_frame(normal_beat_frame) == np.array([1, 0]))

    # Abnormal
    abnormal_beat_frame = np.array(['N'] * 10 + ['S'])
    assert np.all(encoder.reclassify_beats_in_frame(abnormal_beat_frame) == np.array([0, 1]))

def test_rhythm_label_encoding():
    encoder = ECGLabelEncoder()

    # Normal
    normal_rhythm_frame = np.array(['(N'] * 10)
    assert np.all(encoder.reclassify_rhythm_in_frame(normal_rhythm_frame) == np.array([1, 0]))

    # Abnormal
    abnormal_rhythm_frame = np.array(['(N'] * 10 + ['(AFIB'])
    assert np.all(encoder.reclassify_rhythm_in_frame(abnormal_rhythm_frame) == np.array([0, 1]))

def test_is_valid_patient_segment():
    dataset = Icentia11k(dir=Path("./data/icentia11k"), frame_length=800)

    assert dataset.is_valid_patient_segment_id(patient_id=9_000, segment=0), "Lower patient & segment bounds"
    assert dataset.is_valid_patient_segment_id(patient_id=10_999, segment=49), "Upper patient & segment bounds"
    assert not dataset.is_valid_patient_segment_id(patient_id=9_000, segment=50), "Invalid segment ID only"
    assert not dataset.is_valid_patient_segment_id(patient_id=11_000, segment=3), "Invalid patient ID only"
    assert not dataset.is_valid_patient_segment_id(patient_id=0, segment=100), "Invalid patient & segment ID"

def test_download_files_does_not_write_file_on_url_error():
    import urllib
    
    downloader = DownloadManager(Path("./data/icentia11k"))
    file = downloader.output_dir/Path("out.txt")

    with pytest.raises(urllib.error.URLError):
        downloader.download_file("https://unreachable.gov", file)

    assert not file.exists()

def test_empty_ecgdata_is_empty():
    data: ECGData = ECGData.new_empty()
    assert data.is_empty()

    a = np.array([1,2,3])
    data = ECGData(a, a, a, a, a, a)
    assert not data.is_empty()

def test_combine_ecg_data_with_empty_data():
    data_1: ECGData = ECGData.new_empty()
    a = np.array([1,2,3])
    data_2 = ECGData(a, a, a, a, a, a)

    assert data_1.combine(data_2) == data_2
    assert data_2.combine(data_1) == data_2