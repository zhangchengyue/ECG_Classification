from pathlib import Path

import numpy as np
import pytest

from ecg_classification.data_loader import ECGLabelEncoder, Icentia11k


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