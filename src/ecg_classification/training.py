"""training

Module for training the ECG classification model

Refer to https://github.com/shawntan/icentia-ecg/blob/master/eval.py
"""

from dataclasses import dataclass
import math

import numpy as np
import numpy.typing as npt
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils import resample

@dataclass
class DataSplit:
    name: str
    X: npt.NDArray
    y_rhythm: npt.NDArray
    y_beat: npt.NDArray
    patient: npt.NDArray

def make_ecg_data_train_test_split(data: dict[str, npt.NDArray], test_size: float, random_state: int) -> tuple[DataSplit, DataSplit]:
    """Makes a train/test split for ECG data, but stratified by abnormal classes and grouped by patient.
    
    To prevent data leakage, we must ensure data from the same patient is not present in both train and test.
    Thus, we group all ECG frames belonging to a patient together in the same split.
    
    To ensure the presence of normal and abnormal classes in the training set,
        we stratify by class. Specifically, we stratify by the rhythm class because it is rare,
        then select the corresponding beat labels corresponding to the same input signal.
    """
    # Scikit-Learn has no StratifyGroupShuffleSplit class,
    #   however we can approximate the desired behaviour using a similar KFold class
    cv = StratifiedGroupKFold(n_splits=int(1/test_size), shuffle=True,
        random_state=random_state)

    # We stratify split by rhythm to ensure there is an example of abnormal rhythm in the training set.
    #   Abnormal rhythms are more rare than abnormal beats, so we prioritize stratifying by rhythm.
    # We choose not to also stratify by beats because any beat examples we choose should correspond
    #   to input ECG frames that have already been selected for the training set when stratifying by rhythm.
    train_idxs, test_idxs = next(cv.split(data["signal"], data["rhythm"], groups=data["patient"]))

    return (DataSplit("train", data["signal"][train_idxs], data["rhythm"][train_idxs], data["beat"][train_idxs], data["patient"][train_idxs]),
            DataSplit("test", data["signal"][test_idxs], data["rhythm"][test_idxs], data["beat"][test_idxs], data["patient"][test_idxs]))

def print_split_summary(split: DataSplit) -> None:
    """Prints the distribution of ECG classes (abnormal/normal beat/rhythm) in the given split"""
    print(f"{split.name} split", split.X.shape)
    beat, rhythm = split.y_beat, split.y_rhythm
    print(f"{beat.sum()}/{len(beat)} ({beat.sum()/len(beat):.4f}) abnormal beats")
    print(f"{rhythm.sum()}/{len(rhythm)} ({rhythm.sum()/len(rhythm):.4f}) abnormal rhythms")
    print(f"Patients (n={len(np.unique(split.patient))}):", np.unique(split.patient))

def randomly_oversample_minority_class(train_split: DataSplit, ratio: float, random_state: int) -> DataSplit:
    """Oversamples minority class to balance training set
    Note! Do not apply to test set, otherwise you risk overinflating performance.
    Already, random sampling risks overfitting, and isn't ideal, but might be better than nothing.
    
    Args:
        split: the split of data
        ratio: ratio of minority:majority class
    """
    num_rhythm = train_split.y_rhythm.sum()
    num_normal = len(train_split.y_rhythm) - num_rhythm
    n_samples = num_rhythm * math.ceil(ratio / (num_rhythm/num_normal))

    idx = np.argwhere(train_split.y_rhythm == 1)[:, 0]
    X, y_rhythm, y_beat, patient = resample(train_split.X[idx], train_split.y_rhythm[idx], train_split.y_beat[idx], train_split.patient[idx], n_samples=n_samples, random_state=random_state)
    X = np.concatenate([X, train_split.X], axis=0)
    y_rhythm = np.concatenate([y_rhythm, train_split.y_rhythm])
    y_beat = np.concatenate([y_beat, train_split.y_beat])
    patient = np.concatenate([patient, train_split.patient])
    return DataSplit("train", X, y_rhythm, y_beat, patient)