# ECG Classification

For CSC2631 - Mobile & Digital Health. By ChengYue Zhang and Alex Kappen.

## Usage

### Environment Setup

First, clone the repo.

Next, follow the steps below to set up your Python environment

```python
# Create virtual environment (using Python 3.11 specifically)
python3.11 -m venv .venv
# Activate virtual environment (assumes Linux/Mac)
source .venv/bin/activate
# Install dependencies
python3 -m pip install -r requirements.txt
# Install local, editable package
python3 -m pip install -e .
```

### Download and Visualie the Data

Refer to the [eda.ipynb](nbs/eda.ipynb) notebook for a short example of how to download and visualize the data from Icentia11k.

### ECG Classification Model

Refer to the [DeepBeat.ipynb](nbs/DeepBeat.ipynb) notebook for an example of how to train and test the example multi-task learning model.
The model is reproduced from [Torres-Soto & Ashley, 2020](https://www.nature.com/articles/s41746-020-00320-4).

### Multi-Task Learning for Arrythmia Detection and SupraVentricular/Ventricular Ectopic Beat Classification

Refer to the [Icentia_rhythm_SV.ipynb](nbs/Icentia_rhythm_SV.ipynb) notebook for our multi-task learning model.
