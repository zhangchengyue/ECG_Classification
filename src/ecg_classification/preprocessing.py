"""preprocessing

Helpfer functions for preprocessing
"""

class SyntheticNoise:
    """Generates synthetic noise"""

    def gaussian_noise(self):
        raise NotImplementedError()

    def powerline_interference(self):
        raise NotImplementedError()

    def muscle_artifacts(self):
        raise NotImplementedError()

    def baseline_wander(self):
        raise NotImplementedError()