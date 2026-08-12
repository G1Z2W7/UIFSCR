import os
import warnings

from sklearn.exceptions import ConvergenceWarning

from uifscr import ExperimentConfig, PaviaUExperiment


def main():
    os.environ["LOKY_MAX_CPU_COUNT"] = "4"
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    configuration = ExperimentConfig()
    experiment = PaviaUExperiment(configuration)
    return experiment.run()


if __name__ == "__main__":
    main()
