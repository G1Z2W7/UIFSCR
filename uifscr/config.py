from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class DatasetConfig:
    data_path: Path = PROJECT_ROOT / "DataSet" / "PaviaU" / "PaviaU.mat"
    ground_truth_path: Path = PROJECT_ROOT / "DataSet" / "PaviaU" / "PaviaU_gt.mat"
    training_ratio: float = 0.003
    pca_components: int = 20
    random_seed: int | None = None


@dataclass(frozen=True)
class SuperpixelGraphConfig:
    target_segments: int = 700
    compactness: float = 20.0
    smoothing_sigma: float = 1.0
    spectral_neighbors: int = 10
    spatial_neighbors: int = 8


@dataclass(frozen=True)
class UIFSCRConfig:
    sparse_weight: float = 0.1
    graph_weight: float = 1.0
    classification_weight: float = 1.5
    frobenius_weight: float = 0.05
    penalty_parameter: float = 1.0
    svm_regularization: float = 5.0
    maximum_iterations: int = 20
    hinge_step_size: float = 0.02
    z_gradient_steps: int = 3
    svm_kernel: str = "linear"
    svm_maximum_iterations: int = 20000


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    graph: SuperpixelGraphConfig = field(default_factory=SuperpixelGraphConfig)
    optimizer: UIFSCRConfig = field(default_factory=UIFSCRConfig)
