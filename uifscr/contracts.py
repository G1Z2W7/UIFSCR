from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import spmatrix
from sklearn.svm import LinearSVC


FloatMatrix = NDArray[np.float64]
IntegerArray = NDArray[np.int_]
BooleanArray = NDArray[np.bool_]


@dataclass(frozen=True)
class HyperspectralScene:
    spectral_cube: NDArray[Any]
    ground_truth: NDArray[Any]


@dataclass(frozen=True)
class PreparedScene:
    spectral_features: FloatMatrix
    labels: IntegerArray
    valid_indices: IntegerArray
    training_mask: BooleanArray
    testing_mask: BooleanArray
    height: int
    width: int
    classes: IntegerArray
    explained_variance_ratio: float


@dataclass(frozen=True)
class SuperpixelGraph:
    segmentation: IntegerArray
    features: FloatMatrix
    labels: IntegerArray
    training_mask: BooleanArray
    laplacian: spmatrix
    adjacency: spmatrix
    pixel_memberships: tuple[IntegerArray, ...]


@dataclass(frozen=True)
class LossComponents:
    nuclear: float
    sparse: float
    graph: float
    classification: float
    frobenius: float

    @property
    def total(self) -> float:
        return self.nuclear + self.sparse + self.graph + self.classification + self.frobenius


@dataclass(frozen=True)
class OptimizationResult:
    representation: FloatMatrix
    sparse_error: FloatMatrix
    classifier: LinearSVC
    loss_history: tuple[float, ...]
    loss_components: tuple[LossComponents, ...]
    training_indices: IntegerArray
    testing_indices: IntegerArray


@dataclass(frozen=True)
class EvaluationResult:
    overall_accuracy: float
    average_accuracy: float
    kappa: float
    confusion_matrix: IntegerArray
    classification_report: str
    pixel_predictions: IntegerArray
    full_prediction_map: IntegerArray
