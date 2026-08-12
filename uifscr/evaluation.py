import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
)

from .contracts import EvaluationResult, OptimizationResult, PreparedScene, SuperpixelGraph


class UIFSCREvaluator:
    def evaluate(
        self,
        scene: PreparedScene,
        graph: SuperpixelGraph,
        optimization: OptimizationResult,
    ) -> EvaluationResult:
        superpixel_predictions = optimization.classifier.predict(optimization.representation)
        pixel_predictions = self._project_to_pixels(
            superpixel_predictions,
            graph.pixel_memberships,
            len(scene.valid_indices),
        )
        testing_labels = scene.labels[scene.testing_mask]
        testing_predictions = pixel_predictions[scene.testing_mask]
        matrix = confusion_matrix(testing_labels, testing_predictions, labels=scene.classes)
        class_accuracies = np.diag(matrix) / (np.sum(matrix, axis=1) + 1e-10)
        full_prediction_map = np.zeros(scene.height * scene.width, dtype=int)
        full_prediction_map[scene.valid_indices] = pixel_predictions

        return EvaluationResult(
            overall_accuracy=float(accuracy_score(testing_labels, testing_predictions)),
            average_accuracy=float(np.mean(class_accuracies)),
            kappa=float(cohen_kappa_score(testing_labels, testing_predictions)),
            confusion_matrix=matrix,
            classification_report=classification_report(
                testing_labels,
                testing_predictions,
                labels=scene.classes,
                zero_division=0,
            ),
            pixel_predictions=pixel_predictions,
            full_prediction_map=full_prediction_map.reshape(scene.height, scene.width),
        )

    @staticmethod
    def _project_to_pixels(superpixel_predictions, memberships, pixel_count):
        pixel_predictions = np.zeros(pixel_count, dtype=int)
        for superpixel_id, pixel_indices in enumerate(memberships):
            pixel_predictions[pixel_indices] = superpixel_predictions[superpixel_id]
        return pixel_predictions
