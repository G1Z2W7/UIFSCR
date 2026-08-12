from dataclasses import dataclass

import numpy as np
from scipy.linalg import svd
from sklearn.svm import LinearSVC

from .config import UIFSCRConfig
from .contracts import LossComponents, OptimizationResult, SuperpixelGraph


@dataclass
class ADMMState:
    representation: np.ndarray
    sparse_error: np.ndarray
    low_rank_proxy: np.ndarray
    reconstruction_multiplier: np.ndarray
    low_rank_multiplier: np.ndarray


class OVRClassificationObjective:
    @staticmethod
    def extract_parameters(classifier):
        weights = np.asarray(classifier.coef_)
        intercepts = np.asarray(classifier.intercept_)
        if weights.ndim == 1:
            weights = weights[None, :]
        if intercepts.ndim == 0:
            intercepts = intercepts[None]
        return weights, intercepts

    @classmethod
    def gradient_and_loss(cls, features, labels, classifier):
        weights, intercepts = cls.extract_parameters(classifier)
        scores = features @ weights.T + intercepts
        gradient = np.zeros_like(features)
        loss = 0.0

        for class_index, class_label in enumerate(classifier.classes_):
            binary_targets = np.where(labels == class_label, 1.0, -1.0)
            margins = 1.0 - binary_targets * scores[:, class_index]
            active = margins > 0
            if not np.any(active):
                continue
            active_margins = margins[active]
            active_targets = binary_targets[active]
            loss += float(np.sum(active_margins ** 2))
            gradient[active] += (
                -2.0 * active_margins * active_targets
            )[:, None] * weights[class_index][None, :]

        return gradient, loss


class UIFSCROptimizer:
    def __init__(self, config: UIFSCRConfig):
        self.config = config
        self.classification_objective = OVRClassificationObjective()

    def fit(self, graph: SuperpixelGraph, iteration_callback=None) -> OptimizationResult:
        self._validate_configuration()
        observations = graph.features
        sample_count, feature_count = observations.shape
        training_indices = np.flatnonzero(graph.training_mask)
        testing_indices = np.flatnonzero(~graph.training_mask)
        if training_indices.size == 0:
            raise ValueError("UIFSCR requires at least one labeled superpixel")

        state = self._initialize_state(observations)
        classifier = self._create_classifier()
        classifier.fit(state.representation[training_indices], graph.labels[training_indices])
        normalization_factor = sample_count * feature_count
        loss_history = []
        component_history = []

        for iteration in range(self.config.maximum_iterations):
            self._update_low_rank_proxy(state)
            self._update_sparse_error(state, observations)
            self._update_representation(
                state,
                observations,
                graph,
                classifier,
                training_indices,
                iteration,
            )
            self._update_multipliers(state, observations)
            classifier.fit(state.representation[training_indices], graph.labels[training_indices])
            components = self._measure_objective(
                state,
                graph,
                classifier,
                training_indices,
                normalization_factor,
            )
            loss_history.append(components.total)
            component_history.append(components)
            if iteration_callback is not None:
                iteration_callback(iteration + 1, self.config.maximum_iterations, components)

        return OptimizationResult(
            representation=state.representation,
            sparse_error=state.sparse_error,
            classifier=classifier,
            loss_history=tuple(loss_history),
            loss_components=tuple(component_history),
            training_indices=training_indices,
            testing_indices=testing_indices,
        )

    def _initialize_state(self, observations):
        return ADMMState(
            representation=observations.copy(),
            sparse_error=np.zeros_like(observations),
            low_rank_proxy=np.zeros_like(observations),
            reconstruction_multiplier=np.zeros_like(observations),
            low_rank_multiplier=np.zeros_like(observations),
        )

    def _create_classifier(self):
        return LinearSVC(
            C=self.config.svm_regularization,
            loss="squared_hinge",
            penalty="l2",
            dual=True,
            fit_intercept=True,
            multi_class="ovr",
            class_weight=None,
            max_iter=self.config.svm_maximum_iterations,
        )

    def _update_low_rank_proxy(self, state):
        penalty = self.config.penalty_parameter
        left_vectors, singular_values, right_vectors = svd(
            state.representation + state.low_rank_multiplier / penalty,
            full_matrices=False,
        )
        thresholded_values = np.maximum(singular_values - 1.0 / penalty, 0.0)
        state.low_rank_proxy = left_vectors @ np.diag(thresholded_values) @ right_vectors

    def _update_sparse_error(self, state, observations):
        penalty = self.config.penalty_parameter
        residual = observations - state.representation + state.reconstruction_multiplier / penalty
        threshold = self.config.sparse_weight / penalty
        state.sparse_error = np.sign(residual) * np.maximum(np.abs(residual) - threshold, 0.0)

    def _update_representation(
        self,
        state,
        observations,
        graph,
        classifier,
        training_indices,
        iteration,
    ):
        penalty = self.config.penalty_parameter
        reconstruction_target = (
            observations
            - state.sparse_error
            + state.reconstruction_multiplier / penalty
        )
        low_rank_target = state.low_rank_proxy - state.low_rank_multiplier / penalty
        step_size = self.config.hinge_step_size / (1.0 + 0.1 * iteration)

        for _ in range(int(self.config.z_gradient_steps)):
            gradient = 2.0 * self.config.graph_weight * (graph.laplacian @ state.representation)
            gradient += 2.0 * self.config.frobenius_weight * state.representation
            gradient += penalty * (state.representation - reconstruction_target)
            gradient += penalty * (state.representation - low_rank_target)
            classification_gradient, _ = self.classification_objective.gradient_and_loss(
                state.representation[training_indices],
                graph.labels[training_indices],
                classifier,
            )
            gradient[training_indices] += self.config.classification_weight * classification_gradient
            state.representation = state.representation - step_size * gradient

    def _update_multipliers(self, state, observations):
        penalty = self.config.penalty_parameter
        state.reconstruction_multiplier += penalty * (
            observations - state.representation - state.sparse_error
        )
        state.low_rank_multiplier += penalty * (
            state.representation - state.low_rank_proxy
        )

    def _measure_objective(
        self,
        state,
        graph,
        classifier,
        training_indices,
        normalization_factor,
    ):
        nuclear = float(np.sum(svd(state.representation, compute_uv=False))) / normalization_factor
        sparse = (
            self.config.sparse_weight
            * float(np.sum(np.abs(state.sparse_error)))
            / normalization_factor
        )
        laplacian_projection = graph.laplacian.dot(state.representation)
        graph_term = (
            self.config.graph_weight
            * float(np.sum(state.representation * laplacian_projection))
            / normalization_factor
        )
        frobenius = (
            self.config.frobenius_weight
            * float(np.sum(state.representation ** 2))
            / normalization_factor
        )
        _, hinge_loss = self.classification_objective.gradient_and_loss(
            state.representation[training_indices],
            graph.labels[training_indices],
            classifier,
        )
        classification = (
            self.config.classification_weight * hinge_loss / normalization_factor
        )
        return LossComponents(
            nuclear=nuclear,
            sparse=sparse,
            graph=graph_term,
            classification=classification,
            frobenius=frobenius,
        )

    def _validate_configuration(self):
        if self.config.svm_kernel != "linear":
            raise ValueError(
                "The current UIFSCR implementation only supports a linear OVR SVM "
                "(svm_kernel='linear')"
            )
        if self.config.penalty_parameter <= 0:
            raise ValueError("The ADMM penalty parameter must be positive")
        if self.config.maximum_iterations <= 0:
            raise ValueError("The maximum number of iterations must be positive")
