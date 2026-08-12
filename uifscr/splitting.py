from dataclasses import dataclass

import numpy as np

from .contracts import BooleanArray, IntegerArray


@dataclass(frozen=True)
class TrainTestSplit:
    training_indices: IntegerArray
    testing_indices: IntegerArray
    training_mask: BooleanArray
    testing_mask: BooleanArray


class StratifiedTrainTestSplitter:
    def __init__(self, training_ratio: float, random_seed: int | None = None):
        if not 0.0 < training_ratio < 1.0:
            raise ValueError("training_ratio must be between 0 and 1")
        self.training_ratio = training_ratio
        self.random_seed = random_seed

    def split(self, labels: IntegerArray) -> TrainTestSplit:
        labels = np.asarray(labels)
        training_parts = []
        testing_parts = []
        random_generator = np.random.default_rng(self.random_seed)

        for class_label in np.unique(labels):
            class_indices = np.flatnonzero(labels == class_label)
            random_generator.shuffle(class_indices)
            training_count = max(1, int(self.training_ratio * class_indices.size))
            if class_indices.size > 1:
                training_count = min(training_count, class_indices.size - 1)
            training_parts.append(class_indices[:training_count])
            testing_parts.append(class_indices[training_count:])

        training_indices = np.concatenate(training_parts).astype(int, copy=False)
        testing_indices = np.concatenate(testing_parts).astype(int, copy=False)
        training_mask = np.zeros(labels.size, dtype=bool)
        testing_mask = np.zeros(labels.size, dtype=bool)
        training_mask[training_indices] = True
        testing_mask[testing_indices] = True

        return TrainTestSplit(
            training_indices=training_indices,
            testing_indices=testing_indices,
            training_mask=training_mask,
            testing_mask=testing_mask,
        )
