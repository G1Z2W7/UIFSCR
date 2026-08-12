import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .config import DatasetConfig
from .contracts import HyperspectralScene, PreparedScene


class PaviaUDataModule:
    def __init__(self, config: DatasetConfig):
        self.config = config

    def load(self) -> HyperspectralScene:
        data_container = sio.loadmat(self.config.data_path)
        label_container = sio.loadmat(self.config.ground_truth_path)
        spectral_cube = self._resolve_array(data_container, ("pavia", "Pavia"))
        ground_truth = self._resolve_array(label_container, ("pavia_gt", "Pavia_gt"))
        return HyperspectralScene(spectral_cube=spectral_cube, ground_truth=ground_truth)

    def prepare(self, scene: HyperspectralScene) -> PreparedScene:
        height, width, bands = scene.spectral_cube.shape
        flattened_spectra = scene.spectral_cube.reshape(-1, bands)
        flattened_labels = scene.ground_truth.reshape(-1)
        valid_indices = np.flatnonzero(flattened_labels > 0)
        valid_spectra = flattened_spectra[valid_indices]
        valid_labels = flattened_labels[valid_indices]

        standardized_spectra = StandardScaler().fit_transform(valid_spectra)
        pca = PCA(n_components=int(self.config.pca_components))
        spectral_features = pca.fit_transform(standardized_spectra)
        training_mask, testing_mask, classes = self._stratified_partition(valid_labels)

        return PreparedScene(
            spectral_features=spectral_features,
            labels=valid_labels,
            valid_indices=valid_indices,
            training_mask=training_mask,
            testing_mask=testing_mask,
            height=height,
            width=width,
            classes=classes,
            explained_variance_ratio=float(np.sum(pca.explained_variance_ratio_)),
        )

    def _stratified_partition(self, labels):
        classes = np.unique(labels)
        training_indices = []
        testing_indices = []
        random_generator = None
        if self.config.random_seed is not None:
            random_generator = np.random.default_rng(self.config.random_seed)

        for class_label in classes:
            class_indices = np.flatnonzero(labels == class_label)
            if random_generator is None:
                np.random.shuffle(class_indices)
            else:
                random_generator.shuffle(class_indices)
            training_count = max(1, int(self.config.training_ratio * len(class_indices)))
            training_indices.extend(class_indices[:training_count])
            testing_indices.extend(class_indices[training_count:])

        training_mask = np.zeros(len(labels), dtype=bool)
        testing_mask = np.zeros(len(labels), dtype=bool)
        training_mask[training_indices] = True
        testing_mask[testing_indices] = True
        return training_mask, testing_mask, classes

    @staticmethod
    def _resolve_array(container, preferred_keys):
        for key in preferred_keys:
            if key in container:
                return container[key]
        available_keys = [key for key in container if not key.startswith("__")]
        if not available_keys:
            raise KeyError("No data array was found in the MATLAB file")
        return container[available_keys[-1]]
