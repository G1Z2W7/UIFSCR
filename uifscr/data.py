import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .config import DatasetConfig
from .contracts import HyperspectralScene, PreparedScene
from .splitting import StratifiedTrainTestSplitter


class PaviaUDataModule:
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.splitter = StratifiedTrainTestSplitter(
            training_ratio=config.training_ratio,
            random_seed=config.random_seed,
        )

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
        split = self.splitter.split(valid_labels)

        scaler = StandardScaler().fit(valid_spectra[split.training_indices])
        standardized_spectra = scaler.transform(valid_spectra)
        pca = PCA(n_components=int(self.config.pca_components))
        pca.fit(standardized_spectra[split.training_indices])
        spectral_features = pca.transform(standardized_spectra)

        return PreparedScene(
            spectral_features=spectral_features,
            labels=valid_labels,
            valid_indices=valid_indices,
            training_mask=split.training_mask,
            testing_mask=split.testing_mask,
            height=height,
            width=width,
            classes=np.unique(valid_labels),
            explained_variance_ratio=float(np.sum(pca.explained_variance_ratio_)),
        )

    @staticmethod
    def _resolve_array(container, preferred_keys):
        for key in preferred_keys:
            if key in container:
                return container[key]
        available_keys = [key for key in container if not key.startswith("__")]
        if not available_keys:
            raise KeyError("No data array was found in the MATLAB file")
        return container[available_keys[-1]]
