import numpy as np
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from skimage.segmentation import slic

from .config import SuperpixelGraphConfig
from .contracts import PreparedScene, SuperpixelGraph


class SuperpixelGraphBuilder:
    def __init__(self, config: SuperpixelGraphConfig):
        self.config = config

    def build(self, scene: PreparedScene) -> SuperpixelGraph:
        segmentation = self._segment_scene(scene)
        features, labels, training_mask, memberships = self._aggregate_superpixels(scene, segmentation)
        adjacency, laplacian = self._construct_graph(features)
        return SuperpixelGraph(
            segmentation=segmentation,
            features=features,
            labels=labels,
            training_mask=training_mask,
            laplacian=laplacian,
            adjacency=adjacency,
            pixel_memberships=memberships,
        )

    def _segment_scene(self, scene: PreparedScene):
        rgb_features = PCA(n_components=3).fit_transform(scene.spectral_features)
        rgb_image = np.zeros((scene.height, scene.width, 3))
        valid_positions = np.unravel_index(scene.valid_indices, (scene.height, scene.width))
        rgb_image[valid_positions] = rgb_features
        value_range = rgb_image.max() - rgb_image.min()
        rgb_image = (rgb_image - rgb_image.min()) / (value_range + 1e-8) * 255
        return slic(
            rgb_image.astype(np.uint8),
            n_segments=self.config.target_segments,
            compactness=self.config.compactness,
            sigma=self.config.smoothing_sigma,
            start_label=0,
        )

    def _aggregate_superpixels(self, scene: PreparedScene, segmentation):
        superpixel_count = int(np.max(segmentation)) + 1
        feature_count = scene.spectral_features.shape[1]
        features = np.zeros((superpixel_count, feature_count + 2))
        labels = -np.ones(superpixel_count, dtype=int)
        training_mask = np.zeros(superpixel_count, dtype=bool)
        memberships = []
        rows, columns = np.unravel_index(scene.valid_indices, (scene.height, scene.width))
        superpixel_ids = segmentation[rows, columns]

        for superpixel_id in range(superpixel_count):
            pixel_indices = np.flatnonzero(superpixel_ids == superpixel_id)
            memberships.append(pixel_indices)
            if pixel_indices.size == 0:
                continue
            features[superpixel_id, :-2] = np.mean(scene.spectral_features[pixel_indices], axis=0)
            features[superpixel_id, -2:] = np.mean(
                np.vstack((rows[pixel_indices], columns[pixel_indices])), axis=1
            )
            training_pixels = pixel_indices[scene.training_mask[pixel_indices]]
            if training_pixels.size > 0:
                labels[superpixel_id] = np.argmax(np.bincount(scene.labels[training_pixels]))
                training_mask[superpixel_id] = True

        features[:, -2:] = StandardScaler().fit_transform(features[:, -2:])
        return features, labels, training_mask, tuple(memberships)

    def _construct_graph(self, features):
        superpixel_count = features.shape[0]
        spectral_neighbors = min(self.config.spectral_neighbors, superpixel_count - 1)
        spatial_neighbors = min(self.config.spatial_neighbors, superpixel_count - 1)
        spectral_adjacency = kneighbors_graph(
            features,
            n_neighbors=spectral_neighbors,
            mode="connectivity",
            include_self=False,
        )
        spatial_adjacency = kneighbors_graph(
            features[:, -2:],
            n_neighbors=spatial_neighbors,
            mode="connectivity",
            include_self=False,
        )
        adjacency = spectral_adjacency.multiply(spatial_adjacency)
        adjacency = 0.5 * (adjacency + adjacency.T)
        degree = sparse.diags(np.asarray(adjacency.sum(axis=1)).ravel())
        return adjacency, degree - adjacency
