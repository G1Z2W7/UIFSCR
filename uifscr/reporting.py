from .config import ExperimentConfig


class ExperimentReporter:
    separator = "=" * 88

    def experiment_header(self, config: ExperimentConfig):
        dataset = config.dataset
        graph = config.graph
        optimizer = config.optimizer
        print(self.separator)
        print("UIFSCR | Pavia University Hyperspectral Classification")
        print(self.separator)
        print("Dataset configuration")
        print(f"  Training ratio             : {dataset.training_ratio * 100:.3f}%")
        print(f"  PCA components             : {dataset.pca_components}")
        print(f"  Target superpixels         : {graph.target_segments}")
        print("UIFSCR objective weights")
        print(f"  Sparse reconstruction      : {optimizer.sparse_weight}")
        print(f"  Graph regularization       : {optimizer.graph_weight}")
        print(f"  OVR classification         : {optimizer.classification_weight}")
        print(f"  Frobenius regularization   : {optimizer.frobenius_weight}")
        print("Optimization configuration")
        print(f"  ADMM penalty               : {optimizer.penalty_parameter}")
        print(f"  SVM regularization         : {optimizer.svm_regularization}")
        print(f"  Maximum iterations         : {optimizer.maximum_iterations}")
        print(f"  Z gradient steps           : {optimizer.z_gradient_steps}")
        print(self.separator)

    def stage(self, sequence, title):
        print(f"\n[{sequence}/5] {title}")

    def loaded_scene(self, scene):
        print(f"  Spectral cube              : {scene.spectral_cube.shape}")
        print(f"  Ground-truth map           : {scene.ground_truth.shape}")
        print(f"  Ground-truth labels        : {sorted(set(scene.ground_truth.ravel()))}")

    def prepared_scene(self, scene):
        print(f"  Valid pixels               : {len(scene.valid_indices)}")
        print(f"  Spectral representation    : {scene.spectral_features.shape}")
        print(f"  Explained variance         : {scene.explained_variance_ratio:.4f}")
        print(f"  Training pixels            : {int(scene.training_mask.sum())}")
        print(f"  Testing pixels             : {int(scene.testing_mask.sum())}")
        print(f"  Number of classes          : {len(scene.classes)}")

    def graph_summary(self, graph):
        print(f"  Segmentation map           : {graph.segmentation.shape}")
        print(f"  Superpixel representation  : {graph.features.shape}")
        print(f"  Labeled superpixels        : {int(graph.training_mask.sum())}")
        print(f"  Adjacency matrix           : {graph.adjacency.shape}")
        print(f"  Laplacian matrix           : {graph.laplacian.shape}")

    def iteration(self, iteration, maximum_iterations, components):
        print(
            f"  Iteration {iteration:02d}/{maximum_iterations:02d} | "
            f"total={components.total:12.6f} | "
            f"nuclear={components.nuclear:10.6f} | "
            f"sparse={components.sparse:10.6f} | "
            f"graph={components.graph:10.6f} | "
            f"hinge={components.classification:10.6f} | "
            f"l2={components.frobenius:10.6f}"
        )

    def optimization_summary(self, optimization, elapsed_seconds):
        print(f"  Learned representation Z  : {optimization.representation.shape}")
        print(f"  Sparse error E             : {optimization.sparse_error.shape}")
        print(f"  Final objective            : {optimization.loss_history[-1]:.6f}")
        print(f"  Optimization time          : {elapsed_seconds:.2f} seconds")

    def evaluation_summary(self, result):
        print(f"  Overall accuracy           : {result.overall_accuracy:.4f}")
        print(f"  Average accuracy           : {result.average_accuracy:.4f}")
        print(f"  Cohen's kappa              : {result.kappa:.4f}")
        print("\nClassification report")
        print(result.classification_report)
        print("Confusion matrix")
        print(result.confusion_matrix)

    def experiment_footer(self, result, elapsed_seconds):
        print("\n" + self.separator)
        print(
            f"UIFSCR final result | OA={result.overall_accuracy:.4f} | "
            f"AA={result.average_accuracy:.4f} | Kappa={result.kappa:.4f} | "
            f"Runtime={elapsed_seconds:.2f}s"
        )
        print(self.separator)
