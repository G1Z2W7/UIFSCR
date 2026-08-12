from dataclasses import dataclass
from time import perf_counter

from .config import ExperimentConfig
from .contracts import EvaluationResult, OptimizationResult, PreparedScene, SuperpixelGraph
from .data import PaviaUDataModule
from .evaluation import UIFSCREvaluator
from .graph import SuperpixelGraphBuilder
from .optimizer import UIFSCROptimizer
from .reporting import ExperimentReporter


@dataclass(frozen=True)
class ExperimentArtifacts:
    scene: PreparedScene
    graph: SuperpixelGraph
    optimization: OptimizationResult
    evaluation: EvaluationResult
    elapsed_seconds: float


class PaviaUExperiment:
    def __init__(self, config=None, reporter=None):
        self.config = config or ExperimentConfig()
        self.reporter = reporter or ExperimentReporter()
        self.data_module = PaviaUDataModule(self.config.dataset)
        self.graph_builder = SuperpixelGraphBuilder(self.config.graph)
        self.optimizer = UIFSCROptimizer(self.config.optimizer)
        self.evaluator = UIFSCREvaluator()

    def run(self) -> ExperimentArtifacts:
        experiment_start = perf_counter()
        self.reporter.experiment_header(self.config)

        self.reporter.stage(1, "Loading the PaviaU scene")
        raw_scene = self.data_module.load()
        self.reporter.loaded_scene(raw_scene)

        self.reporter.stage(2, "Standardization, PCA, and stratified sampling")
        prepared_scene = self.data_module.prepare(raw_scene)
        self.reporter.prepared_scene(prepared_scene)

        self.reporter.stage(3, "SLIC aggregation and dual-graph construction")
        graph = self.graph_builder.build(prepared_scene)
        self.reporter.graph_summary(graph)

        self.reporter.stage(4, "Joint UIFSCR representation optimization")
        optimization_start = perf_counter()
        optimization = self.optimizer.fit(graph, iteration_callback=self.reporter.iteration)
        optimization_seconds = perf_counter() - optimization_start
        self.reporter.optimization_summary(optimization, optimization_seconds)

        self.reporter.stage(5, "Full-Z inference and test-pixel evaluation")
        evaluation = self.evaluator.evaluate(prepared_scene, graph, optimization)
        self.reporter.evaluation_summary(evaluation)

        elapsed_seconds = perf_counter() - experiment_start
        self.reporter.experiment_footer(evaluation, elapsed_seconds)
        return ExperimentArtifacts(
            scene=prepared_scene,
            graph=graph,
            optimization=optimization,
            evaluation=evaluation,
            elapsed_seconds=elapsed_seconds,
        )
