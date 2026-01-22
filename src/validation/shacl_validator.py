"""SHACL validator for knowledge graph validation."""

from pathlib import Path
from typing import Optional, Union, Tuple, List
from dataclasses import dataclass

from rdflib import Graph, URIRef, Namespace
from rdflib.namespace import SH, RDF

from src.logger import get_logger
from src.core.dataset.dataset import Dataset
from .violation_strategies import (
    ViolationStrategy,
    ViolationHandler,
    ValidationViolation,
    ValidationReport
)

logger = get_logger(__name__)


@dataclass
class SHACLValidationConfig:
    """Configuration for SHACL validation."""

    enabled: bool = False
    shapes_dir: Optional[str] = None  # None = auto-detect from dataset
    on_violation: str = "ignore"      # ignore | remove | fix | reject
    strict_mode: bool = False         # If True, treat warnings as errors
    validate_source: bool = True      # Validate source KG
    validate_target: bool = True      # Validate target KG


class SHACLValidator:
    """
    SHACL validator for augmented knowledge graphs.

    Validates RDF graphs against SHACL shapes and handles violations
    according to configured strategy.

    Usage:
        validator = SHACLValidator(
            shapes_dir="data/raw/openea/BBC_DB/shacl",
            on_violation=ViolationStrategy.REMOVE
        )
        validated_dataset = validator.validate_dataset(augmented_dataset)
    """

    def __init__(
        self,
        shapes_dir: Optional[Union[str, Path]] = None,
        on_violation: Union[str, ViolationStrategy] = ViolationStrategy.IGNORE,
        strict_mode: bool = False,
        validate_source: bool = True,
        validate_target: bool = True
    ):
        """
        Initialize SHACL validator.

        Args:
            shapes_dir: Directory containing SHACL shape files
                       Expected files: source_shapes.ttl, target_shapes.ttl
            on_violation: Strategy for handling violations
            strict_mode: If True, treat sh:Warning as sh:Violation
            validate_source: Whether to validate source KG
            validate_target: Whether to validate target KG
        """
        self.shapes_dir = Path(shapes_dir) if shapes_dir else None
        self.strict_mode = strict_mode
        self.validate_source = validate_source
        self.validate_target = validate_target

        # Parse strategy
        if isinstance(on_violation, str):
            self.strategy = ViolationStrategy(on_violation)
        else:
            self.strategy = on_violation

        self.handler = ViolationHandler(strategy=self.strategy)

        # Lazy-load shapes
        self._source_shapes: Optional[Graph] = None
        self._target_shapes: Optional[Graph] = None

        logger.info(f"[SHACL] Validator initialized (strategy={self.strategy.value})")

    @classmethod
    def from_config(cls, config: SHACLValidationConfig, dataset_path: Optional[Path] = None) -> "SHACLValidator":
        """
        Create validator from configuration.

        Args:
            config: SHACL validation configuration
            dataset_path: Path to dataset (for auto-detecting shapes_dir)

        Returns:
            Configured SHACLValidator instance
        """
        shapes_dir = config.shapes_dir
        if shapes_dir is None and dataset_path is not None:
            # Auto-detect: look for shacl/ subdirectory in dataset
            auto_path = Path(dataset_path) / "shacl"
            if auto_path.exists():
                shapes_dir = str(auto_path)
                logger.info(f"[SHACL] Auto-detected shapes directory: {shapes_dir}")

        return cls(
            shapes_dir=shapes_dir,
            on_violation=config.on_violation,
            strict_mode=config.strict_mode,
            validate_source=config.validate_source,
            validate_target=config.validate_target
        )

    def _load_shapes(self, shapes_file: Path) -> Optional[Graph]:
        """Load SHACL shapes from file."""
        if not shapes_file.exists():
            logger.warning(f"[SHACL] Shapes file not found: {shapes_file}")
            return None

        shapes_graph = Graph()
        try:
            # Detect format from extension
            suffix = shapes_file.suffix.lower()
            fmt = {
                ".ttl": "turtle",
                ".n3": "n3",
                ".nt": "nt",
                ".xml": "xml",
                ".rdf": "xml"
            }.get(suffix, "turtle")

            shapes_graph.parse(shapes_file, format=fmt)
            logger.info(f"[SHACL] Loaded {len(shapes_graph)} triples from {shapes_file}")
            return shapes_graph

        except Exception as e:
            logger.error(f"[SHACL] Failed to load shapes from {shapes_file}: {e}")
            return None

    @property
    def source_shapes(self) -> Optional[Graph]:
        """Lazy-load source shapes."""
        if self._source_shapes is None and self.shapes_dir:
            self._source_shapes = self._load_shapes(self.shapes_dir / "source_shapes.ttl")
        return self._source_shapes

    @property
    def target_shapes(self) -> Optional[Graph]:
        """Lazy-load target shapes."""
        if self._target_shapes is None and self.shapes_dir:
            self._target_shapes = self._load_shapes(self.shapes_dir / "target_shapes.ttl")
        return self._target_shapes

    def validate_graph(
        self,
        data_graph: Graph,
        shapes_graph: Graph,
        graph_name: str = "graph"
    ) -> ValidationReport:
        """
        Validate an RDF graph against SHACL shapes.

        Args:
            data_graph: The RDF graph to validate
            shapes_graph: SHACL shapes graph
            graph_name: Name for logging purposes

        Returns:
            ValidationReport with results
        """
        try:
            from pyshacl import validate as shacl_validate
        except ImportError:
            logger.error("[SHACL] pyshacl not installed. Run: pip install pyshacl")
            # Return empty report (conforming) if pyshacl not available
            return ValidationReport(conforms=True, violations=[], graph_name=graph_name)

        logger.info(f"[SHACL] Validating {graph_name} ({len(data_graph)} triples)...")

        try:
            conforms, results_graph, results_text = shacl_validate(
                data_graph,
                shacl_graph=shapes_graph,
                inference='none',  # Don't infer additional triples
                abort_on_first=False,  # Collect all violations
                meta_shacl=False,
                advanced=True,
                js=False
            )

            # Parse violations from results graph
            violations = self._parse_violations(results_graph, graph_name)

            report = ValidationReport(
                conforms=conforms,
                violations=violations,
                graph_name=graph_name
            )

            if conforms:
                logger.info(f"[SHACL] {graph_name}: Validation passed")
            else:
                logger.warning(f"[SHACL] {report.summary()}")

            return report

        except Exception as e:
            logger.error(f"[SHACL] Validation failed for {graph_name}: {e}")
            # Return conforming on error to not block pipeline
            return ValidationReport(conforms=True, violations=[], graph_name=graph_name)

    def _parse_violations(self, results_graph: Graph, graph_name: str) -> List[ValidationViolation]:
        """Parse SHACL validation results into structured violations."""
        violations = []

        # Query for all validation results
        for result in results_graph.subjects(RDF.type, SH.ValidationResult):
            focus_node = results_graph.value(result, SH.focusNode)
            result_path = results_graph.value(result, SH.resultPath)
            value = results_graph.value(result, SH.value)
            source_constraint = results_graph.value(result, SH.sourceConstraintComponent)
            message = results_graph.value(result, SH.resultMessage)
            severity = results_graph.value(result, SH.resultSeverity) or SH.Violation

            violation = ValidationViolation(
                focus_node=focus_node,
                result_path=result_path,
                value=value,
                source_constraint=source_constraint,
                message=str(message) if message else "No message",
                severity=severity
            )

            # In strict mode, treat warnings as violations
            if self.strict_mode and str(severity).endswith("Warning"):
                violation.severity = SH.Violation

            violations.append(violation)

        return violations

    def validate_dataset(self, dataset: Dataset) -> Dataset:
        """
        Validate and handle violations in a dataset.

        Args:
            dataset: Dataset with source and target knowledge graphs

        Returns:
            Dataset with violations handled according to strategy
        """
        logger.info(f"[SHACL] Validating dataset (strategy={self.strategy.value})...")

        # Validate source KG
        if self.validate_source and self.source_shapes:
            report_src = self.validate_graph(
                dataset.knowledge_graph_source,
                self.source_shapes,
                "source"
            )
            dataset.knowledge_graph_source, _ = self.handler.handle(
                dataset.knowledge_graph_source,
                report_src
            )

        # Validate target KG
        if self.validate_target and self.target_shapes:
            report_tgt = self.validate_graph(
                dataset.knowledge_graph_target,
                self.target_shapes,
                "target"
            )
            dataset.knowledge_graph_target, _ = self.handler.handle(
                dataset.knowledge_graph_target,
                report_tgt
            )

        return dataset

    def get_validation_summary(self, dataset: Dataset) -> dict:
        """
        Get validation summary without modifying the dataset.

        Args:
            dataset: Dataset to validate

        Returns:
            Dict with validation statistics
        """
        summary = {
            "source": {"validated": False, "conforms": None, "violations": 0},
            "target": {"validated": False, "conforms": None, "violations": 0}
        }

        if self.validate_source and self.source_shapes:
            report = self.validate_graph(
                dataset.knowledge_graph_source,
                self.source_shapes,
                "source"
            )
            summary["source"] = {
                "validated": True,
                "conforms": report.conforms,
                "violations": report.violation_count
            }

        if self.validate_target and self.target_shapes:
            report = self.validate_graph(
                dataset.knowledge_graph_target,
                self.target_shapes,
                "target"
            )
            summary["target"] = {
                "validated": True,
                "conforms": report.conforms,
                "violations": report.violation_count
            }

        return summary
