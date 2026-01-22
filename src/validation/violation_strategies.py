"""Violation handling strategies for SHACL validation."""

from enum import Enum
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass
from rdflib import Graph, URIRef, Literal, BNode
from rdflib.namespace import SH

from src.logger import get_logger

logger = get_logger(__name__)


class ViolationStrategy(Enum):
    """Strategy for handling SHACL violations."""

    IGNORE = "ignore"       # Log warning, continue without changes
    REMOVE = "remove"       # Remove triples that violate constraints
    FIX = "fix"             # Attempt automatic correction
    REJECT = "reject"       # Raise exception on any violation


@dataclass
class ValidationViolation:
    """Represents a single SHACL validation violation."""

    focus_node: URIRef          # The node that was validated
    result_path: Optional[URIRef]  # The property path (if applicable)
    value: Optional[any]        # The value that caused the violation
    source_constraint: URIRef   # The constraint component that was violated
    message: str                # Human-readable message
    severity: URIRef            # sh:Violation, sh:Warning, sh:Info

    def __str__(self) -> str:
        return f"[{self.severity.split('#')[-1]}] {self.focus_node}: {self.message}"


@dataclass
class ValidationReport:
    """Summary of SHACL validation results."""

    conforms: bool                          # True if no violations
    violations: List[ValidationViolation]   # List of all violations
    graph_name: str                         # "source" or "target"

    @property
    def violation_count(self) -> int:
        return len(self.violations)

    @property
    def has_errors(self) -> bool:
        """Check if there are any sh:Violation severity issues."""
        return any(
            str(v.severity).endswith("Violation")
            for v in self.violations
        )

    def summary(self) -> str:
        if self.conforms:
            return f"[{self.graph_name}] Validation passed"

        by_severity = {}
        for v in self.violations:
            sev = str(v.severity).split('#')[-1]
            by_severity[sev] = by_severity.get(sev, 0) + 1

        parts = [f"{count} {sev}" for sev, count in by_severity.items()]
        return f"[{self.graph_name}] Validation failed: {', '.join(parts)}"


class ViolationHandler:
    """Handles SHACL violations according to configured strategy."""

    def __init__(
        self,
        strategy: ViolationStrategy = ViolationStrategy.IGNORE,
        custom_fixers: Optional[dict] = None
    ):
        """
        Initialize violation handler.

        Args:
            strategy: How to handle violations
            custom_fixers: Dict mapping constraint URIs to fixer functions
                          Function signature: (graph, violation) -> bool
        """
        self.strategy = strategy
        self.custom_fixers = custom_fixers or {}

        # Built-in fixers for common constraints
        self._builtin_fixers = {
            str(SH.DatatypeConstraintComponent): self._fix_datatype,
            str(SH.MinLengthConstraintComponent): self._fix_min_length,
            str(SH.MaxLengthConstraintComponent): self._fix_max_length,
            str(SH.PatternConstraintComponent): self._fix_pattern,
        }

    def handle(
        self,
        graph: Graph,
        report: ValidationReport
    ) -> Tuple[Graph, ValidationReport]:
        """
        Handle violations according to strategy.

        Args:
            graph: The RDF graph being validated
            report: Validation report with violations

        Returns:
            Tuple of (modified_graph, updated_report)

        Raises:
            ValueError: If strategy is REJECT and violations exist
        """
        if report.conforms:
            return graph, report

        if self.strategy == ViolationStrategy.IGNORE:
            logger.warning(report.summary())
            for v in report.violations[:5]:  # Show first 5
                logger.warning(f"  - {v}")
            if len(report.violations) > 5:
                logger.warning(f"  ... and {len(report.violations) - 5} more")
            return graph, report

        elif self.strategy == ViolationStrategy.REJECT:
            raise ValueError(
                f"SHACL validation failed with {report.violation_count} violations. "
                f"First violation: {report.violations[0]}"
            )

        elif self.strategy == ViolationStrategy.REMOVE:
            return self._remove_violating_triples(graph, report)

        elif self.strategy == ViolationStrategy.FIX:
            return self._fix_violations(graph, report)

        return graph, report

    def _remove_violating_triples(
        self,
        graph: Graph,
        report: ValidationReport
    ) -> Tuple[Graph, ValidationReport]:
        """Remove triples that caused violations."""
        removed_count = 0
        remaining_violations = []

        for violation in report.violations:
            if violation.result_path and violation.value is not None:
                # Try to find and remove the offending triple
                triple = (
                    violation.focus_node,
                    violation.result_path,
                    violation.value if isinstance(violation.value, (URIRef, Literal, BNode))
                    else Literal(violation.value)
                )

                if triple in graph:
                    graph.remove(triple)
                    removed_count += 1
                    logger.debug(f"Removed triple: {triple}")
                else:
                    remaining_violations.append(violation)
            else:
                remaining_violations.append(violation)

        logger.info(f"[{report.graph_name}] Removed {removed_count} violating triples")

        updated_report = ValidationReport(
            conforms=len(remaining_violations) == 0,
            violations=remaining_violations,
            graph_name=report.graph_name
        )

        return graph, updated_report

    def _fix_violations(
        self,
        graph: Graph,
        report: ValidationReport
    ) -> Tuple[Graph, ValidationReport]:
        """Attempt to fix violations using registered fixers."""
        fixed_count = 0
        remaining_violations = []

        for violation in report.violations:
            constraint = str(violation.source_constraint)

            # Check custom fixers first, then builtin
            fixer = self.custom_fixers.get(constraint) or self._builtin_fixers.get(constraint)

            if fixer:
                try:
                    if fixer(graph, violation):
                        fixed_count += 1
                        logger.debug(f"Fixed violation: {violation}")
                    else:
                        remaining_violations.append(violation)
                except Exception as e:
                    logger.warning(f"Failed to fix violation: {e}")
                    remaining_violations.append(violation)
            else:
                # No fixer available, keep violation
                remaining_violations.append(violation)

        logger.info(f"[{report.graph_name}] Fixed {fixed_count} violations, {len(remaining_violations)} remaining")

        updated_report = ValidationReport(
            conforms=len(remaining_violations) == 0,
            violations=remaining_violations,
            graph_name=report.graph_name
        )

        return graph, updated_report

    # --- Built-in fixers ---

    def _fix_datatype(self, graph: Graph, violation: ValidationViolation) -> bool:
        """Attempt to fix datatype violations by casting."""
        # This is a placeholder - actual implementation depends on target datatype
        # Would need to parse the shape to know the expected datatype
        return False

    def _fix_min_length(self, graph: Graph, violation: ValidationViolation) -> bool:
        """Remove values that are too short (can't really fix, just remove)."""
        if violation.result_path and violation.value is not None:
            triple = (violation.focus_node, violation.result_path, Literal(violation.value))
            if triple in graph:
                graph.remove(triple)
                return True
        return False

    def _fix_max_length(self, graph: Graph, violation: ValidationViolation) -> bool:
        """Truncate values that are too long."""
        # Would need to know max length from shape - placeholder
        return False

    def _fix_pattern(self, graph: Graph, violation: ValidationViolation) -> bool:
        """Pattern violations typically can't be auto-fixed."""
        return False
