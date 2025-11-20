"""Configuration loader for statistics module."""

from pathlib import Path
from typing import Dict, List, Any
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.loader import PROJECT_ROOT, load_yaml


class StatisticsConfig:
    """Configuration manager for statistics module."""

    def __init__(self, config_path: Path | None = None):
        """Load configuration from YAML file.

        Args:
            config_path: Path to configuration file. If None, uses default location.
        """
        if config_path is None:
            config_path = PROJECT_ROOT / "config" / "statistics.yaml"

        self.config_path = config_path
        self._config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Statistics configuration not found: {self.config_path}")

        return load_yaml(self.config_path)

    # File names
    @property
    def results_filename(self) -> str:
        return self._config.get("files", {}).get("results", "results.json")

    @property
    def summary_filename(self) -> str:
        return self._config.get("files", {}).get("summary", "summary.json")

    @property
    def metadata_filename(self) -> str:
        return self._config.get("files", {}).get("metadata", "metadata.json")

    # Stages
    @property
    def stages(self) -> List[str]:
        return self._config.get("stages", ["reduction", "augmentation"])

    # Metrics
    @property
    def default_metrics(self) -> List[str]:
        return self._config.get("metrics", {}).get("default", [
            "hits@1", "hits@5", "hits@10", "hits@25", "hits@50",
            "mrr", "mr", "precision", "recall", "f-measure"
        ])

    @property
    def plot_metrics(self) -> List[str]:
        return self._config.get("metrics", {}).get("plot", ["hits@1", "hits@10"])

    @property
    def normalized_metrics(self) -> List[str]:
        """Metrics that should be clamped to [0, 1] range."""
        return self._config.get("filtering", {}).get("normalized_metrics", [])

    # Colors
    @property
    def metric_colors(self) -> Dict[str, str]:
        return self._config.get("colors", {})

    # Plot settings
    @property
    def default_dpi(self) -> int:
        return self._config.get("plots", {}).get("dpi", 200)

    @property
    def figure_size_default(self) -> List[int]:
        return self._config.get("plots", {}).get("figure_size", {}).get("default", [5, 4])

    @property
    def figure_size_ratio_chart(self) -> List[int]:
        return self._config.get("plots", {}).get("figure_size", {}).get("ratio_chart", [8, 5])

    @property
    def figure_size_trend_chart(self) -> List[int]:
        return self._config.get("plots", {}).get("figure_size", {}).get("trend_chart", [8, 4.5])

    @property
    def bar_color_reduction(self) -> str:
        return self._config.get("plots", {}).get("bar_colors", {}).get("reduction", "#457b9d")

    @property
    def bar_color_augmentation(self) -> str:
        return self._config.get("plots", {}).get("bar_colors", {}).get("augmentation", "#e76f51")

    @property
    def ratio_chart_bar_width(self) -> float:
        return self._config.get("plots", {}).get("ratio_chart", {}).get("bar_width", 0.35)

    @property
    def ratio_chart_gap_between_metrics(self) -> float:
        return self._config.get("plots", {}).get("ratio_chart", {}).get("gap_between_metrics", 0.15)

    @property
    def ratio_chart_gap_between_ratios(self) -> float:
        return self._config.get("plots", {}).get("ratio_chart", {}).get("gap_between_ratios", 0.8)

    # Export settings
    @property
    def default_export_formats(self) -> List[str]:
        return self._config.get("export", {}).get("default_formats", ["tsv", "csv"])

    @property
    def plots_output_dir(self) -> str:
        return self._config.get("export", {}).get("output_dirs", {}).get("plots", "results_analysis")

    @property
    def exports_output_dir(self) -> str:
        return self._config.get("export", {}).get("output_dirs", {}).get("exports", "results_analysis")

    # Advanced statistics
    @property
    def advanced_stats_enabled(self) -> bool:
        return self._config.get("advanced_stats", {}).get("enabled", False)

    @property
    def confidence_level(self) -> float:
        return self._config.get("advanced_stats", {}).get("confidence_level", 0.95)

    @property
    def effect_size_thresholds(self) -> Dict[str, float]:
        return self._config.get("advanced_stats", {}).get("effect_size_thresholds", {
            "small": 0.2,
            "medium": 0.5,
            "large": 0.8
        })

    # Advanced plots
    @property
    def advanced_plots_enabled(self) -> bool:
        return self._config.get("advanced_plots", {}).get("enabled", False)

    @property
    def advanced_plot_types(self) -> List[str]:
        return self._config.get("advanced_plots", {}).get("types", [
            "heatmap", "boxplot", "violin", "scatter", "delta_chart"
        ])

    # Filtering
    @property
    def skip_directories(self) -> List[str]:
        return self._config.get("filtering", {}).get("skip_directories", ["logs", "statistics"])


# Global instance (lazy-loaded)
_config_instance: StatisticsConfig | None = None


def get_statistics_config(config_path: Path | None = None) -> StatisticsConfig:
    """Get or create the global statistics configuration instance.

    Args:
        config_path: Optional path to configuration file. Only used on first call.

    Returns:
        StatisticsConfig instance
    """
    global _config_instance

    if _config_instance is None:
        _config_instance = StatisticsConfig(config_path)

    return _config_instance
