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
    def metric_groups(self) -> Dict[str, List[str]]:
        """Get metric groups (ranking, classification, etc.)."""
        return self._config.get("metrics", {}).get("groups", {
            "ranking": ["hits@1", "hits@5", "hits@10", "mrr", "mr"],
            "classification": ["precision", "recall", "f-measure"]
        })

    @property
    def plot_metrics(self) -> Dict[str, List[str]]:
        """Get plot metrics organized by group."""
        plot_config = self._config.get("metrics", {}).get("plot", {})
        if isinstance(plot_config, list):
            # Backward compatibility: if plot is a list, return as ranking metrics
            return {"ranking": plot_config}
        return plot_config

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
        """Legacy property for backward compatibility. Use stage_color_reduction_primary instead."""
        return self.stage_color_reduction_primary

    @property
    def bar_color_augmentation(self) -> str:
        """Legacy property for backward compatibility. Use stage_color_augmentation_primary instead."""
        return self.stage_color_augmentation_primary

    # Unified stage colors (NEW)
    @property
    def stage_color_reduction_primary(self) -> str:
        return self._config.get("plots", {}).get("stage_colors", {}).get("reduction", {}).get("primary", "#264653")

    @property
    def stage_color_reduction_light(self) -> str:
        return self._config.get("plots", {}).get("stage_colors", {}).get("reduction", {}).get("light", "#2a9d8f")

    @property
    def stage_color_reduction_linestyle(self) -> str:
        return self._config.get("plots", {}).get("stage_colors", {}).get("reduction", {}).get("linestyle", "--")

    @property
    def stage_color_reduction_marker(self) -> str:
        return self._config.get("plots", {}).get("stage_colors", {}).get("reduction", {}).get("marker", "x")

    @property
    def stage_color_reduction_alpha(self) -> float:
        return self._config.get("plots", {}).get("stage_colors", {}).get("reduction", {}).get("alpha", 0.85)

    @property
    def stage_color_augmentation_primary(self) -> str:
        return self._config.get("plots", {}).get("stage_colors", {}).get("augmentation", {}).get("primary", "#e76f51")

    @property
    def stage_color_augmentation_light(self) -> str:
        return self._config.get("plots", {}).get("stage_colors", {}).get("augmentation", {}).get("light", "#f4a261")

    @property
    def stage_color_augmentation_linestyle(self) -> str:
        return self._config.get("plots", {}).get("stage_colors", {}).get("augmentation", {}).get("linestyle", "-")

    @property
    def stage_color_augmentation_marker(self) -> str:
        return self._config.get("plots", {}).get("stage_colors", {}).get("augmentation", {}).get("marker", "o")

    @property
    def stage_color_augmentation_alpha(self) -> float:
        return self._config.get("plots", {}).get("stage_colors", {}).get("augmentation", {}).get("alpha", 0.85)

    # Delta/improvement colors
    @property
    def delta_color_positive(self) -> str:
        return self._config.get("plots", {}).get("delta_colors", {}).get("positive", "#2a9d8f")

    @property
    def delta_color_negative(self) -> str:
        return self._config.get("plots", {}).get("delta_colors", {}).get("negative", "#e63946")

    @property
    def delta_color_neutral(self) -> str:
        return self._config.get("plots", {}).get("delta_colors", {}).get("neutral", "#6c757d")

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
    def plots_ranking_output_dir(self) -> str:
        return self._config.get("export", {}).get("output_dirs", {}).get("plots_ranking", "results_analysis/ranking_metrics")

    @property
    def plots_classification_output_dir(self) -> str:
        return self._config.get("export", {}).get("output_dirs", {}).get("plots_classification", "results_analysis/classification_metrics")

    def get_plots_output_dir_for_group(self, group: str) -> str:
        """Get output directory for a specific metric group."""
        if group == "ranking":
            return self.plots_ranking_output_dir
        elif group == "classification":
            return self.plots_classification_output_dir
        else:
            return self.plots_output_dir

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

    # New figure sizes
    @property
    def figure_size_radar_chart(self) -> List[int]:
        return self._config.get("plots", {}).get("figure_size", {}).get("radar_chart", [7, 7])

    @property
    def figure_size_ridge_plot(self) -> List[int]:
        return self._config.get("plots", {}).get("figure_size", {}).get("ridge_plot", [10, 6])

    @property
    def figure_size_performance_matrix(self) -> List[int]:
        return self._config.get("plots", {}).get("figure_size", {}).get("performance_matrix", [10, 8])

    # Console settings
    @property
    def console_use_rich(self) -> bool:
        return self._config.get("console", {}).get("use_rich", True)

    @property
    def console_show_progress(self) -> bool:
        return self._config.get("console", {}).get("show_progress", True)

    @property
    def console_colors(self) -> Dict[str, str]:
        return self._config.get("console", {}).get("colors", {})

    @property
    def console_delta_thresholds(self) -> Dict[str, float]:
        return self._config.get("console", {}).get("delta_thresholds", {})

    @property
    def console_summary_table_show(self) -> bool:
        return self._config.get("console", {}).get("summary_table", {}).get("show", True)

    @property
    def console_summary_table_top_n(self) -> int:
        return self._config.get("console", {}).get("summary_table", {}).get("top_n", 5)

    @property
    def console_summary_table_sort_by(self) -> str:
        return self._config.get("console", {}).get("summary_table", {}).get("sort_by", "hits@1")

    # Outlier detection
    @property
    def outliers_enabled(self) -> bool:
        return self._config.get("outliers", {}).get("enabled", True)

    @property
    def outliers_method(self) -> str:
        return self._config.get("outliers", {}).get("method", "iqr")

    @property
    def outliers_iqr_multiplier(self) -> float:
        return self._config.get("outliers", {}).get("iqr_multiplier", 1.5)

    @property
    def outliers_zscore_threshold(self) -> float:
        return self._config.get("outliers", {}).get("zscore_threshold", 3.0)

    @property
    def outliers_highlight_in_plots(self) -> bool:
        return self._config.get("outliers", {}).get("highlight_in_plots", True)

    # Best/worst tracking
    @property
    def tracking_enabled(self) -> bool:
        return self._config.get("tracking", {}).get("enabled", True)

    @property
    def tracking_criteria(self) -> List[Dict[str, Any]]:
        return self._config.get("tracking", {}).get("criteria", [])

    @property
    def tracking_save_summary(self) -> bool:
        return self._config.get("tracking", {}).get("save_summary", True)


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
