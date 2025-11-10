"""Progress bar utilities for experiment orchestration."""

import logging
import sys
from typing import Optional

from tqdm import tqdm

logger = logging.getLogger(__name__)


class ProgressTracker:
    """Wrapper around tqdm that gracefully degrades when progress is disabled."""

    def __init__(self, total: int, enabled: bool) -> None:
        self.enabled = enabled and total > 0

        if not self.enabled and enabled:
            logger.debug("Progress bar disabled (total <= 0)")

        self._bar = (
            tqdm(
                total=total,
                desc="🚀 Running DAKGEA experiments",
                unit="ratio",
                dynamic_ncols=True,
                colour="cyan",
                smoothing=0.3,
                bar_format="{l_bar}{bar:40} | {n_fmt}/{total_fmt} ratios | ⏱️ {elapsed} | ETA {remaining}",
                file=sys.stdout,
            )
            if self.enabled
            else None
        )

    def set_description(self, description: str) -> None:
        if not self.enabled or self._bar is None:
            return
        self._bar.set_description_str(description)

    def step(self) -> None:
        if not self.enabled or self._bar is None:
            return
        self._bar.update(1)

    def close(self) -> None:
        if self.enabled and self._bar is not None:
            self._bar.close()
