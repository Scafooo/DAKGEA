"""PLM augmentation package."""

from src.augmentation.methods.plm.plm_augmenter import PLMAugmenter
from src.augmentation.methods.plm.plm_augmenter_refactored import PLMAugmenterRefactored

# Service layer (new OOP architecture)
from src.augmentation.methods.plm.services import (
    AttributeMatchingService,
    BARTService,
    GraphExpansionService,
)

# Domain models
from src.augmentation.methods.plm.models import (
    AttributeCorrespondence,
    ExpansionContext,
    InterpolationConfig,
)

__all__ = [
    # Legacy augmenter (maintained for backward compatibility)
    "PLMAugmenter",
    # Refactored augmenter (new OOP architecture)
    "PLMAugmenterRefactored",
    # Services
    "AttributeMatchingService",
    "BARTService",
    "GraphExpansionService",
    # Models
    "AttributeCorrespondence",
    "ExpansionContext",
    "InterpolationConfig",
]
