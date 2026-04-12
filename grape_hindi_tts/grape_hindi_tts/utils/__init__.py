"""Utility functions and helpers"""

# Import new utility modules
from . import config_utils
from . import audio_utils
from . import logging_utils

# Import from config_utils
from .config_utils import DotDict, load_config, save_config, merge_configs, validate_config

# Import from audio_utils
from .audio_utils import (
    load_audio,
    save_audio,
    compute_mel_spectrogram,
    mel_to_audio_griffin_lim,
    get_audio_duration,
    normalize_audio,
    trim_silence,
    resample_audio,
)

# Import from logging_utils
from .logging_utils import (
    setup_logger,
    MetricsTracker,
    TensorboardLogger,
    WandbLogger,
    ExperimentTracker,
)

# Optional imports for existing modules (if they exist)
try:
    from . import config_loader
except ImportError:
    pass

try:
    from . import text_utils
except ImportError:
    pass

try:
    from . import logging_utils as existing_logging
except ImportError:
    pass

try:
    from . import checkpoint_utils
except ImportError:
    pass

__all__ = [
    # Config utilities
    "config_utils",
    "DotDict",
    "load_config",
    "save_config",
    "merge_configs",
    "validate_config",
    # Audio utilities
    "audio_utils",
    "load_audio",
    "save_audio",
    "compute_mel_spectrogram",
    "mel_to_audio_griffin_lim",
    "get_audio_duration",
    "normalize_audio",
    "trim_silence",
    "resample_audio",
    # Logging utilities
    "logging_utils",
    "setup_logger",
    "MetricsTracker",
    "TensorboardLogger",
    "WandbLogger",
    "ExperimentTracker",
]
