"""Evaluation and metrics modules for SupertonicTTS Hindi.

Complete evaluation pipeline including:
- Inference: SupertonicTTSInference class for synthesis
- Metrics: MetricComputer for WER, CER, UTMOS, PESQ, STOI, speaker similarity, RTF
- Evaluate: SupertonicEvaluator for comprehensive evaluation on manifests
- Generate Samples: SampleGenerator for demo sample generation with HTML page
"""

from .inference import SupertonicTTSInference
from .metrics import MetricComputer
from .evaluate import SupertonicEvaluator
from .generate_samples import SampleGenerator, DEFAULT_TEST_SENTENCES

__all__ = [
    "SupertonicTTSInference",
    "MetricComputer",
    "SupertonicEvaluator",
    "SampleGenerator",
    "DEFAULT_TEST_SENTENCES",
]
