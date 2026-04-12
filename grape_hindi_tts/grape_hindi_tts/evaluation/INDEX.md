# Evaluation Module - File Index

Complete inference and evaluation pipeline for SupertonicTTS Hindi.

## Files Created

### Python Modules (2,235 lines total)

#### 1. inference.py (586 lines, 19 KB)
Main synthesis module with `SupertonicTTSInference` class.

**Key Components:**
- `SupertonicTTSInference`: Main inference class
- `synthesize()`: Single text-to-speech synthesis
- `synthesize_batch()`: Batch processing
- Flow matching with Euler ODE solver
- Classifier-free guidance (CFG) support
- Duration scaling control
- Real-time factor (RTF) measurement

**Features:**
- Text tokenization and normalization
- Reference audio encoding
- Duration prediction
- Latent generation via flow matching
- Waveform decoding
- GPU/CPU support
- Batch processing

**CLI Interface:**
```bash
python -m supertonic_hindi_tts.evaluation.inference \
    --autoencoder path/to/model.pt \
    --text-to-latent path/to/model.pt \
    --duration-predictor path/to/model.pt \
    --text "Hindi text" \
    --reference ref.wav \
    --output output.wav
```

---

#### 2. metrics.py (456 lines, 15 KB)
Quality metrics computation with `MetricComputer` class.

**Key Components:**
- `MetricComputer`: Metric computation class
- Speech quality metrics: UTMOS, PESQ, STOI
- Text quality metrics: WER, CER (via Whisper ASR)
- Speaker metrics: Cosine similarity of embeddings
- Efficiency metrics: Real-Time Factor (RTF)

**Metrics Computed:**
- WER: Word Error Rate (Whisper ASR, Hindi)
- CER: Character Error Rate
- UTMOS: MOS prediction (1-5)
- PESQ: Perceptual quality (-0.5 to 4.5)
- STOI: Intelligibility (0-1)
- Speaker Similarity: Embedding cosine distance (0-1)
- RTF: Real-time factor

**Features:**
- Lazy model loading
- Supports multiple speaker encoder backends
- Comprehensive error handling
- Whisper transcription with Hindi language

**CLI Interface:**
```bash
python -m supertonic_hindi_tts.evaluation.metrics \
    --gen-audio generated.wav \
    --ref-audio reference.wav \
    --text "Hindi text" \
    --time 2.5
```

---

#### 3. evaluate.py (565 lines, 19 KB)
Full evaluation pipeline with `SupertonicEvaluator` class.

**Key Components:**
- `SupertonicEvaluator`: Complete evaluation orchestrator
- `evaluate_manifest()`: Run evaluation on test manifests
- Manifest loading (JSONL, CSV, JSON)
- Sample filtering (speaker, duration, gender)
- Per-sample evaluation and synthesis
- Aggregated metrics and reporting

**Features:**
- Multi-format manifest support (JSONL, CSV, JSON)
- Filtering by speaker ID, duration range, gender
- Per-sample metric computation
- CSV detailed results export
- JSON summary export
- Speaker-wise performance breakdown
- Console output of statistics

**Manifest Format:**
```json
{"text": "...", "reference_audio": "...", "audio": "...", "speaker": "...", "id": "..."}
```

**Output Files:**
- `results_detailed.csv`: Per-sample metrics
- `summary.json`: Aggregated statistics

**CLI Interface:**
```bash
python -m supertonic_hindi_tts.evaluation.evaluate \
    --manifest test_manifest.jsonl \
    --output results/ \
    --speaker speaker1 \
    --duration-min 2.0 --duration-max 10.0 \
    --max-samples 50 \
    [checkpoint paths...]
```

---

#### 4. generate_samples.py (607 lines, 19 KB)
Demo sample generation with `SampleGenerator` class.

**Key Components:**
- `SampleGenerator`: Demo sample generation
- `generate_samples()`: Generate audio from sentences
- Interactive HTML page generation
- 25+ default Hindi test sentences

**Default Test Sentences (6 categories):**
1. Simple declarative sentences (5)
2. Questions (5)
3. Exclamations (5)
4. Long paragraph (~30 seconds, 1)
5. Numbers and dates (5)
6. Hindi-English code-switching (5)

**Features:**
- 25+ pre-built Hindi test sentences
- Organized by semantic category
- Interactive HTML page with audio players
- Responsive design
- Metadata JSON export
- Audio duration and RTF logging

**Output:**
- Audio files: `{category}_{index}.wav`
- HTML page: `index.html` (open in browser)
- Metadata: `metadata.json`

**CLI Interface:**
```bash
python -m supertonic_hindi_tts.evaluation.generate_samples \
    --autoencoder path/to/model.pt \
    --text-to-latent path/to/model.pt \
    --duration-predictor path/to/model.pt \
    --reference speaker.wav \
    --output demo/
```

---

#### 5. __init__.py (21 lines, 0.7 KB)
Module package initialization and exports.

**Exports:**
```python
from .inference import SupertonicTTSInference
from .metrics import MetricComputer
from .evaluate import SupertonicEvaluator
from .generate_samples import SampleGenerator, DEFAULT_TEST_SENTENCES
```

**Usage:**
```python
from supertonic_hindi_tts.evaluation import SupertonicTTSInference
from supertonic_hindi_tts.evaluation import MetricComputer
from supertonic_hindi_tts.evaluation import SupertonicEvaluator
from supertonic_hindi_tts.evaluation import SampleGenerator
```

---

### Documentation (1,288 lines total)

#### 1. README.md (300 lines, 12 KB)
**Overview and Reference**

Contents:
- Project overview
- Architecture description
- Feature highlights
- Performance characteristics
- Quick start examples
- Dependencies
- File structure
- Citation information

**Use this to:**
- Understand the system architecture
- Get high-level overview
- Find quick examples
- Understand performance expectations

---

#### 2. QUICK_START.md (150 lines, 7.7 KB)
**5-Minute Getting Started Guide**

Contents:
- Installation (30 seconds)
- Code examples (CLI and Python API)
- Common parameters
- Output interpretation
- Troubleshooting
- Tips and tricks
- Expected results

**Use this to:**
- Get running in 5 minutes
- Copy-paste working examples
- Understand common issues
- See expected outputs

---

#### 3. EVALUATION_GUIDE.md (400 lines, 13 KB)
**Comprehensive Reference Manual**

Contents:
- Installation with all dependencies
- Complete usage examples (CLI and Python API)
- Parameter descriptions
- Manifest format specifications
- Output format documentation
- Architecture details (algorithm explanation)
- Performance tips and trade-offs
- Evaluation results interpretation
- Troubleshooting guide (comprehensive)

**Use this to:**
- Learn complete system
- Understand all parameters
- Debug issues
- Optimize for your use case
- Interpret results

---

#### 4. INDEX.md (this file)
**File Index and Navigation**

Quick reference to all files and their purposes.

---

## Quick Navigation

### I want to...

**Synthesize Hindi text:**
→ `inference.py` or QUICK_START.md Example 1

**Evaluate quality metrics:**
→ `metrics.py` or README.md Metrics section

**Run full evaluation on test data:**
→ `evaluate.py` or QUICK_START.md Example 3

**Generate demo samples:**
→ `generate_samples.py` or QUICK_START.md Example 2

**Understand the system:**
→ README.md

**Get started quickly:**
→ QUICK_START.md

**Find detailed documentation:**
→ EVALUATION_GUIDE.md

**Debug an issue:**
→ EVALUATION_GUIDE.md Troubleshooting section

---

## Key Statistics

| Metric | Value |
|--------|-------|
| Python Code | 2,235 lines |
| Documentation | 1,288 lines |
| Total Files | 8 (5 Python + 3 Markdown + __pycache__) |
| Total Size | 116 KB |
| Classes | 4 main classes |
| CLI Commands | 4 (inference, metrics, evaluate, generate_samples) |
| Metrics Computed | 7 (WER, CER, UTMOS, PESQ, STOI, Speaker Sim, RTF) |
| Default Test Sentences | 25+ in 6 categories |

---

## Architecture Overview

```
evaluation/
├── Core Modules (Python)
│   ├── inference.py        # Synthesis: text → waveform
│   ├── metrics.py          # Quality metrics computation
│   ├── evaluate.py         # Full evaluation pipeline
│   ├── generate_samples.py # Demo sample generation
│   └── __init__.py         # Package exports
│
├── Documentation (Markdown)
│   ├── README.md           # Overview & architecture
│   ├── QUICK_START.md      # 5-minute guide
│   ├── EVALUATION_GUIDE.md # Comprehensive reference
│   └── INDEX.md            # This file
│
└── Generated Outputs (example)
    ├── results/
    │   ├── results_detailed.csv
    │   ├── summary.json
    │   └── audio files...
    ├── demo/
    │   ├── index.html
    │   ├── metadata.json
    │   └── audio files...
```

---

## Getting Started Checklist

- [ ] Install dependencies: `pip install openai-whisper speechbrain pesq pystoi jiwer`
- [ ] Review QUICK_START.md (5 minutes)
- [ ] Run first synthesis (1 minute)
- [ ] Generate demo samples (2 minutes)
- [ ] Read EVALUATION_GUIDE.md (10 minutes)
- [ ] Prepare test manifest (JSONL format)
- [ ] Run full evaluation
- [ ] Analyze results (CSV + JSON)

---

## Common Tasks

### Task 1: Single Synthesis
Use: `inference.py`
Reference: QUICK_START.md Example 1, README.md

### Task 2: Batch Evaluation
Use: `evaluate.py`
Reference: QUICK_START.md Example 3, EVALUATION_GUIDE.md

### Task 3: Demo Generation
Use: `generate_samples.py`
Reference: QUICK_START.md Example 2

### Task 4: Custom Metrics
Use: `metrics.py`
Reference: EVALUATION_GUIDE.md, README.md Metrics section

### Task 5: Debugging Issues
Use: EVALUATION_GUIDE.md Troubleshooting section

---

## Summary

Complete inference and evaluation pipeline for SupertonicTTS Hindi:

✓ 2,235 lines of production-ready Python  
✓ 1,288 lines of comprehensive documentation  
✓ 4 main classes covering all functionality  
✓ 7 quality metrics (WER, CER, UTMOS, PESQ, STOI, Speaker Sim, RTF)  
✓ 25+ default Hindi test sentences  
✓ Both CLI and Python API support  
✓ Interactive HTML demo page  
✓ Detailed CSV and JSON results  

Ready for research, development, and production use.

---

**Created:** April 12, 2026  
**Status:** Production Ready  
**Version:** 1.0
