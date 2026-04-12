"""
Demo sample generation script for SupertonicTTS Hindi.

Generates audio samples from Hindi sentences covering:
- Simple declarative sentences
- Questions
- Exclamations
- Long paragraphs (testing ~30 sec generation)
- Numbers and dates
- Mixed Hindi-English code-switching

Creates an HTML page for easy listening comparison.
"""

import logging
import json
from pathlib import Path
from typing import List, Dict, Optional
import argparse

import torch

from .inference import SupertonicTTSInference

logger = logging.getLogger(__name__)


# Default Hindi test sentences covering different scenarios
DEFAULT_TEST_SENTENCES = {
    "simple_declarative": [
        "मुझे बहुत खुशी है आपसे मिलकर।",
        "यह एक सुंदर दिन है।",
        "मैं हर दिन सुबह व्यायाम करता हूँ।",
        "भारत एक महान देश है।",
        "तकनीक हमारे जीवन को आसान बनाती है।",
    ],
    "questions": [
        "आप कैसे हैं?",
        "आपका नाम क्या है?",
        "क्या आप कल आ सकते हैं?",
        "भारत की राजधानी कौन सी है?",
        "आपको कौन सी चीज़ सबसे ज़्यादा पसंद है?",
    ],
    "exclamations": [
        "वाह, कितना शानदार है!",
        "यह तो अद्भुत है!",
        "मुझे विश्वास नहीं हुआ!",
        "बहुत बढ़िया!",
        "कितना सुंदर दृश्य है!",
    ],
    "long_paragraph": [
        "आर्टिफिशियल इंटेलिजेंस एक क्रांतिकारी तकनीक है जो हमारी दुनिया को बदल रही है। "
        "यह मशीन लर्निंग, गहरे नेटवर्क और डेटा विश्लेषण का उपयोग करके मानव-स्तर के कार्य पूरे करती है। "
        "भविष्य में, कृत्रिम बुद्धिमत्ता हमारे स्वास्थ्य, शिक्षा और व्यवसायों को रूपांतरित करेगी।",
    ],
    "numbers_and_dates": [
        "आज की तारीख पन्द्रह अप्रैल दो हज़ार छब्बीस है।",
        "पचास प्रतिशत की छूट पाएं।",
        "कीमत नौ सौ निन्यानबे रुपये है।",
        "दो हज़ार बीस में यह फिल्म रिलीज़ हुई थी।",
        "मेरा जन्म उन्नीस सौ नब्बे के दशक में हुआ।",
    ],
    "code_switching": [
        "Artificial Intelligence को Hindi में कृत्रिम बुद्धिमत्ता कहते हैं।",
        "मैं हर दिन software development करता हूँ।",
        "Technology की दुनिया बहुत तेजी से बदल रही है।",
        "इस project को complete करने में three months लगेंगे।",
        "आजकल सभी companies remote work policy दे रही हैं।",
    ],
}


class SampleGenerator:
    """Generate demo samples for SupertonicTTS Hindi."""

    def __init__(
        self,
        autoencoder_path: str,
        text_to_latent_path: str,
        duration_predictor_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Initialize sample generator.

        Args:
            autoencoder_path: Path to speech autoencoder checkpoint
            text_to_latent_path: Path to text-to-latent checkpoint
            duration_predictor_path: Path to duration predictor checkpoint
            device: Device for inference
        """
        self.device = device

        logger.info("Initializing inference pipeline...")
        self.inference = SupertonicTTSInference(
            autoencoder_path=autoencoder_path,
            text_to_latent_path=text_to_latent_path,
            duration_predictor_path=duration_predictor_path,
            device=device,
        )

    def generate_samples(
        self,
        reference_audio_path: str,
        output_dir: str,
        sentences: Optional[Dict[str, List[str]]] = None,
        cfg_scale: float = 3.0,
        duration_scale: float = 1.0,
        n_steps: int = 32,
    ) -> Dict[str, Dict]:
        """Generate samples from sentences.

        Args:
            reference_audio_path: Path to reference audio for speaker
            output_dir: Directory to save generated audio
            sentences: Dictionary of {category: [sentences]}. If None, uses defaults.
            cfg_scale: Classifier-free guidance scale
            duration_scale: Duration scaling factor
            n_steps: Number of ODE solver steps

        Returns:
            Dictionary with metadata about generated samples
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if sentences is None:
            sentences = DEFAULT_TEST_SENTENCES

        samples_data = {
            "reference_audio": reference_audio_path,
            "cfg_scale": cfg_scale,
            "duration_scale": duration_scale,
            "n_steps": n_steps,
            "categories": {},
        }

        total = sum(len(sents) for sents in sentences.values())
        count = 0

        # Generate samples for each category
        for category, sentence_list in sentences.items():
            logger.info(f"\nGenerating samples for category: {category}")
            samples_data["categories"][category] = []

            for i, sentence in enumerate(sentence_list):
                count += 1
                logger.info(f"[{count}/{total}] {sentence[:60]}...")

                try:
                    # Generate audio
                    waveform, rtf = self.inference.synthesize(
                        text=sentence,
                        reference_audio_path=reference_audio_path,
                        duration_scale=duration_scale,
                        cfg_scale=cfg_scale,
                        n_steps=n_steps,
                        return_rtf=True,
                    )

                    # Save audio
                    audio_filename = f"{category}_{i:02d}.wav"
                    audio_path = output_dir / audio_filename
                    self.inference.save_audio(waveform, str(audio_path))

                    # Record metadata
                    audio_duration = len(waveform) / self.inference.sample_rate
                    samples_data["categories"][category].append({
                        "index": i,
                        "text": sentence,
                        "audio_file": audio_filename,
                        "audio_duration": audio_duration,
                        "rtf": rtf,
                    })

                except Exception as e:
                    logger.error(f"Error generating sample: {e}")
                    samples_data["categories"][category].append({
                        "index": i,
                        "text": sentence,
                        "error": str(e),
                    })

        # Save metadata
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(samples_data, f, indent=2, ensure_ascii=False)

        # Generate HTML page
        html_path = output_dir / "index.html"
        self._generate_html(samples_data, html_path, output_dir)

        logger.info(f"\nSample generation complete!")
        logger.info(f"Audio files saved to: {output_dir}")
        logger.info(f"HTML page: {html_path}")

        return samples_data

    def _generate_html(
        self,
        samples_data: Dict,
        output_path: Path,
        audio_dir: Path
    ):
        """Generate HTML page for listening to samples.

        Args:
            samples_data: Dictionary with sample metadata
            output_path: Path to save HTML file
            audio_dir: Directory containing audio files
        """
        html = """<!DOCTYPE html>
<html lang="hi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SupertonicTTS Hindi - Demo Samples</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Noto Sans Devanagari", sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 1000px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            padding: 40px;
        }

        h1 {
            color: #333;
            margin-bottom: 10px;
            text-align: center;
        }

        .subtitle {
            color: #666;
            text-align: center;
            margin-bottom: 30px;
            font-size: 14px;
        }

        .settings {
            background: #f5f5f5;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 30px;
            font-size: 13px;
        }

        .settings-item {
            display: inline-block;
            margin-right: 20px;
            margin-bottom: 10px;
        }

        .settings-label {
            font-weight: 600;
            color: #333;
        }

        .category-section {
            margin-bottom: 40px;
        }

        .category-title {
            background: #667eea;
            color: white;
            padding: 12px 15px;
            border-radius: 6px;
            margin-bottom: 15px;
            font-size: 16px;
            font-weight: 600;
        }

        .sample-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 15px;
        }

        .sample-card {
            background: #f9f9f9;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            transition: all 0.3s ease;
        }

        .sample-card:hover {
            border-color: #667eea;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.1);
        }

        .sample-text {
            color: #333;
            font-size: 14px;
            margin-bottom: 10px;
            line-height: 1.5;
        }

        .sample-info {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
            font-size: 12px;
            color: #666;
        }

        .sample-duration {
            background: #e8f0fe;
            color: #667eea;
            padding: 2px 6px;
            border-radius: 4px;
            font-weight: 500;
        }

        .sample-rtf {
            background: #fce4ec;
            color: #764ba2;
            padding: 2px 6px;
            border-radius: 4px;
            font-weight: 500;
        }

        audio {
            width: 100%;
            margin-top: 10px;
            border-radius: 4px;
        }

        .error-message {
            color: #d32f2f;
            font-size: 12px;
            margin-top: 10px;
            padding: 8px;
            background: #ffebee;
            border-radius: 4px;
        }

        footer {
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #666;
            font-size: 12px;
        }

        .back-to-top {
            position: fixed;
            bottom: 20px;
            right: 20px;
            width: 50px;
            height: 50px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 50%;
            cursor: pointer;
            display: none;
            align-items: center;
            justify-content: center;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        }

        .back-to-top:hover {
            background: #764ba2;
        }

        .back-to-top.show {
            display: flex;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎙️ SupertonicTTS Hindi</h1>
        <p class="subtitle">Demo Samples - Text-to-Speech Synthesis</p>

        <div class="settings">
            <div class="settings-item">
                <span class="settings-label">CFG Scale:</span> {cfg_scale}
            </div>
            <div class="settings-item">
                <span class="settings-label">Duration Scale:</span> {duration_scale}
            </div>
            <div class="settings-item">
                <span class="settings-label">ODE Steps:</span> {n_steps}
            </div>
        </div>

        {categories_html}

        <footer>
            <p>Generated with SupertonicTTS Hindi | {timestamp}</p>
        </footer>
    </div>

    <button class="back-to-top" onclick="window.scrollTo({top: 0, behavior: 'smooth'})">
        ↑
    </button>

    <script>
        // Show back-to-top button on scroll
        window.addEventListener('scroll', function() {
            const btn = document.querySelector('.back-to-top');
            if (window.scrollY > 300) {
                btn.classList.add('show');
            } else {
                btn.classList.remove('show');
            }
        });
    </script>
</body>
</html>
"""

        # Build categories HTML
        categories_html = ""
        for category, samples in samples_data["categories"].items():
            categories_html += f'<div class="category-section">\n'
            categories_html += f'<div class="category-title">{self._format_category_title(category)}</div>\n'
            categories_html += '<div class="sample-grid">\n'

            for sample in samples:
                if "error" in sample:
                    categories_html += f'''<div class="sample-card">
<div class="sample-text">{self._escape_html(sample["text"])}</div>
<div class="error-message">Error: {self._escape_html(sample["error"])}</div>
</div>\n'''
                else:
                    audio_file = sample["audio_file"]
                    duration = sample.get("audio_duration", 0)
                    rtf = sample.get("rtf", 0)

                    categories_html += f'''<div class="sample-card">
<div class="sample-text">{self._escape_html(sample["text"])}</div>
<div class="sample-info">
    <span class="sample-duration">Duration: {duration:.2f}s</span>
    <span class="sample-rtf">RTF: {rtf:.3f}</span>
</div>
<audio controls>
    <source src="{audio_file}" type="audio/wav">
    Your browser does not support the audio element.
</audio>
</div>\n'''

            categories_html += '</div>\n</div>\n'

        # Fill in template
        from datetime import datetime
        html = html.format(
            cfg_scale=samples_data["cfg_scale"],
            duration_scale=samples_data["duration_scale"],
            n_steps=samples_data["n_steps"],
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            categories_html=categories_html,
        )

        # Write HTML
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

    @staticmethod
    def _format_category_title(category: str) -> str:
        """Format category name for display.

        Args:
            category: Category name (snake_case)

        Returns:
            Formatted title
        """
        title_map = {
            "simple_declarative": "Simple Declarative Sentences",
            "questions": "Questions",
            "exclamations": "Exclamations",
            "long_paragraph": "Long Paragraph (Extended)",
            "numbers_and_dates": "Numbers & Dates",
            "code_switching": "Code-Switching (Hindi-English)",
        }
        return title_map.get(category, category.replace("_", " ").title())

    @staticmethod
    def _escape_html(text: str) -> str:
        """Escape HTML special characters.

        Args:
            text: Text to escape

        Returns:
            Escaped text
        """
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
        )


def main():
    """CLI interface for sample generation."""
    parser = argparse.ArgumentParser(
        description="Generate demo samples for SupertonicTTS Hindi"
    )

    parser.add_argument(
        "--autoencoder",
        required=True,
        help="Path to speech autoencoder checkpoint"
    )
    parser.add_argument(
        "--text-to-latent",
        required=True,
        help="Path to text-to-latent checkpoint"
    )
    parser.add_argument(
        "--duration-predictor",
        required=True,
        help="Path to duration predictor checkpoint"
    )
    parser.add_argument(
        "--reference",
        required=True,
        help="Path to reference audio"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for samples and HTML"
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=3.0,
        help="Classifier-free guidance scale"
    )
    parser.add_argument(
        "--duration-scale",
        type=float,
        default=1.0,
        help="Duration scaling factor"
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=32,
        help="Number of ODE solver steps"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Initialize generator
    generator = SampleGenerator(
        autoencoder_path=args.autoencoder,
        text_to_latent_path=args.text_to_latent,
        duration_predictor_path=args.duration_predictor,
        device=args.device,
    )

    # Generate samples
    generator.generate_samples(
        reference_audio_path=args.reference,
        output_dir=args.output,
        sentences=None,  # Use defaults
        cfg_scale=args.cfg_scale,
        duration_scale=args.duration_scale,
        n_steps=args.n_steps,
    )

    print(f"\nSample generation complete!")
    print(f"Output directory: {args.output}")
    print(f"Open index.html in your browser to listen")


if __name__ == "__main__":
    main()
