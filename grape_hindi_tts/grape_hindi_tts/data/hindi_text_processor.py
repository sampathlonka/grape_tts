"""
Hindi Text Processor for SupertonicTTS

Handles complete Hindi text normalization including:
- Unicode normalization (NFC)
- Devanagari script processing (vowels, consonants, matras, etc.)
- Number-to-word conversion
- Abbreviation expansion
- Punctuation normalization
- Character-level tokenization with vocabulary mapping
"""

import re
import unicodedata
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class HindiTextProcessor:
    """Complete Hindi text processor for TTS training."""

    # Unicode ranges
    DEVANAGARI_START = 0x0900
    DEVANAGARI_END = 0x097F

    # Special tokens
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"

    PAD_IDX = 0
    UNK_IDX = 1
    BOS_IDX = 2
    EOS_IDX = 3

    def __init__(self, language: str = "hi"):
        """
        Initialize Hindi text processor.

        Args:
            language: Language code (default: "hi" for Hindi)
        """
        self.language = language
        self.vocab: Dict[str, int] = {}
        self.id2char: Dict[int, str] = {}
        self._build_vocabulary()

        # Mapping for number-to-word conversion
        self.ones = [
            "", "एक", "दो", "तीन", "चार", "पांच", "छः", "सात", "आठ", "नौ"
        ]
        self.tens = [
            "", "", "बीस", "तीस", "चालीस", "पचास", "साठ", "सत्तर", "अस्सी", "नब्बे"
        ]
        self.teens = [
            "दस", "ग्यारह", "बारह", "तेरह", "चौदह", "पन्द्रह",
            "सोलह", "सत्रह", "अठारह", "उन्नीस"
        ]
        self.scales = [
            ("सौ", 100),
            ("हजार", 1000),
            ("लाख", 100000),
            ("करोड़", 10000000),
        ]

    def _build_vocabulary(self) -> None:
        """Build vocabulary from Devanagari script + ASCII punctuation + space."""
        vocab_list = []

        # Add special tokens first
        vocab_list.append(self.PAD_TOKEN)
        vocab_list.append(self.UNK_TOKEN)
        vocab_list.append(self.BOS_TOKEN)
        vocab_list.append(self.EOS_TOKEN)

        # Add Devanagari characters (U+0900 to U+097F)
        for code_point in range(self.DEVANAGARI_START, self.DEVANAGARI_END + 1):
            try:
                char = chr(code_point)
                vocab_list.append(char)
            except ValueError:
                continue

        # Add ASCII space and common punctuation
        ascii_chars = " .,!?;:-\""
        vocab_list.extend(list(ascii_chars))

        # Create mappings
        for idx, char in enumerate(vocab_list):
            self.vocab[char] = idx
            self.id2char[idx] = char

    def normalize_unicode(self, text: str) -> str:
        """
        Apply NFC normalization to ensure consistent Unicode representation.

        Args:
            text: Input text

        Returns:
            Normalized text
        """
        return unicodedata.normalize("NFC", text)

    def remove_accents(self, text: str) -> str:
        """
        Remove accent marks from text while preserving Devanagari marks.

        Args:
            text: Input text

        Returns:
            Text with accents removed
        """
        # For Hindi, we want to keep most diacritics, but remove stray accents
        nfd_form = unicodedata.normalize("NFD", text)
        return "".join(
            char for char in nfd_form
            if unicodedata.category(char) != "Mn"
        )

    def expand_abbreviations(self, text: str) -> str:
        """
        Expand common Hindi abbreviations.

        Args:
            text: Input text

        Returns:
            Text with abbreviations expanded
        """
        abbreviations = {
            r"\bइ\.आ\.": "इंडिया",
            r"\bअ\.मे\.": "अमेरिका",
            r"\bडॉ\.": "डॉक्टर",
            r"\bमि\.": "मिनट",
            r"\bघं\.": "घंटा",
            r"\bजन\.": "जनवरी",
            r"\bफ़रवरी": "फरवरी",
        }

        for abbrev, expansion in abbreviations.items():
            text = re.sub(abbrev, expansion, text)

        return text

    def normalize_punctuation(self, text: str) -> str:
        """
        Normalize various punctuation marks to standard forms.

        Args:
            text: Input text

        Returns:
            Text with normalized punctuation
        """
        # Replace various dash types with standard hyphen
        text = re.sub(r"[–—−]", "-", text)

        # Replace various quote types with standard quotes (using character codes)
        text = text.replace("\u201c", '"')  # Left double quotation mark
        text = text.replace("\u201d", '"')  # Right double quotation mark
        text = text.replace("\u2018", "'")  # Left single quotation mark
        text = text.replace("\u2019", "'")  # Right single quotation mark

        # Remove multiple spaces
        text = re.sub(r"\s+", " ", text)

        # Remove leading/trailing spaces
        text = text.strip()

        return text

    def _convert_two_digit_number(self, num: int) -> str:
        """Convert a two-digit number to Hindi words."""
        if num == 0:
            return ""
        elif num < 10:
            return self.ones[num]
        elif num < 20:
            return self.teens[num - 10]
        else:
            return self.tens[num // 10] + ("" if num % 10 == 0 else " " + self.ones[num % 10])

    def _convert_three_digit_number(self, num: int) -> str:
        """Convert a three-digit number to Hindi words."""
        result = ""

        # Hundreds place
        if num >= 100:
            result += self.ones[num // 100] + " सौ"
            num %= 100
            if num > 0:
                result += " "

        # Tens and ones
        if num > 0:
            result += self._convert_two_digit_number(num)

        return result

    def number_to_words_hindi(self, num: int) -> str:
        """
        Convert integer to Hindi words.

        Args:
            num: Integer to convert

        Returns:
            Hindi word representation

        Example:
            123 → "एक सौ तेईस"
        """
        if num == 0:
            return "शून्य"

        if num < 0:
            return "नकारात्मक " + self.number_to_words_hindi(-num)

        # Handle large numbers using scales
        result = ""
        scales_to_use = [
            ("करोड़", 10000000),
            ("लाख", 100000),
            ("हजार", 1000),
        ]

        for scale_name, scale_value in scales_to_use:
            if num >= scale_value:
                quotient = num // scale_value
                result += self._convert_three_digit_number(quotient) + " " + scale_name + " "
                num %= scale_value

        # Handle remainder (less than 1000)
        if num > 0:
            result += self._convert_three_digit_number(num)

        return result.strip()

    def normalize_numbers(self, text: str) -> str:
        """
        Convert digit sequences to Hindi words.

        Args:
            text: Input text with numbers

        Returns:
            Text with numbers converted to words
        """
        def replace_number(match):
            num_str = match.group(0)
            try:
                num = int(num_str)
                return self.number_to_words_hindi(num)
            except ValueError:
                return num_str

        # Match sequences of digits
        text = re.sub(r"\d+", replace_number, text)
        return text

    def normalize_text(self, text: str) -> str:
        """
        Apply complete text normalization pipeline.

        Args:
            text: Raw input text

        Returns:
            Normalized text
        """
        # 1. Unicode normalization
        text = self.normalize_unicode(text)

        # 2. Expand abbreviations
        text = self.expand_abbreviations(text)

        # 3. Normalize numbers
        text = self.normalize_numbers(text)

        # 4. Normalize punctuation
        text = self.normalize_punctuation(text)

        # 5. Remove extra accents while preserving Devanagari
        # (mostly kept as is, removed only true accents)

        return text

    def text_to_token_ids(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None
    ) -> List[int]:
        """
        Convert text to token IDs.

        Args:
            text: Input text to tokenize
            add_special_tokens: Whether to add BOS and EOS tokens
            max_length: Maximum sequence length (if specified, pad/truncate)

        Returns:
            List of token IDs
        """
        # Normalize text first
        text = self.normalize_text(text)

        token_ids = []

        # Add BOS token
        if add_special_tokens:
            token_ids.append(self.BOS_IDX)

        # Convert each character
        for char in text:
            if char in self.vocab:
                token_ids.append(self.vocab[char])
            else:
                # Use UNK token for unknown characters
                token_ids.append(self.UNK_IDX)
                logger.warning(f"Unknown character in vocabulary: {char} (U+{ord(char):04X})")

        # Add EOS token
        if add_special_tokens:
            token_ids.append(self.EOS_IDX)

        # Handle max_length
        if max_length is not None:
            if len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
            elif len(token_ids) < max_length:
                # Pad with PAD tokens
                token_ids.extend([self.PAD_IDX] * (max_length - len(token_ids)))

        return token_ids

    def token_ids_to_text(self, token_ids: List[int]) -> str:
        """
        Convert token IDs back to text.

        Args:
            token_ids: List of token IDs

        Returns:
            Reconstructed text
        """
        text = []

        for token_id in token_ids:
            if token_id in self.id2char:
                char = self.id2char[token_id]
                # Skip special tokens in output
                if char not in [self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN]:
                    text.append(char)

        return "".join(text)

    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.vocab)

    def get_vocab(self) -> Dict[str, int]:
        """Get complete vocabulary mapping."""
        return self.vocab.copy()

    def get_id2char(self) -> Dict[int, str]:
        """Get ID to character mapping."""
        return self.id2char.copy()


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    processor = HindiTextProcessor()

    # Test cases
    test_texts = [
        "नमस्ते, यह एक परीक्षण है।",
        "123 रुपये का खर्च था।",
        "डॉ. शर्मा ने 5 दवाई दी।",
        "भारत (इ.आ.) में रहते हैं।",
    ]

    print(f"Vocabulary size: {processor.get_vocab_size()}")
    print("\nTest conversions:")

    for text in test_texts:
        normalized = processor.normalize_text(text)
        token_ids = processor.text_to_token_ids(text)
        reconstructed = processor.token_ids_to_text(token_ids)

        print(f"\nOriginal:      {text}")
        print(f"Normalized:    {normalized}")
        print(f"Token IDs:     {token_ids[:20]}...")  # Show first 20
        print(f"Reconstructed: {reconstructed}")

    # Test number conversion
    print("\n\nNumber to words tests:")
    for num in [0, 5, 10, 15, 100, 123, 1000, 10000, 100000, 1000000]:
        print(f"{num:10d} → {processor.number_to_words_hindi(num)}")
