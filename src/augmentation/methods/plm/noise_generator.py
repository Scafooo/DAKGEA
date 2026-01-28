"""
NoiseGenerator per Data Augmentation Semantica.
Gestisce diverse strategie di corruzione del testo per addestrare modelli di Denoising.
"""

import random
import string
import re
from typing import Optional

class NoiseGenerator:
    def __init__(
        self,
        char_prob: float = 0.1,
        word_prob: float = 0.05,
        digit_prob: float = 0.1,
        seed: Optional[int] = None
    ):
        self.char_prob = char_prob
        self.word_prob = word_prob
        self.digit_prob = digit_prob
        if seed is not None:
            random.seed(seed)

    def apply(self, text: str, predicate: Optional[str] = None) -> str:
        """
        Applica il rumore al testo in modo intelligente basandosi sul predicato o sul contenuto.
        """
        if not text or len(text) < 2:
            return text

        # 1. Se sembra un anno o una data corta (es. 1980)
        if re.fullmatch(r'\d{4}', text):
            return self._numeric_noise(text, intensity=1)
        
        # 2. Se è un valore puramente numerico o ID strutturato (es. p7789)
        if any(c.isdigit() for c in text) and len(text) < 10:
            return self._id_noise(text)

        # 3. Se è testo lungo (> 5 parole)
        if len(text.split()) > 5:
            return self._mixed_noise(text)

        # 4. Default: Character-level noise
        return self._character_noise(text)

    def _character_noise(self, text: str) -> str:
        """Typo a livello di carattere."""
        chars = list(text)
        n_noise = max(1, int(len(chars) * self.char_prob))
        
        for _ in range(n_noise):
            op = random.choice(['del', 'swap', 'sub', 'ins'])
            idx = random.randint(0, len(chars) - 1)
            
            if op == 'del' and len(chars) > 2:
                del chars[idx]
            elif op == 'swap' and idx < len(chars) - 1:
                chars[idx], chars[idx+1] = chars[idx+1], chars[idx]
            elif op == 'sub':
                chars[idx] = random.choice(string.ascii_letters + string.digits)
            elif op == 'ins':
                chars.insert(idx, random.choice(string.ascii_letters))
        
        return "".join(chars)

    def _word_noise(self, text: str) -> str:
        """Corruzione a livello di parola (drop, repeat)."""
        words = text.split()
        if len(words) < 2:
            return self._character_noise(text)
            
        n_noise = max(1, int(len(words) * self.word_prob))
        for _ in range(n_noise):
            op = random.choice(['drop', 'repeat', 'swap'])
            idx = random.randint(0, len(words) - 1)
            
            if op == 'drop' and len(words) > 2:
                del words[idx]
            elif op == 'repeat':
                words.insert(idx, words[idx])
            elif op == 'swap' and idx < len(words) - 1:
                words[idx], words[idx+1] = words[idx+1], words[idx]
                
        return " ".join(words)

    def _numeric_noise(self, text: str, intensity: int = 1) -> str:
        """Cambia leggermente un numero (es. 1980 -> 1981)."""
        if not text.isdigit():
            return self._character_noise(text)
        
        val = int(text)
        delta = random.randint(-intensity, intensity)
        if delta == 0: delta = 1
        return str(val + delta)

    def _id_noise(self, text: str) -> str:
        """Rumore per ID: cambia una cifra o una lettera mantenendo la lunghezza."""
        chars = list(text)
        idx = random.randint(0, len(chars) - 1)
        if chars[idx].isdigit():
            chars[idx] = random.choice(string.digits)
        else:
            chars[idx] = random.choice(string.ascii_lowercase)
        return "".join(chars)

    def _mixed_noise(self, text: str) -> str:
        """Applica sia word che char noise."""
        text = self._word_noise(text)
        return self._character_noise(text)
