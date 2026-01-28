"""
NoiseGenerator stabile per Denoising.
Genera versioni 'sporche' o 'abbreviate' dell'input per addestrare il modello a ricostruire l'originale.
"""

import random
import string
import re
from typing import Optional, List

class NoiseGenerator:
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)

    def apply(self, text: str, predicate: Optional[str] = None) -> str:
        """Produce una versione 'rumorosa' dell'input. Il target del training sarà il testo originale."""
        if not text or len(text) < 2:
            return text

        p_name = (predicate or "").lower()
        words = text.split()

        # 1. DATE: Cambia un numero (Denoising: sposta 1980 -> 1981, deve tornare a 1980)
        if any(x in p_name for x in ["date", "born", "died", "year"]) or re.search(r'\d{4}', text):
            return self._date_noise(text)

        # 2. NOMI/TESTO BREVE: Abbreviazione o Swap (Denoising: J. Smith -> John Smith)
        if not any(c.isdigit() for c in text) and 2 <= len(words) <= 3:
            return self._name_noise(text)

        # 3. ID / CODICI: Typo di carattere (Denoising: p7789 -> p7729)
        if (any(c.isdigit() for c in text) and " " not in text) or "id" in p_name:
            return self._id_noise(text)

        # 4. TESTO LUNGO: Drop di parole
        if len(words) > 5:
            return self._long_text_noise(text)

        return self._char_corruption(text)

    def _name_noise(self, text: str) -> str:
        parts = text.split()
        op = random.choice(['abbr', 'swap', 'typo'])
        if op == 'abbr':
            # Abbreviazione: 'John Smith' -> 'J. Smith'
            idx = random.randint(0, len(parts) - 1)
            parts[idx] = f"{parts[idx][0].upper()}."
            return " ".join(parts)
        if op == 'swap' and len(parts) == 2:
            # Swap: 'John Smith' -> 'Smith John'
            return f"{parts[1]} {parts[0]}"
        return self._char_corruption(text)

    def _date_noise(self, text: str) -> str:
        nums = re.findall(r'\d+', text)
        if not nums: return self._char_corruption(text)
        target = random.choice(nums)
        new_val = str(int(target) + random.choice([-1, 1]))
        return text.replace(target, new_val, 1)

    def _id_noise(self, text: str) -> str:
        return self._char_corruption(text, max_mods=1)

    def _long_text_noise(self, text: str) -> str:
        words = text.split()
        if len(words) > 4:
            del words[random.randint(0, len(words) - 1)]
        return " ".join(words)

    def _char_corruption(self, text: str, max_mods: int = 2) -> str:
        chars = list(text)
        n_mods = min(max_mods, max(1, int(len(chars) * 0.1)))
        for _ in range(n_mods):
            idx = random.randint(0, len(chars) - 1)
            op = random.choice(['del', 'swap', 'sub'])
            if op == 'del' and len(chars) > 3: del chars[idx]
            elif op == 'swap' and idx < len(chars) - 1: chars[idx], chars[idx+1] = chars[idx+1], chars[idx]
            elif op == 'sub': chars[idx] = random.choice(string.ascii_lowercase)
        return "".join(chars)
