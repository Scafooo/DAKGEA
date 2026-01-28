"""
NoiseGenerator avanzato per Data Augmentation.
Gestisce strategie specifiche per tipologia di dato con intensità controllata.
"""

import random
import string
import re
from typing import Optional, List

class NoiseGenerator:
    def __init__(
        self,
        char_prob: float = 0.1,
        word_drop_prob: float = 0.15,
        max_mods_short: int = 2,  # Massimo 2 modifiche per nomi/ID
        max_mods_long: int = 5,   # Fino a 5 modifiche per testi lunghi
        seed: Optional[int] = None
    ):
        self.char_prob = char_prob
        self.word_drop_prob = word_drop_prob
        self.max_mods_short = max_mods_short
        self.max_mods_long = max_mods_long
        if seed is not None:
            random.seed(seed)

    def apply(self, text: str, predicate: Optional[str] = None) -> str:
        if not text or len(text) < 2:
            return text

        p_name = (predicate or "").lower()
        words = text.split()

        # 1. DATE (Massimo 1 modifica per preservare l'ancoraggio temporale)
        if any(x in p_name for x in ["date", "born", "died", "year"]) or re.search(r'\d{4}', text):
            return self._date_noise(text)

        # 2. IDENTIFICATORI (Massimo 1-2 modifiche di cifre/caratteri)
        if (any(c.isdigit() for c in text) and " " not in text) or "id" in p_name:
            return self._id_noise(text, max_mods=self.max_mods_short)

        # 3. TESTO LUNGO (> 5 parole) - Mix di Word Drop e Caratteri
        if len(words) > 5:
            return self._long_text_noise(text)

        # 4. TESTO BREVE (Nomi, Label) - Abbreviazione O 1-2 Typo
        return self._short_text_noise(text)

    def _short_text_noise(self, text: str) -> str:
        # 40% Abbreviazione (se possibile), altrimenti 1-2 Typo
        if len(text.split()) >= 2 and random.random() < 0.4:
            return self._abbreviate(text)
        return self._character_corruption(text, max_mods=self.max_mods_short)

    def _abbreviate(self, text: str) -> str:
        parts = text.split()
        if not parts: return text
        idx = random.randint(0, len(parts) - 1)
        if len(parts[idx]) > 1:
            parts[idx] = f"{parts[idx][0].upper()}."
        return " ".join(parts)

    def _character_corruption(self, text: str, max_mods: int) -> str:
        chars = list(text)
        n_mods = min(max_mods, max(1, int(len(chars) * self.char_prob)))
        
        for _ in range(n_mods):
            op = random.choice(['del', 'swap', 'sub_char', 'sub_digit'])
            idx = random.randint(0, len(chars) - 1)
            if op == 'del' and len(chars) > 3: del chars[idx]
            elif op == 'swap' and idx < len(chars) - 1: chars[idx], chars[idx+1] = chars[idx+1], chars[idx]
            elif op == 'sub_char' and chars[idx].isalpha(): chars[idx] = random.choice(string.ascii_lowercase)
            elif op == 'sub_digit' and chars[idx].isdigit(): chars[idx] = random.choice(string.digits)
        return "".join(chars)

    def _long_text_noise(self, text: str) -> str:
        # 1. Rimuovi parole (Word Drop)
        words = text.split()
        n_to_drop = min(3, max(1, int(len(words) * self.word_drop_prob)))
        for _ in range(n_to_drop):
            if len(words) > 3:
                del words[random.randint(0, len(words) - 1)]
        
        # 2. Applica typo su 1-2 caratteri del risultato
        corrupted_text = " ".join(words)
        return self._character_corruption(corrupted_text, max_mods=2)

    def _date_noise(self, text: str) -> str:
        # Modifica solo un'istanza numerica (o anno o giorno)
        nums = re.findall(r'\d+', text)
        if not nums: return self._character_corruption(text, 1)
        
        target_num = random.choice(nums)
        if len(target_num) == 4: # Anno
            new_val = str(int(target_num) + random.choice([-1, 1, -2, 2]))
        else: # Giorno/Mese
            val = int(target_num)
            new_val = str(random.randint(1, 28)) if 1 <= val <= 31 else str(val + 1)
            
        return text.replace(target_num, new_val, 1)

    def _id_noise(self, text: str, max_mods: int) -> str:
        # Cambia esattamente 1 o 2 caratteri per ID
        return self._character_corruption(text, max_mods=max_mods)
