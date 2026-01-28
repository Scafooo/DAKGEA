"""
VariationGenerator bilanciato per Data Augmentation.
- Nomi: Genera variazioni coerenti (Abbreviazioni, Titoli, Middle Initials, Suffissi).
- ID/Date: Denoising strutturale per preservare il formato.
"""

import random
import string
import re
from typing import Optional, List

class VariationGenerator:
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
        
        # Risorse per nomi coerenti
        self.name_suffixes = ["son", "y", "er", "field", "well", "ton"]
        self.titles = ["Mr.", "Sir", "Dr.", "Prof.", "Hon."]
        self.common_nicknames = {
            "robert": ["Bob", "Bobby", "Rob"],
            "william": ["Bill", "Billy", "Will"],
            "richard": ["Dick", "Rick"],
            "james": ["Jim", "Jimmy"],
            "john": ["Jack", "Johnny"]
        }

    def generate(self, text: str, predicate: Optional[str] = None) -> str:
        if not text or len(text) < 2:
            return text

        p_name = (predicate or "").lower()
        words = text.split()

        # 1. LOGICA NOMI - VARIAZIONE COERENTE
        if not any(c.isdigit() for c in text) and 2 <= len(words) <= 4:
            return self._name_variation(text)

        # 2. LOGICA DATE - DENOISING
        if any(x in p_name for x in ["date", "born", "died", "year"]) or re.search(r'\d{4}', text):
            return self._date_denoising(text)

        # 3. LOGICA ID - DENOISING
        if (any(c.isdigit() for c in text) and " " not in text) or "id" in p_name:
            return self._id_denoising(text)

        return self._char_corruption(text)

    def _name_variation(self, text: str) -> str:
        parts = text.split()
        op = random.choice(['abbr', 'title', 'middle_init', 'nickname', 'swap', 'suffix'])

        if op == 'abbr' and len(parts) >= 2:
            # John Smith -> J. Smith
            idx = random.randint(0, len(parts) - 1)
            parts[idx] = f"{parts[idx][0].upper()}."
            return " ".join(parts)

        if op == 'title':
            # Smith -> Mr. Smith
            return f"{random.choice(self.titles)} {text}"

        if op == 'middle_init' and len(parts) == 2:
            # John Smith -> John M. Smith
            mid = random.choice(string.ascii_uppercase) + "."
            return f"{parts[0]} {mid} {parts[1]}"

        if op == 'nickname':
            # Robert Smith -> Bob Smith
            for i, p in enumerate(parts):
                p_low = p.lower()
                if p_low in self.common_nicknames:
                    parts[i] = random.choice(self.common_nicknames[p_low])
                    return " ".join(parts)
            return self._abbreviate_simple(text)

        if op == 'swap' and len(parts) == 2:
            # John Smith -> Smith, John
            return f"{parts[1]}, {parts[0]}"

        if op == 'suffix' and len(parts[0]) > 3:
            # Smith -> Smithson
            target = random.choice(parts)
            if len(target) > 3:
                return text.replace(target, target + random.choice(self.name_suffixes))

        return self._char_corruption(text)

    def _abbreviate_simple(self, text: str) -> str:
        parts = text.split()
        parts[0] = f"{parts[0][0].upper()}."
        return " ".join(parts)

    def _date_denoising(self, text: str) -> str:
        nums = re.findall(r'\d+', text)
        if not nums: return self._char_corruption(text)
        target = random.choice(nums)
        new_val = str(int(target) + random.choice([-1, 1]))
        return text.replace(target, new_val, 1)

    def _id_denoising(self, text: str) -> str:
        chars = list(text)
        idx = random.randint(0, len(chars) - 1)
        if chars[idx].isdigit(): chars[idx] = random.choice(string.digits)
        else: chars[idx] = random.choice(string.ascii_lowercase)
        return "".join(chars)

    def _char_corruption(self, text: str) -> str:
        chars = list(text)
        if len(chars) < 3: return text
        idx = random.randint(0, len(chars) - 1)
        op = random.choice(['del', 'swap', 'sub'])
        if op == 'del' and len(chars) > 3: del chars[idx]
        elif op == 'swap' and idx < len(chars) - 1: chars[idx], chars[idx+1] = chars[idx+1], chars[idx]
        elif op == 'sub': chars[idx] = random.choice(string.ascii_lowercase)
        return "".join(chars)