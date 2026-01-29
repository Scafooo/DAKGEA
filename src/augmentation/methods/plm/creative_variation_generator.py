"""
Creative Variation Generator per Training Mix-up.

NUOVO PARADIGMA: Input → Variazione Creativa
(Invece di Denoising: Corrupted → Original)

CLASSIFICAZIONE PER CONTENUTO (non predicato!):
1. NUMERICO/DATA/ID → pattern detection → trattamento speciale
2. TESTO CORTO (≤3 parole) → VARIAZIONI FORTI (nomi, titoli)
3. TESTO LUNGO (>3 parole) → riformulazione, shuffle

OSIAMO DI PIÙ: Lo scoring filtra il garbage!
"""

import random
import re
from typing import Optional, List, Tuple
from difflib import SequenceMatcher


# Soglia parole per distinguere corto/lungo
SHORT_TEXT_THRESHOLD = 3

# Vocali per validazione
VOWELS = set('aeiouAEIOU')

# Cluster di consonanti impossibili all'inizio
IMPOSSIBLE_STARTS = {'sn', 'sr', 'sb', 'sd', 'sf', 'sg', 'sv', 'sz',
                     'nm', 'ng', 'nr', 'nl', 'nb', 'nd',
                     'mr', 'ml', 'mn', 'mt', 'md',
                     'lr', 'lm', 'ln', 'lt', 'ld', 'lk',
                     'tn', 'tm', 'tk', 'tp', 'tb', 'td',
                     'pn', 'pm', 'pk', 'pb', 'pd',
                     'bn', 'bm', 'bk', 'bp', 'bd',
                     'dn', 'dm', 'dk', 'dp', 'db'}


def _is_plausible_name(text: str) -> bool:
    """
    Verifica se un testo "suona" come un nome plausibile.

    Regole:
    1. Deve avere almeno una vocale
    2. Non più di 3 consonanti consecutive
    3. Non iniziare con cluster impossibili
    """
    text_lower = text.lower().strip()

    if len(text_lower) < 2:
        return True  # Troppo corto per giudicare

    # 1. Almeno una vocale
    if not any(c in VOWELS for c in text_lower):
        return False

    # 2. Max 3 consonanti consecutive
    consonant_count = 0
    for c in text_lower:
        if c.isalpha() and c not in VOWELS:
            consonant_count += 1
            if consonant_count > 3:
                return False
        else:
            consonant_count = 0

    # 3. Check inizio
    if len(text_lower) >= 2:
        start = text_lower[:2]
        if start in IMPOSSIBLE_STARTS:
            return False

    return True


class CreativeVariationGenerator:
    """Genera variazioni creative per il training."""

    # Suffissi comuni per nomi
    NAME_SUFFIXES = ["son", "sen", "s", "ez", "ski", "ov", "ini"]

    # FILLER WORDS da NON usare nel training (causano pattern "the")
    FILLER_WORDS = {'the', 'a', 'an', 'and', 'or', 'of', 'to', 'in', 'on', 'at', 'by', 'for'}

    # Mappatura vocali per variazioni ortografiche
    VOWEL_VARIANTS = {
        'a': ['a', 'ae', 'ah'],
        'e': ['e', 'ee', 'ea'],
        'i': ['i', 'y', 'ie'],
        'o': ['o', 'oh', 'ou'],
        'u': ['u', 'oo', 'ou'],
    }

    def __init__(
        self,
        abbrev_prob: float = 0.20,
        ortho_prob: float = 0.20,
        mix_prob: float = 0.20,
        swap_prob: float = 0.15,
        combo_prob: float = 0.25,  # NUOVO: combinazione di strategie
    ):
        """
        Args:
            abbrev_prob: Probabilità di abbreviazione
            ortho_prob: Probabilità di variazione ortografica
            mix_prob: Probabilità di mix (richiede other_text)
            swap_prob: Probabilità di swap + variazione
            combo_prob: Probabilità di combo (2 strategie insieme)
        """
        self.abbrev_prob = abbrev_prob
        self.ortho_prob = ortho_prob
        self.mix_prob = mix_prob
        self.swap_prob = swap_prob
        self.combo_prob = combo_prob

    # Nomi dei mesi per detection date
    MONTH_NAMES = {
        'january', 'february', 'march', 'april', 'may', 'june',
        'july', 'august', 'september', 'october', 'november', 'december',
        'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'
    }

    # Valori booleani/costanti da NON variare
    INVARIANT_VALUES = {'yes', 'no', 'true', 'false', 'none', 'null', 'n/a', 'na', 'unknown'}

    def _detect_type(self, text: str) -> str:
        """
        Rileva il tipo basandosi SOLO sul contenuto (non predicato!).

        Returns: 'numeric', 'short', 'long', 'invariant'
        """
        text_clean = text.strip()
        text_lower = text_clean.lower()

        # INVARIANT: booleani, costanti - NON variare!
        if text_lower in self.INVARIANT_VALUES:
            return 'invariant'

        # 1. NUMERICO: date, numeri, ID, codici
        # Pattern data (2023-01-15, 15/01/2023, etc)
        if re.match(r'^\d{4}[-/]\d{2}[-/]\d{2}', text_clean) or \
           re.match(r'^\d{2}[-/]\d{2}[-/]\d{4}', text_clean):
            return 'numeric'

        # Solo numeri
        if re.match(r'^[\d\s\-/:.]+$', text_clean):
            return 'numeric'

        # DATE CON NOMI MESI: "2010 september", "january 2010", "15 march 2020"
        # Se contiene un mese E un anno (4 cifre), è una data!
        words_lower = text_lower.split()
        has_month = any(w in self.MONTH_NAMES for w in words_lower)
        has_year = any(re.match(r'^\d{4}$', w) for w in words_lower)
        if has_month and has_year:
            return 'numeric'

        # Codici alfanumerici: DEVE contenere numeri, non solo lettere!
        # "ABC123XYZ" → numeric, "supertramp" → short
        if re.match(r'^[A-Za-z0-9\-_]{5,}$', text_clean) and ' ' not in text_clean:
            # Solo se contiene ALMENO un numero
            if any(c.isdigit() for c in text_clean):
                return 'numeric'

        # URL, email
        if re.match(r'^https?://', text_clean) or '@' in text_clean:
            return 'numeric'

        # 2. BASATO SU LUNGHEZZA
        words = [w for w in text_clean.split() if len(w) > 0]

        if len(words) <= SHORT_TEXT_THRESHOLD:
            return 'short'  # Nomi, titoli → VARIAZIONI FORTI
        return 'long'  # Descrizioni → riformulazione

    def generate(self, text: str, other_text: Optional[str] = None, predicate: Optional[str] = None) -> str:
        """
        Genera una variazione creativa del testo.

        Classificazione per CONTENUTO (non predicato!):
        - numeric: date, numeri, ID → trattamento speciale
        - short (≤3 parole): nomi, titoli → VARIAZIONI FORTI
        - long (>3 parole): descrizioni → riformulazione

        Args:
            text: Testo originale
            other_text: Testo opzionale per mix
            predicate: Ignorato (kept for compatibility)

        Returns:
            Variazione creativa del testo
        """
        if not text or len(text.strip()) < 2:
            return text

        # Rileva tipo basato su CONTENUTO
        data_type = self._detect_type(text)

        # === BOOLEANI: a volte inverti! ===
        if data_type == 'invariant':
            text_lower = text.strip().lower()
            # 50% probabilità di inversione
            if random.random() < 0.5:
                bool_flip = {
                    'yes': 'no', 'no': 'yes',
                    'true': 'false', 'false': 'true',
                }
                if text_lower in bool_flip:
                    return bool_flip[text_lower]
            return text  # Altrimenti invariato

        # === NUMERICO: date, numeri, ID ===
        if data_type == 'numeric':
            return self._vary_numeric(text)

        # === TESTO LUNGO (>3 parole): riformulazione ===
        if data_type == 'long':
            return self._vary_long_text(text)

        # === TESTO CORTO (≤3 parole): VARIAZIONI FORTI! ===
        words = text.strip().split()
        if not words:
            return text

        # PAROLA SINGOLA LUNGA (>6 char): strategie conservative per evitare garbage
        if len(words) == 1 and len(words[0]) > 6:
            return self._vary_single_long_word(words[0])

        # MULTI-PAROLA o PAROLA CORTA: variazioni PIÙ AGGRESSIVE!
        r = random.random()

        # 30% COMBO (2-3 strategie insieme!) - PIÙ FREQUENTE
        if r < 0.30:
            return self._combo_variation_aggressive(words, other_text)

        # 25% CHAR SWAP + altra variazione
        if r < 0.55:
            # Char swap + 50% anche suffix/abbrev
            result = self._char_swap_word_list(words)
            if random.random() < 0.5 and len(words) >= 2:
                result_words = result.split()
                # Aggiungi suffix a una parola non swappata
                idx = random.randint(0, len(result_words) - 1)
                if len(result_words[idx]) >= 3:
                    result_words[idx] = result_words[idx] + random.choice(self.NAME_SUFFIXES)
                result = " ".join(result_words)
            return result

        # 15% MIX (se disponibile other_text)
        if r < 0.70 and other_text:
            return self._mix_texts_aggressive(words, other_text)

        # 15% ABBREVIAZIONE (solo se multi-parola)
        if r < 0.85 and len(words) >= 2:
            return self._abbreviate(words)

        # 15% ORTOGRAFICA FORTE
        return self._orthographic_variation_strong(words)

        return self._char_swap_word_list(words)  # Fallback: char swap

    def _vary_numeric(self, text: str) -> str:
        """
        Variazione FORTE per contenuti numerici!
        - Numeri: swap cifre, nuovi numeri (stessa lunghezza)
        - Date: cambio formato
        - Codici: shuffle caratteri
        """
        text = text.strip()

        # Pattern data YYYY-MM-DD → cambio formato
        match = re.match(r'^(\d{4})[-/](\d{2})[-/](\d{2})', text)
        if match:
            y, m, d = match.groups()
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            m_idx = int(m) - 1
            if 0 <= m_idx < 12:
                formats = [
                    f"{int(d)} {months[m_idx]} {y}",
                    f"{d}/{m}/{y}",
                    f"{m}/{d}/{y}",
                    f"{y}{m}{d}",  # Compatto
                ]
                return random.choice(formats)

        # DATE CON NOMI MESI: "2010 september", "january 2010", "15 march 2020"
        text_lower = text.lower()
        words = text.split()
        month_found = None
        year_found = None
        day_found = None

        month_map = {
            'january': ('Jan', '01'), 'february': ('Feb', '02'), 'march': ('Mar', '03'),
            'april': ('Apr', '04'), 'may': ('May', '05'), 'june': ('Jun', '06'),
            'july': ('Jul', '07'), 'august': ('Aug', '08'), 'september': ('Sep', '09'),
            'october': ('Oct', '10'), 'november': ('Nov', '11'), 'december': ('Dec', '12'),
            'jan': ('Jan', '01'), 'feb': ('Feb', '02'), 'mar': ('Mar', '03'),
            'apr': ('Apr', '04'), 'jun': ('Jun', '06'), 'jul': ('Jul', '07'),
            'aug': ('Aug', '08'), 'sep': ('Sep', '09'), 'oct': ('Oct', '10'),
            'nov': ('Nov', '11'), 'dec': ('Dec', '12')
        }

        for w in words:
            w_lower = w.lower()
            if w_lower in month_map:
                month_found = month_map[w_lower]
            elif re.match(r'^\d{4}$', w):
                year_found = w
            elif re.match(r'^\d{1,2}$', w):
                day_found = w

        if month_found and year_found:
            short_month, num_month = month_found
            all_months = [
                ('Jan', '01'), ('Feb', '02'), ('Mar', '03'), ('Apr', '04'),
                ('May', '05'), ('Jun', '06'), ('Jul', '07'), ('Aug', '08'),
                ('Sep', '09'), ('Oct', '10'), ('Nov', '11'), ('Dec', '12')
            ]

            # COMBINAZIONE DI VARIAZIONI: mese + anno + giorno + formato
            # 70% cambia mese
            if random.random() < 0.7:
                other_months = [(s, n) for s, n in all_months if n != num_month]
                short_month, num_month = random.choice(other_months)

            # 50% shift anno (±1-3 anni)
            year_int = int(year_found)
            if random.random() < 0.5:
                year_int += random.choice([-3, -2, -1, 1, 2, 3])
            year_str = str(year_int)

            # Genera giorno (se non presente, a volte aggiungilo)
            if day_found:
                new_day = str(max(1, min(28, int(day_found) + random.randint(-10, 10))))
            elif random.random() < 0.3:
                new_day = str(random.randint(1, 28))
            else:
                new_day = None

            # Genera variazioni di formato
            if new_day:
                formats = [
                    f"{new_day} {short_month} {year_str}",      # "15 Sep 2010"
                    f"{short_month} {new_day}, {year_str}",     # "Sep 15, 2010"
                    f"{num_month}/{new_day}/{year_str}",        # "09/15/2010"
                    f"{new_day}/{num_month}/{year_str}",        # "15/09/2010"
                    f"{year_str}-{num_month}-{new_day.zfill(2)}",  # "2010-09-15"
                    f"{year_str}{num_month}{new_day.zfill(2)}",    # "20100915"
                ]
            else:
                formats = [
                    f"{short_month} {year_str}",           # "Sep 2010"
                    f"{num_month}/{year_str}",             # "09/2010"
                    f"{year_str}-{num_month}",             # "2010-09"
                    f"{year_str} {short_month}",           # "2010 Sep"
                    f"{year_str}/{num_month}",             # "2010/09"
                ]
            return random.choice(formats)

        # Pattern solo anno → variazione +/- qualche anno
        if re.match(r'^\d{4}$', text):
            year = int(text)
            delta = random.choice([-2, -1, 1, 2])
            return str(year + delta)

        # NUMERI PURI → variazione FORTE!
        if re.match(r'^\d+$', text):
            strategy = random.choice(['shuffle', 'new', 'reverse', 'partial'])

            if strategy == 'shuffle':
                # Shuffle delle cifre
                digits = list(text)
                random.shuffle(digits)
                return ''.join(digits)

            elif strategy == 'new':
                # Nuovi numeri casuali, stessa lunghezza
                return ''.join(random.choices('0123456789', k=len(text)))

            elif strategy == 'reverse':
                # Reverse
                return text[::-1]

            else:  # partial
                # Cambia alcune cifre
                digits = list(text)
                n_change = max(1, len(digits) // 3)
                for _ in range(n_change):
                    idx = random.randint(0, len(digits) - 1)
                    digits[idx] = random.choice('0123456789')
                return ''.join(digits)

        # CODICI ALFANUMERICI → shuffle o partial change
        if re.match(r'^[A-Za-z0-9\-_]+$', text) and ' ' not in text:
            chars = list(text)
            if random.random() < 0.5:
                # Shuffle
                random.shuffle(chars)
            else:
                # Cambia alcuni caratteri
                n_change = max(1, len(chars) // 4)
                for _ in range(n_change):
                    idx = random.randint(0, len(chars) - 1)
                    if chars[idx].isdigit():
                        chars[idx] = random.choice('0123456789')
                    elif chars[idx].isupper():
                        chars[idx] = random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
                    elif chars[idx].islower():
                        chars[idx] = random.choice('abcdefghijklmnopqrstuvwxyz')
            return ''.join(chars)

        # URL/email: invariato (troppo rischioso)
        if 'http' in text.lower() or '@' in text:
            return text

        return text

    def _vary_single_long_word(self, word: str) -> str:
        """
        Variazione SICURA per parole singole lunghe (>6 caratteri).
        NO shuffle completo che produce garbage!

        "supertramp" → "Supertrampson", "SuperTramp", "supertramps", "supertarmp"
        """
        strategy = random.choice([
            'suffix', 'prefix', 'camel', 'typo_safe', 'double_letter', 'char_swap'
        ])

        if strategy == 'suffix':
            return word + random.choice(self.NAME_SUFFIXES)

        elif strategy == 'prefix':
            prefixes = ['The ', 'Super', 'New', 'Neo', 'Ultra']
            return random.choice(prefixes) + word

        elif strategy == 'camel':
            # CamelCase: supertramp → SuperTramp
            mid = len(word) // 2
            return word[:mid].title() + word[mid:].title()

        elif strategy == 'typo_safe':
            # Singolo typo sicuro (non distruttivo)
            chars = list(word)
            # Cambia UNA vocale o raddoppia UNA consonante
            vowel_idx = [i for i, c in enumerate(chars) if c.lower() in VOWELS]
            if vowel_idx and random.random() < 0.5:
                idx = random.choice(vowel_idx)
                # Cambia vocale
                vowels = 'aeiou'
                current = chars[idx].lower()
                new_vowel = random.choice([v for v in vowels if v != current])
                chars[idx] = new_vowel if chars[idx].islower() else new_vowel.upper()
            else:
                # Raddoppia una lettera
                idx = random.randint(1, len(chars) - 2)
                chars.insert(idx, chars[idx])
            return ''.join(chars)

        elif strategy == 'double_letter':
            idx = random.randint(1, len(word) - 2)
            return word[:idx] + word[idx] + word[idx:]

        else:  # char_swap
            return self._char_swap(word)

    def _char_swap(self, word: str) -> str:
        """
        Swap di caratteri adiacenti (typo-like).

        "John" → "Jonh", "Jhon"
        "Smith" → "Smtih", "Simth"

        Produce variazioni naturali tipo errori di battitura.
        """
        if len(word) < 3:
            return word

        chars = list(word)

        # Scegli una posizione per lo swap (non primo/ultimo carattere)
        # Evita di swappare vocali consecutive (suonerebbe male)
        valid_positions = []
        for i in range(1, len(chars) - 1):
            # Swap chars[i] con chars[i+1]
            c1, c2 = chars[i].lower(), chars[i+1].lower() if i+1 < len(chars) else ''
            # Evita swap vocale-vocale
            if not (c1 in VOWELS and c2 in VOWELS):
                valid_positions.append(i)

        if not valid_positions:
            # Fallback: qualsiasi posizione centrale
            valid_positions = list(range(1, len(chars) - 1))

        if valid_positions:
            idx = random.choice(valid_positions)
            if idx + 1 < len(chars):
                chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]

        result = ''.join(chars)

        # Valida che sia plausibile
        if _is_plausible_name(result):
            return result
        return word  # Fallback all'originale se garbage

    def _char_swap_word_list(self, words: List[str]) -> str:
        """
        Applica char swap a una o più parole della lista.

        "John Smith" → "Jonh Smith" o "John Smtih" o "Jonh Smtih"
        """
        if not words:
            return ""

        result = words.copy()

        # Scegli quante parole swappare (1 o 2)
        n_swap = random.randint(1, min(2, len(words)))
        indices = random.sample(range(len(words)), n_swap)

        for idx in indices:
            word = result[idx]
            if len(word) >= 3:
                swapped = self._char_swap(word)
                if swapped != word:  # Solo se effettivamente cambiato
                    result[idx] = swapped

        return " ".join(result)

    def _vary_long_text(self, text: str) -> str:
        """
        Variazione per testi lunghi: riordino, taglio, sinonimi base.
        """
        words = text.split()

        strategy = random.choice(['shuffle_part', 'truncate', 'reorder'])

        if strategy == 'shuffle_part':
            # Mescola una porzione
            start = random.randint(0, len(words) // 2)
            end = start + random.randint(2, 4)
            portion = words[start:end]
            random.shuffle(portion)
            words[start:end] = portion

        elif strategy == 'truncate':
            # Taglia (summarization)
            keep = int(len(words) * random.uniform(0.5, 0.8))
            words = words[:keep]

        elif strategy == 'reorder':
            # Sposta una frase
            if len(words) > 6:
                mid = len(words) // 2
                words = words[mid:] + words[:mid]

        return " ".join(words)

    def _abbreviate(self, words: List[str]) -> str:
        """
        Abbrevia una parola E modifica l'altra.

        "John Smith" → "Jon. Smtih" o "Joh. Smithson" o "Jo. Smiht"

        REGOLE:
        - Abbreviazioni di 2-3 caratteri (non singolo!)
        - Quando abbrevia una parola, l'altra viene modificata (char swap, suffix, etc.)
        """
        if len(words) < 1:
            return " ".join(words)

        result = words.copy()

        if len(words) == 1:
            # Parola singola: abbrevia a 2-3 caratteri
            word = words[0]
            if len(word) >= 3:
                abbrev_len = random.randint(2, min(3, len(word) - 1))
                result[0] = word[:abbrev_len].title() + "."
            return " ".join(result)

        # Multi-parola: abbrevia UNA parola, modifica l'ALTRA
        abbrev_idx = random.randint(0, len(words) - 1)
        other_idx = 1 - abbrev_idx if len(words) == 2 else random.choice([i for i in range(len(words)) if i != abbrev_idx])

        # Abbrevia (2-3 caratteri, non singolo!)
        word = result[abbrev_idx]
        if len(word) >= 3:
            abbrev_len = random.randint(2, min(3, len(word) - 1))
            result[abbrev_idx] = word[:abbrev_len].title() + "."
        elif len(word) >= 2:
            result[abbrev_idx] = word[:2].title() + "."

        # Modifica l'altra parola (char swap, suffix, double letter)
        other_word = result[other_idx]
        if len(other_word) >= 3:
            variation = random.choice(['char_swap', 'suffix', 'double', 'vowel'])

            if variation == 'char_swap':
                swapped = self._char_swap(other_word)
                if swapped != other_word:
                    result[other_idx] = swapped
            elif variation == 'suffix':
                result[other_idx] = other_word + random.choice(self.NAME_SUFFIXES)
            elif variation == 'double':
                idx = random.randint(1, len(other_word) - 2)
                result[other_idx] = other_word[:idx+1] + other_word[idx] + other_word[idx+1:]
            else:  # vowel change
                chars = list(other_word)
                vowel_idx = [i for i, c in enumerate(chars) if c.lower() in VOWELS]
                if vowel_idx:
                    i = random.choice(vowel_idx)
                    vowels = 'aeiou'
                    new_v = random.choice([v for v in vowels if v != chars[i].lower()])
                    chars[i] = new_v if chars[i].islower() else new_v.upper()
                    result[other_idx] = ''.join(chars)

        return " ".join(result)

    def _orthographic_variation(self, words: List[str]) -> str:
        """
        Crea variazione ortografica.
        "Smith" → "Smyth", "Smithson", "Smiths"
        """
        if not words:
            return ""

        result = words.copy()
        idx = random.randint(0, len(words) - 1)
        word = result[idx]

        if len(word) < 3:
            return " ".join(result)

        variation_type = random.choice(['suffix', 'vowel', 'double', 'case'])

        if variation_type == 'suffix':
            # Aggiungi suffisso
            suffix = random.choice(self.NAME_SUFFIXES)
            result[idx] = word + suffix

        elif variation_type == 'vowel':
            # Cambia una vocale
            for i, c in enumerate(word.lower()):
                if c in self.VOWEL_VARIANTS:
                    variants = [v for v in self.VOWEL_VARIANTS[c] if v != c]
                    if variants:
                        new_char = random.choice(variants)
                        # Preserva case
                        if word[i].isupper():
                            new_char = new_char.upper()
                        result[idx] = word[:i] + new_char + word[i+1:]
                        break

        elif variation_type == 'double':
            # Raddoppia una consonante
            consonants = [i for i, c in enumerate(word) if c.lower() not in 'aeiou' and c.isalpha()]
            if consonants:
                i = random.choice(consonants)
                result[idx] = word[:i+1] + word[i] + word[i+1:]

        elif variation_type == 'case':
            # Cambia case pattern
            result[idx] = word.title() if word.islower() else word.lower()

        return " ".join(result)

    def _mix_texts(self, words: List[str], other_text: str) -> str:
        """
        Crea mix tra due testi.
        "John Smith" + "Mary Johnson" → "J. Marith" o "Mohn Johnsmith"
        """
        other_words = other_text.strip().split()
        if not other_words:
            return " ".join(words)

        result = []
        max_len = max(len(words), len(other_words))

        for i in range(max_len):
            w1 = words[i] if i < len(words) else ""
            w2 = other_words[i] if i < len(other_words) else ""

            if not w1:
                result.append(w2)
            elif not w2:
                result.append(w1)
            else:
                # Mix delle due parole
                mix_type = random.choice(['initial', 'blend', 'choose'])

                if mix_type == 'initial':
                    # Iniziale di una + resto dell'altra
                    if random.random() < 0.5:
                        result.append(w1[0].upper() + ".")
                    else:
                        result.append(w1[0] + w2[1:] if len(w2) > 1 else w1)

                elif mix_type == 'blend':
                    # Blend: prima metà di una + seconda metà dell'altra
                    mid1, mid2 = len(w1) // 2, len(w2) // 2
                    blended = w1[:mid1] + w2[mid2:]
                    result.append(blended)

                else:
                    # Scegli una delle due
                    result.append(random.choice([w1, w2]))

        return " ".join(result)

    def _swap_and_vary(self, words: List[str]) -> str:
        """
        Swap ordine + variazione leggera.
        "John Smith" → "Smith J." o "Smithson John"
        """
        if len(words) < 2:
            return " ".join(words)

        result = words[::-1]  # Reverse

        # Aggiungi variazione a una parola
        if random.random() < 0.5:
            idx = random.randint(0, len(result) - 1)
            word = result[idx]
            if len(word) > 2:
                # Abbrevia o aggiungi suffisso
                if random.random() < 0.5:
                    result[idx] = word[0].upper() + "."
                else:
                    result[idx] = word + random.choice(self.NAME_SUFFIXES)

        return " ".join(result)

    def _light_variation(self, words: List[str]) -> str:
        """Variazione leggera: case o punteggiatura."""
        result = words.copy()
        if result:
            # Cambia case della prima parola
            result[0] = result[0].title()
        return " ".join(result)

    def _combo_variation(self, words: List[str], other_text: Optional[str] = None) -> str:
        """COMBO base: 2 strategie."""
        return self._combo_variation_aggressive(words, other_text)

    def _combo_variation_aggressive(self, words: List[str], other_text: Optional[str] = None) -> str:
        """
        COMBO MOLTO AGGRESSIVO: Applica 3-4 strategie insieme!

        Esempi:
        - "John Smith" → char_swap + suffix + vowel → "Jhon Smythson"
        - "Nina Simone" → char_swap + abbrev + double → "Nnia Sim."
        - "Mary Johnson" → vowel + suffix + swap → "Johnsonez Mery"
        """
        if not words:
            return ""

        result = words.copy()

        # SEMPRE 3-4 strategie per variazioni più creative!
        n_strategies = random.choice([3, 3, 4])
        strategies = ['char_swap', 'suffix', 'vowel_change', 'double_letter', 'abbrev', 'swap']

        selected = random.sample(strategies, min(n_strategies, len(strategies)))

        for strat in selected:
            if strat == 'char_swap':
                # Applica char swap a una parola random
                if result:
                    idx = random.randint(0, len(result) - 1)
                    if len(result[idx]) >= 3:
                        swapped = self._char_swap(result[idx])
                        if swapped != result[idx]:
                            result[idx] = swapped

            elif strat == 'suffix':
                # Aggiungi suffisso a una parola
                if result:
                    idx = random.randint(0, len(result) - 1)
                    if len(result[idx]) >= 2:
                        result[idx] = result[idx] + random.choice(self.NAME_SUFFIXES)

            elif strat == 'vowel_change':
                # Cambia una vocale in una parola
                if result:
                    idx = random.randint(0, len(result) - 1)
                    word = result[idx]
                    chars = list(word)
                    vowel_idx = [i for i, c in enumerate(chars) if c.lower() in VOWELS]
                    if vowel_idx:
                        i = random.choice(vowel_idx)
                        vowels = 'aeiou'
                        new_v = random.choice([v for v in vowels if v != chars[i].lower()])
                        chars[i] = new_v if chars[i].islower() else new_v.upper()
                        result[idx] = ''.join(chars)

            elif strat == 'double_letter':
                # Raddoppia una lettera
                if result:
                    idx = random.randint(0, len(result) - 1)
                    word = result[idx]
                    if len(word) >= 3:
                        pos = random.randint(1, len(word) - 2)
                        result[idx] = word[:pos+1] + word[pos] + word[pos+1:]

            elif strat == 'abbrev' and len(result) >= 2:
                # Abbrevia una parola (2-3 char)
                idx = random.randint(0, len(result) - 1)
                word = result[idx]
                if len(word) >= 3:
                    abbrev_len = random.randint(2, min(3, len(word) - 1))
                    result[idx] = word[:abbrev_len].title() + "."

            elif strat == 'swap' and len(result) >= 2:
                result = result[::-1]

        return " ".join(result)

    def _find_vowel_cut(self, word: str, prefer_after: bool = True) -> int:
        """Trova un punto di taglio naturale (dopo una vocale)."""
        vowel_positions = [i for i, c in enumerate(word) if c.lower() in VOWELS]

        if not vowel_positions:
            return len(word) // 2  # Fallback: metà

        if prefer_after:
            # Taglia DOPO una vocale (più naturale)
            # Preferisci posizioni nel mezzo
            mid = len(word) // 2
            best = min(vowel_positions, key=lambda x: abs(x - mid))
            return best + 1  # +1 per tagliare DOPO la vocale
        else:
            # Taglia PRIMA di una vocale
            mid = len(word) // 2
            best = min(vowel_positions, key=lambda x: abs(x - mid))
            return best

    def _mix_texts_aggressive(self, words: List[str], other_text: str) -> str:
        """
        MIX SMART: Blend che produce nomi PLAUSIBILI.

        "John Smith" + "Mary Johnson" → "Jary Smithson" o "Mohn Johnsmith"
        """
        other_words = other_text.strip().split()
        if not other_words:
            return " ".join(words)

        result = []
        max_len = max(len(words), len(other_words))

        for i in range(max_len):
            w1 = words[i] if i < len(words) else ""
            w2 = other_words[i] if i < len(other_words) else ""

            if not w1:
                result.append(w2)
            elif not w2:
                result.append(w1)
            else:
                # MIX SMART - con validazione
                mix_type = random.choice(['blend_smart', 'initial_suffix', 'swap_parts'])

                if mix_type == 'blend_smart':
                    # Blend intelligente: taglia dopo vocale
                    cut1 = self._find_vowel_cut(w1, prefer_after=True)
                    cut2 = self._find_vowel_cut(w2, prefer_after=False)

                    # Prova diverse combinazioni
                    candidates = [
                        w1[:cut1] + w2[cut2:],  # Prima parte w1 + seconda parte w2
                        w2[:cut2] + w1[cut1:],  # Prima parte w2 + seconda parte w1
                        w1[:2] + w2[2:],        # Iniziali w1 + resto w2
                        w2[:2] + w1[2:],        # Iniziali w2 + resto w1
                    ]

                    # Scegli il primo plausibile
                    for cand in candidates:
                        if _is_plausible_name(cand):
                            result.append(cand)
                            break
                    else:
                        # Nessuno plausibile, usa il più sicuro
                        result.append(w1[:2] + w2[2:] if len(w2) > 2 else w1)

                elif mix_type == 'initial_suffix':
                    # Iniziale + nome + suffisso
                    if random.random() < 0.5:
                        blended = w2 + random.choice(self.NAME_SUFFIXES)
                    else:
                        blended = w1 + random.choice(self.NAME_SUFFIXES)
                    result.append(blended)

                else:  # swap_parts
                    # Scambia parti: cognome1 + nome2
                    if i == 0:
                        result.append(w2)  # Usa il nome dell'altro
                    else:
                        result.append(w1 + w2[-3:] if len(w2) > 3 else w1)

        # Validazione finale
        final = " ".join(result)
        if all(_is_plausible_name(w) for w in final.split()):
            return final
        else:
            # Fallback sicuro
            return words[0] + other_words[-1] if other_words else " ".join(words)

    def _orthographic_variation_strong(self, words: List[str]) -> str:
        """
        Variazione ortografica FORTE.

        "Smith" → "Schmitt", "Smythe", "Smithsson"
        """
        if not words:
            return ""

        result = words.copy()
        # Varia più parole se possibile
        n_vary = min(len(words), random.randint(1, 2))
        indices = random.sample(range(len(words)), n_vary)

        for idx in indices:
            word = result[idx]
            if len(word) < 2:
                continue

            variation = random.choice([
                'double_suffix', 'prefix', 'vowel_shift', 'consonant_double', 'creative_spelling'
            ])

            if variation == 'double_suffix':
                # Doppio suffisso
                suffix1 = random.choice(self.NAME_SUFFIXES)
                suffix2 = random.choice(['', 'i', 'y', 'e'])
                result[idx] = word + suffix1 + suffix2

            elif variation == 'prefix':
                # Aggiungi prefisso
                prefixes = ['Mc', 'Mac', 'O\'', 'Van ', 'Von ', 'De ', 'La ']
                result[idx] = random.choice(prefixes) + word

            elif variation == 'vowel_shift':
                # Shift di tutte le vocali
                shifts = {'a': 'e', 'e': 'i', 'i': 'y', 'o': 'u', 'u': 'a',
                          'A': 'E', 'E': 'I', 'I': 'Y', 'O': 'U', 'U': 'A'}
                result[idx] = ''.join(shifts.get(c, c) for c in word)

            elif variation == 'consonant_double':
                # Raddoppia una consonante
                consonants = [i for i, c in enumerate(word) if c.lower() not in 'aeiou' and c.isalpha()]
                if consonants:
                    i = random.choice(consonants)
                    result[idx] = word[:i+1] + word[i] + word[i+1:]

            else:  # creative_spelling
                # Spelling creativo
                replacements = [
                    ('ph', 'f'), ('f', 'ph'),
                    ('ck', 'k'), ('k', 'ck'),
                    ('tion', 'shun'), ('sion', 'tion'),
                    ('y', 'ie'), ('ie', 'y'),
                    ('ee', 'ea'), ('ea', 'ee'),
                ]
                new_word = word
                for old, new in replacements:
                    if old in word.lower():
                        new_word = word.replace(old, new).replace(old.upper(), new.upper())
                        break
                result[idx] = new_word

        return " ".join(result)

    def generate_multiple(
        self,
        text: str,
        other_text: Optional[str] = None,
        predicate: Optional[str] = None,
        n: int = 3,
        validate: bool = True
    ) -> List[str]:
        """
        Genera multiple variazioni diverse e PLAUSIBILI.

        Args:
            text: Testo originale
            other_text: Testo opzionale per mix
            predicate: Nome del predicato (ignorato, per compatibilità)
            n: Numero di variazioni
            validate: Se True, filtra variazioni non plausibili

        Returns:
            Lista di variazioni uniche e plausibili
        """
        variations = set()
        attempts = 0
        max_attempts = n * 10  # Più tentativi per trovare variazioni plausibili

        while len(variations) < n and attempts < max_attempts:
            var = self.generate(text, other_text, predicate)

            # Escludi originale
            if var == text:
                attempts += 1
                continue

            # Validazione plausibilità (solo per testi corti)
            if validate and self._detect_type(text) == 'short':
                words = var.split()
                if all(_is_plausible_name(w) for w in words):
                    variations.add(var)
            else:
                variations.add(var)

            attempts += 1

        return list(variations)


# === TEST ===
if __name__ == "__main__":
    gen = CreativeVariationGenerator()

    print("=== TEST CREATIVE VARIATION - AGGRESSIVE ===\n")

    # TESTI CORTI (≤3 parole) - VARIAZIONI FORTI
    print("--- TESTI CORTI (nomi) - VARIAZIONI FORTI ---")
    short_tests = [
        ("John Smith", None),
        ("Mary Johnson", None),
        ("John Smith", "Mary Johnson"),  # Con mix
        ("supertramp", None),
    ]
    for text, other in short_tests:
        print(f"Input: '{text}'" + (f" + '{other}'" if other else ""))
        variations = gen.generate_multiple(text, other, n=6)
        for i, var in enumerate(variations, 1):
            print(f"  {i}. {var}")
        print()

    # NUMERI - VARIAZIONI FORTI!
    print("--- NUMERI - VARIAZIONI FORTI ---")
    num_tests = [
        "12345678",
        "1985",
        "2023-01-15",
        "ABC123XYZ",
    ]
    for text in num_tests:
        print(f"Input: '{text}'")
        variations = gen.generate_multiple(text, n=4)
        for i, var in enumerate(variations, 1):
            status = "✓ diverso" if var != text else "✗ uguale"
            print(f"  {i}. {var} ({status})")
        print()

    # TESTO LUNGO (>3 parole) - Riformulazione
    print("--- TESTO LUNGO - Riformulazione ---")
    long_text = "British rock band formed in London in 1970"
    print(f"Input: '{long_text}'")
    variations = gen.generate_multiple(long_text, n=3)
    for i, var in enumerate(variations, 1):
        print(f"  {i}. {var}")
