
import re
import pandas as pd
from collections import defaultdict
import markovify
from rdflib import Literal


class AttributePatternLearner:
    def __init__(self, min_len=3, max_len=60, state_size=2):
        """
        :param min_len: lunghezza minima accettata per i literal
        :param max_len: lunghezza massima accettata
        :param state_size: profondità del modello Markov
        """
        self.models = {}
        self.min_len = min_len
        self.max_len = max_len
        self.state_size = state_size

    # ---------------------------------------------------------------------- #
    # TRAINING
    # ---------------------------------------------------------------------- #
    def fit(self, triples):
        """
        Addestra un modello di pattern per ogni predicato nel grafo.
        :param triples: iterable di triple RDF (s, p, o)
        """
        data = defaultdict(list)
        for s, p, o in triples:
            if isinstance(o, Literal):
                val = str(o).strip()
                if self.min_len <= len(val) <= self.max_len:
                    data[str(p)].append(val)

        for predicate, values in data.items():
            if len(values) < 5:
                continue  # serve un po' di testo per stimare pattern
            text = "\n".join(values)
            try:
                model = markovify.NewlineText(text, state_size=self.state_size)
                self.models[predicate] = model
            except Exception as e:
                print(f"Skipping {predicate}: model error -> {e}")

        print(f"[AttributePatternLearner] Trained on {len(self.models)} predicates")

    # ---------------------------------------------------------------------- #
    # INFERENZA / GENERAZIONE
    # ---------------------------------------------------------------------- #
    def generate(self, predicate: str, tries: int = 5):
        """
        Genera un nuovo valore coerente con il pattern del predicato.
        """
        if predicate not in self.models:
            return None
        for _ in range(tries):
            val = self.models[predicate].make_sentence(tries=100)
            if val:
                return self._clean_output(val)
        return None

    # ---------------------------------------------------------------------- #
    # NORMALIZZAZIONE
    # ---------------------------------------------------------------------- #
    def normalize(self, predicate: str, value: str):
        """
        Pulisce un valore raw usando i pattern noti del predicato.
        Se possibile, ne sostituisce la forma con una coerente al modello.
        """
        value = str(value).strip()
        value = re.sub(r"^\d+[\.\-]\s*", "", value)
        value = re.sub(r"\s+", " ", value)
        value = value.strip(" .-")

        if predicate in self.models:
            new_val = self.generate(predicate)
            if new_val:
                return new_val

        # fallback: pulizia minimale
        value = re.sub(r"http\S+", "", value)
        value = re.sub(r"[^\w\s\.\-']", " ", value)
        value = re.sub(r"\s+", " ", value).strip()
        value = re.sub(r"\b(\w+)( \1\b)+", r"\1", value, flags=re.IGNORECASE)

        # capitalizzazione euristica
        if any(k in predicate.lower() for k in ["name", "givenname", "surname", "birthname"]):
            value = " ".join([w.capitalize() for w in value.split()])

        return value

    # ---------------------------------------------------------------------- #
    # UTILITÀ
    # ---------------------------------------------------------------------- #
    def _clean_output(self, text: str):
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\w\s\.\-']", " ", text)
        return text.strip(" .-")

    def save(self, path: str):
        """
        Salva i modelli addestrati (solo testi, non oggetti binari)
        """
        all_data = {pred: model.chain.model for pred, model in self.models.items()}
        import json
        with open(path, "w", encoding="utf-8") as f:
            json.dump(all_data, f, indent=2)

    def load(self, path: str):
        """
        Ricarica modelli salvati.
        """
        import json
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for pred, chain in data.items():
            model = markovify.NewlineText.from_chain(chain)
            self.models[pred] = model
        print(f"[AttributePatternLearner] Loaded {len(self.models)} predicate models")
