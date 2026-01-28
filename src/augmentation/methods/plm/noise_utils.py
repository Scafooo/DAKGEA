"""
Noise Utilities per Data Augmentation (Denoising).
Questo modulo fornisce funzioni per corrompere il testo in modo controllato,
simulando errori comuni o variazioni strutturali per addestrare modelli di Denoising (es. T5/BART).
"""

import random
import string
import re

def corrupt_text(text: str, noise_prob: float = 0.15) -> str:
    """
    Introduce rumore casuale in una stringa (cancellazione, swap, sostituzione caratteri).
    
    Args:
        text (str): Il testo originale.
        noise_prob (float): Probabilità di alterare ogni carattere (o densità del rumore).
        
    Returns:
        str: Il testo corrotto.
    """
    if not text or len(text) < 3: 
        return text # Troppo corto per essere corrotto senza distruggere l'informazione
    
    chars = list(text)
    n_chars = len(chars)
    # Numero di operazioni di rumore da applicare
    n_noise = max(1, int(n_chars * noise_prob))
    
    # Operazioni possibili
    ops = ['del', 'swap', 'sub', 'ins']
    
    for _ in range(n_noise):
        op = random.choice(ops)
        idx = random.randint(0, len(chars) - 1)
        
        if op == 'del':
            # Cancellazione: rimuove un carattere
            if len(chars) > 2: 
                del chars[idx]
                
        elif op == 'swap':
            # Scambio: scambia con il vicino
            if idx < len(chars) - 1:
                chars[idx], chars[idx+1] = chars[idx+1], chars[idx]
            elif idx > 0:
                chars[idx], chars[idx-1] = chars[idx-1], chars[idx]
                
        elif op == 'sub':
            # Sostituzione: cambia con un carattere casuale (simula typo estremo)
            chars[idx] = random.choice(string.ascii_letters + string.digits)
            
        elif op == 'ins':
            # Inserimento: aggiunge un carattere casuale
            char_to_add = random.choice(string.ascii_letters + string.digits + ' ')
            chars.insert(idx, char_to_add)
            
    return "".join(chars)

def corrupt_span(text: str, mask_ratio: float = 0.15) -> str:
    """
    Mascheramento alla BERT/T5: sostituisce span di testo con token speciali o li rimuove.
    Utile per insegnare al modello a 'riempire i buchi' (infilling).
    Per ora implementiamo una versione semplificata che droppa parole intere.
    """
    words = text.split()
    if len(words) < 3: return corrupt_text(text, mask_ratio)
    
    n_mask = max(1, int(len(words) * mask_ratio))
    indices = sorted(random.sample(range(len(words)), n_mask), reverse=True)
    
    for idx in indices:
        # 80% rimuovi parola, 20% sostituisci con [UNK] o simile (simulato)
        if random.random() < 0.8:
            del words[idx]
        else:
            words[idx] = "[MASK]" # O un placeholder che il tokenizer gestisce
            
    return " ".join(words)
