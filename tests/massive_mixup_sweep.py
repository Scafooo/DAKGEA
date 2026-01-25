import sys
import torch
import random
import re
from pathlib import Path
from collections import defaultdict
from difflib import SequenceMatcher

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.dataset.reader.openea_dataset_reader import OpeneaDatasetReader
from src.augmentation.methods.plm.mixup_bart_interpolator import MixupBartInterpolator
from src.augmentation.methods.plm.mixup_data_builder import MixupDataBuilder

# Canonical Predicate Map
OPENEA_CANONICAL_MAP = {
    "http://purl.org/ontology/mo/name": "<NAME>",
    "http://purl.org/dc/elements/1.1/title": "<NAME>",
    "http://xmlns.com/foaf/0.1/name": "<NAME>",
    "http://purl.org/dc/terms/date": "<DATE>",
    "http://purl.org/ontology/mo/genre": "<GENRE>",
    "http://xmlns.com/foaf/0.1/based_near": "<LOCATION>",
    "http://www.w3.org/2000/01/rdf-schema#comment": "<COMMENT>"
}

def string_sim(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def main():
    print("="*100)
    print(" MASSIVE MIX-UP PARAMETER SWEEP (300 EXAMPLES) ".center(100))
    print("="*100)

    # 1. Load Dataset & Model
    reader = OpeneaDatasetReader()
    dataset = reader.read("data/raw/openea/BBC_DB")
    model_path = "results/mixup_anchoring_test/checkpoint-23390"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    interpolator = MixupBartInterpolator(model_name=model_path, device=device, reuse_if_available=True)
    interpolator.set_predicate_mapping(OPENEA_CANONICAL_MAP)

    # 2. Extract 300 Real Pairs
    print("\nExtracting 300 real pairs...")
    builder = MixupDataBuilder()
    raw_pairs = builder.build_denoising_pairs(dataset, max_pairs_per_pred=1000)
    sampled_pairs = random.sample(raw_pairs, min(len(raw_pairs), 300))

    # 3. Sweep Grid
    noise_grid = [0.1, 0.2, 0.3]
    temp_grid = [1.0, 1.3, 1.6]
    
    best_overall_score = -1
    best_config = {}

    print(f"\n{'NOISE':<6} | {'TEMP':<5} | {'DIVERSITY':<10} | {'REMEMBRANCE':<12} | {'BALANCE'}")
    print("-" * 65)

    for noise in noise_grid:
        for temp in temp_grid:
            interpolator.latent_noise_std = noise
            interpolator.gen_temperature = temp
            
            total_diversity = 0
            total_remembrance = 0
            count = 0
            
            for p_uri, v_s, v_t in sampled_pairs:
                try:
                    res, _ = interpolator.interpolate_pair(v_s, v_t, predicate=p_uri)
                    if not res: continue
                    
                    # 1. Diversity: Distanza dagli input (1 - max_sim)
                    sim_s = string_sim(res, v_s)
                    sim_t = string_sim(res, v_t)
                    diversity = 1.0 - max(sim_s, sim_t)
                    
                    # 2. Remembrance: Quanto assomiglia ad almeno uno degli input
                    remembrance = max(sim_s, sim_t)
                    
                    total_diversity += diversity
                    total_remembrance += remembrance
                    count += 1
                except: continue
            
            avg_div = total_diversity / count
            avg_rem = total_remembrance / count
            # Balance Score: Harmonic mean per penalizzare eccessi (0 creativity o 0 remembrance)
            if avg_div + avg_rem > 0:
                balance = (2 * avg_div * avg_rem) / (avg_div + avg_rem)
            else:
                balance = 0
                
            print(f"{noise:<6.2f} | {temp:<5.1f} | {avg_div:<10.4f} | {avg_rem:<12.4f} | {balance:.4f}")
            
            if balance > best_overall_score:
                best_overall_score = balance
                best_config = {'noise': noise, 'temp': temp, 'div': avg_div, 'rem': avg_rem}

    print("\n" + "="*80)
    print(" BEST BALANCED CONFIGURATION ".center(80))
    print("="*80)
    print(f"Noise: {best_config['noise']}, Temp: {best_config['temp']}")
    print(f"Diversity: {best_config['div']:.4f} | Remembrance: {best_config['rem']:.4f}")
    print(f"Final Balance Score: {best_overall_score:.4f}")
    print("="*80)

    # 4. Show 10 examples with BEST config
    print("\nExamples with Best Config:")
    interpolator.latent_noise_std = best_config['noise']
    interpolator.gen_temperature = best_config['temp']
    for p_uri, v_s, v_t in sampled_pairs[:10]:
        res, _ = interpolator.interpolate_pair(v_s, v_t, predicate=p_uri)
        print(f"  [{p_uri.split('/')[-1]}] '{v_s}' + '{v_t}' -> '{res}'")

if __name__ == "__main__":
    main()
