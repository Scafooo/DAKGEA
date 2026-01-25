import sys
import torch
import random
from pathlib import Path
from tabulate import tabulate

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.dataset.reader.openea_dataset_reader import OpeneaDatasetReader
from src.augmentation.methods.plm.mixup_bart_interpolator import MixupBartInterpolator
from src.augmentation.methods.plm.mixup_data_builder import MixupDataBuilder

# PARAMETRI CAMPIONI (Dal tuo Sweep)
BEST_NOISE = 0.5
BEST_TEMP = 1.0
BEST_ALPHA = 0.3

def run_champion_report():
    print("\n" + "█"*100)
    print("█" + " CHAMPION REPORT: BART-LARGE (BBC_DB) ".center(98) + "█")
    print("█" + f" Config: Noise={BEST_NOISE}, Temp={BEST_TEMP}, Alpha={BEST_ALPHA} ".center(98) + "█")
    print("█"*100)

    # 1. Caricamento Dataset e Builder per i Token
    data_path = PROJECT_ROOT / "data" / "raw" / "openea" / "BBC_DB"
    reader = OpeneaDatasetReader()
    dataset = reader.read(str(data_path))
    builder = MixupDataBuilder()
    train_rows, canonical_map = builder.build_training_data(dataset)
    
    # 2. Caricamento Modello
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = "./results/sweep_model_large"
    
    interpolator = MixupBartInterpolator(
        model_name=out_dir, 
        out_dir=out_dir,
        device=device
    )
    interpolator.set_predicate_mapping(canonical_map)
    interpolator.latent_noise_std = BEST_NOISE
    interpolator.gen_temperature = BEST_TEMP

    # 3. Generazione Report Qualitativo (50 campioni)
    print("\n>>> GENERATING QUALITATIVE EXAMPLES...")
    report_data = []
    
    # Prendiamo campioni diversi (sia traduzioni che orfani)
    eval_subset = random.sample(train_rows, 50)
    
    for i, row in enumerate(eval_subset):
        inp, tgt = row['input'], row['target']
        parts_inp = inp.split(' ', 1)
        parts_tgt = tgt.split(' ', 1)
        
        pred = parts_inp[0]
        v1 = parts_inp[1] if len(parts_inp) > 1 else ""
        v2 = parts_tgt[1] if len(parts_tgt) > 1 else ""
        
        # Generazione
        aug, _ = interpolator.interpolate_pair(v1, v2, predicate=pred, alpha=BEST_ALPHA)
        
        # Status
        is_new = aug.lower().strip() not in [v1.lower().strip(), v2.lower().strip()]
        tag = "[✨ NEW]" if is_new else "[⚠️ COPY]"
        
        report_data.append([i+1, pred, v1[:25], v2[:25], f"{aug[:35]} {tag}"])

    print("\n" + "="*100)
    print(" QUALITATIVE REPORT (BBC_DB) ".center(100))
    print("="*100)
    print(tabulate(report_data, headers=["#", "PRED", "VAL A", "VAL B", "AUGMENTED"], tablefmt="grid"))
    print("="*100)

if __name__ == "__main__":
    run_champion_report()
