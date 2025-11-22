"""Tune PLM creativity parameters with iterative testing."""
import sys
sys.path.insert(0, '/home/federico/Programming/Python/DAKGEA')

from src.augmentation.methods.plm import PLMAugmenter
from src.core.dataset.reader.dataset_reader_factory import DatasetReaderFactory

# Load dataset once
print("Loading dataset...")
reader = DatasetReaderFactory.create_reader("bert_int")
dataset = reader.read("/home/federico/Programming/Python/DAKGEA/data/raw/openea/BBC_DB/attribute_data")
print(f"Dataset loaded: {len(dataset.aligned_entities)} aligned pairs\n")

# Test cases - varietà di pattern (non solo speculari!)
TEST_PAIRS = [
    # Speculari classici (2 token)
    ("downing kenneth", "kenneth downing", "name"),
    ("robert plant", "plant robert", "name"),
    ("jimmy page", "page jimmy", "name"),

    # Con 3 token in ordine diverso
    ("john paul jones", "jones john paul", "name"),
    ("ronnie james dio", "dio ronnie james", "name"),

    # Stesso nome identico (test creatività pura)
    ("led zeppelin", "led zeppelin", "name"),
    ("deep purple", "deep purple", "name"),

    # Varianti con middle name in posizioni diverse
    ("james patrick page", "page james patrick", "name"),
    ("robert anthony plant", "plant anthony robert", "name"),

    # Casi con iniziali
    ("j p jones", "jones j p", "name"),

    # Nomi parzialmente diversi ma simili
    ("ozzy osbourne", "john osbourne", "name"),
    ("bruce dickinson", "paul dickinson", "name"),
]

def test_configuration(config_dict: dict, iteration_name: str):
    """Test a specific configuration and show results."""
    print("=" * 80)
    print(f"ITERATION: {iteration_name}")
    print("=" * 80)
    print("Configuration:")
    for key, value in config_dict.get("augmentation", {}).get("bart", {}).items():
        print(f"  {key}: {value}")
    print()

    # Create augmenter with configuration
    augmenter = PLMAugmenter(config_dict)

    # Initialize BART (no retraining after first iteration)
    if config_dict.get("augmentation", {}).get("bart", {}).get("enable_finetuning", False):
        # Will train on first iteration only
        augmenter.augment(dataset)  # This will trigger training
    else:
        # Use existing model
        augmenter._initialize_bart_only(dataset)

    print("\nTesting interpolation on example pairs:")
    print("-" * 80)

    for src_val, tgt_val, predicate in TEST_PAIRS:
        # Generate 3 variations to see consistency and creativity
        results = []
        for i in range(3):
            aug_src, aug_tgt = augmenter.bart_interpolator.interpolate_pair(
                src_val, tgt_val, predicate=predicate
            )
            results.append((aug_src, aug_tgt))

        print(f"\n'{src_val}' + '{tgt_val}' →")
        for i, (aug_src, aug_tgt) in enumerate(results, 1):
            print(f"  Run {i}: '{aug_src}' / '{aug_tgt}'")

    print("\n" + "=" * 80)
    print()
    return augmenter


# ITERATION 1: Current settings (baseline)
print("\n\n")
print("╔" + "=" * 78 + "╗")
print("║" + " " * 20 + "STARTING PARAMETER TUNING" + " " * 33 + "║")
print("╚" + "=" * 78 + "╝")
print()

config1 = {
    'augmentation': {
        'max_pairs': 3,
        'max_depth': 1,
        'bart': {
            'model_name': 'facebook/bart-large',  # Using BART-LARGE
            'enable_finetuning': True,  # Need to train on large model
            'force_retrain': True,
            'out_dir': './bart_plm_model_large',
            'epochs': 3,  # Quick training
            'batch_size': 8,  # Smaller batch for large model
            'max_train_samples': 1000,  # Faster training
            'generation': {
                'temperature': 1.8,
                'top_p': 0.92,
                'num_beams': 5,
                'repetition_penalty': 1.7,
            },
            'base_alpha': 0.5,
            'alpha_spread': 0.35,
        }
    },
    'experiment': {'seed': 42}
}

test_configuration(config1, "1 - Current Settings (Baseline)")

# Wait for user input before next iteration
print("\n")
print("📊 Analizza i risultati sopra.")
print("   - Troppo creativo? (output irriconoscibili)")
print("   - Poco creativo? (troppo simili all'input)")
print("   - Buon equilibrio? (modifiche leggere ma creative)")
print()
input("Premi INVIO per testare la prossima configurazione...")
print("\n" * 3)

# ITERATION 2: Reduce temperature slightly
config2 = {
    'augmentation': {
        'max_pairs': 3,
        'max_depth': 1,
        'bart': {
            'model_name': 'facebook/bart-large',
            'enable_finetuning': False,  # Use model from iteration 1
            'out_dir': './bart_plm_model_large',
            'generation': {
                'temperature': 1.5,  # Ridotto da 1.8
                'top_p': 0.92,
                'num_beams': 5,
                'repetition_penalty': 1.7,
            },
            'base_alpha': 0.5,
            'alpha_spread': 0.35,
        }
    },
    'experiment': {'seed': 42}
}

test_configuration(config2, "2 - Lower Temperature (1.5)")

input("Premi INVIO per testare la prossima configurazione...")
print("\n" * 3)

# ITERATION 3: Reduce temperature more + lower alpha_spread
config3 = {
    'augmentation': {
        'max_pairs': 3,
        'max_depth': 1,
        'bart': {
            'model_name': 'facebook/bart-large',
            'enable_finetuning': False,
            'out_dir': './bart_plm_model_large',
            'generation': {
                'temperature': 1.2,  # Più conservativo
                'top_p': 0.90,       # Più conservativo
                'num_beams': 5,
                'repetition_penalty': 1.7,
            },
            'base_alpha': 0.5,
            'alpha_spread': 0.25,  # Meno variazione nell'interpolazione
        }
    },
    'experiment': {'seed': 42}
}

test_configuration(config3, "3 - Conservative (temp=1.2, alpha_spread=0.25)")

input("Premi INVIO per testare la prossima configurazione...")
print("\n" * 3)

# ITERATION 4: Medium creativity
config4 = {
    'augmentation': {
        'max_pairs': 3,
        'max_depth': 1,
        'bart': {
            'model_name': 'facebook/bart-large',
            'enable_finetuning': False,
            'out_dir': './bart_plm_model_large',
            'generation': {
                'temperature': 1.4,  # Medio
                'top_p': 0.90,
                'num_beams': 5,
                'repetition_penalty': 1.7,
            },
            'base_alpha': 0.5,
            'alpha_spread': 0.30,  # Medio
        }
    },
    'experiment': {'seed': 42}
}

test_configuration(config4, "4 - Medium Creativity (temp=1.4, alpha_spread=0.30)")

input("Premi INVIO per testare la prossima configurazione...")
print("\n" * 3)

# ITERATION 5: High beams for more quality
config5 = {
    'augmentation': {
        'max_pairs': 3,
        'max_depth': 1,
        'bart': {
            'model_name': 'facebook/bart-large',
            'enable_finetuning': False,
            'out_dir': './bart_plm_model_large',
            'generation': {
                'temperature': 1.4,
                'top_p': 0.90,
                'num_beams': 10,  # Più beams = output più coerenti
                'repetition_penalty': 1.5,  # Meno penalità
            },
            'base_alpha': 0.5,
            'alpha_spread': 0.30,
        }
    },
    'experiment': {'seed': 42}
}

test_configuration(config5, "5 - More Beams (num_beams=10)")

print("\n\n")
print("╔" + "=" * 78 + "╗")
print("║" + " " * 25 + "TUNING COMPLETATO" + " " * 36 + "║")
print("╚" + "=" * 78 + "╝")
print()
print("Quale configurazione preferisci? (1-5)")
print("  1. Current (temp=1.8, spread=0.35) - Molto creativo")
print("  2. Lower temp (temp=1.5, spread=0.35)")
print("  3. Conservative (temp=1.2, spread=0.25) - Poco creativo")
print("  4. Medium (temp=1.4, spread=0.30) - Equilibrato")
print("  5. More beams (temp=1.4, beams=10) - Più coerente")
print()
choice = input("Scelta (1-5): ")
print(f"\n✅ Hai scelto la configurazione {choice}")
