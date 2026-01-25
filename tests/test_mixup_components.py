import sys
import torch
import logging
from pathlib import Path
from rdflib import URIRef

# Aggiunta del progetto al path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.dataset.reader.openea_dataset_reader import OpeneaDatasetReader
from src.augmentation.methods.plm.mixup_bart_interpolator import MixupBartInterpolator
from src.augmentation.methods.plm.mixup_data_builder import MixupDataBuilder

# Configurazione logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_mixup_pipeline():
    print("\n" + "="*80)
    print("TESTING MIX-UP COMPONENTS (BUILDER + INTERPOLATOR)")
    print("="*80)

    # 1. Caricamento Dataset (BBC_DB)
    print("\n1. Caricamento BBC_DB...")
    reader = OpeneaDatasetReader()
    dataset = reader.read("data/raw/openea/BBC_DB")
    print(f"   ✓ Dataset caricato: {len(dataset.aligned_entities)} allineamenti")

    # 2. Test MixupDataBuilder (Proximity Logic)
    print("\n2. Verifica MixupDataBuilder (Proximity Alignment)...")
    builder = MixupDataBuilder(confidence_threshold=0.6)
    # Prendiamo solo i primi 50 allineamenti per velocità
    dataset.aligned_entities = list(dataset.aligned_entities)[:50]
    
    raw_pairs = builder.build_denoising_pairs(dataset)
    
    print(f"   ✓ Coppie estratte: {len(raw_pairs)}")
    # Verifichiamo un esempio di coppia estratta
    if raw_pairs:
        p, v1, v2 = raw_pairs[0]
        print(f"   ✓ Esempio accoppiamento: [{p.split('/')[-1]}] '{v1}' ↔ '{v2}'")

    # 3. Test MixupBartInterpolator (Special Tokens + Mix-up)
    print("\n3. Verifica MixupBartInterpolator (Latent Mix-up)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Inizializziamo l'interpolatore
    interpolator = MixupBartInterpolator(
        model_name="facebook/bart-base",
        device=device,
        reuse_if_available=False
    )
    
    # Registriamo i predicati trovati nel builder
    unique_preds = list(set(p for p, _, _ in raw_pairs))
    interpolator.register_predicates(unique_preds)
    
    # Verifica resize embeddings
    vocab_size = len(interpolator.tokenizer)
    embed_size = interpolator.model.get_input_embeddings().weight.shape[0]
    print(f"   ✓ Vocab size: {vocab_size}, Embeddings size: {embed_size}")
    assert vocab_size == embed_size, "Modello non ridimensionato correttamente!"

    # 4. Test Interpolazione Reale
    print("\n4. Test Interpolazione Latente...")
    test_p = unique_preds[0] if unique_preds else ""
    v_src = "Judas Priest"
    v_tgt = "Priest, Judas"
    
    print(f"   Interpolazione di '{v_src}' e '{v_tgt}' con predicato {test_p}...")
    res_src, res_tgt = interpolator.interpolate_pair(v_src, v_tgt, predicate=test_p, alpha=0.5)
    
    print(f"   → Risultato: '{res_src}'")
    assert isinstance(res_src, str), "L'output dovrebbe essere una stringa!"
    
    print("\n" + "="*80)
    print("✅ TUTTI I COMPONENTI FUNZIONANO CORRETTAMENTE")
    print("="*80)

if __name__ == "__main__":
    try:
        test_mixup_pipeline()
    except Exception as e:
        logger.error(f"❌ Test fallito: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
