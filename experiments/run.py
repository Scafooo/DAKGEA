#!/usr/bin/env python3
import sys
import json
import yaml
import time
import threading
from pathlib import Path
from tqdm import tqdm

from src.dataset.reader.ReaderFactory import ReaderFactory
from src.config.loader import Config
from src.reduction.registry import REDUCTION_REGISTRY
from src.augmentation.registry import AUGMENTATION_REGISTRY
from src.alignment_models.registry import MODEL_REGISTRY
from src.logger import logger

# --- autoload registries
REDUCTION_REGISTRY.autoload("src.reduction.methods")
AUGMENTATION_REGISTRY.autoload("src.augmentation.methods")
MODEL_REGISTRY.autoload("src.alignment_models.methods")


# 🔸 thread per ETA aggiornato in tempo reale
def auto_refresh_bar(bar, interval=0.3):
    """Aggiorna ETA costantemente ogni `interval` secondi."""
    stop_flag = {"stop": False}

    def refresher():
        while not stop_flag["stop"]:
            bar.refresh()
            time.sleep(interval)

    t = threading.Thread(target=refresher, daemon=True)
    t.start()
    return stop_flag


def run_all_experiments(config_path: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)["experiment"]

    global_cfg = Config().get()
    base_data = Path(global_cfg["paths"]["raw_data"])
    base_output = Path(global_cfg["paths"]["results"]) / cfg["name"]
    base_output.mkdir(parents=True, exist_ok=True)

    logger.info(f"=== Starting experiment suite '{cfg['name']}' ===")

    # setup
    datasets = cfg["datasets"]
    ratios = cfg["reduction_ratios"]
    augmentations = cfg["augmentation_methods"]
    models = cfg["models_to_run"]

    ReducerClass = REDUCTION_REGISTRY.get(cfg.get("reduction_method", "ids"))

    total_steps = len(datasets) * len(ratios)

    progress = tqdm(
        total=total_steps,
        desc="🚀 Running DAKGEA experiments",
        unit="ratio",
        dynamic_ncols=True,
        colour="cyan",
        smoothing=0.3,
        # ✅ aggiunto contatore numerico e ETA fluido
        bar_format="{l_bar}{bar:40} | {n_fmt}/{total_fmt} ratios | ⏱️ {elapsed} | ETA {remaining}",
        file=sys.stdout,
    )

    # start auto-refresh
    refresh_flag = auto_refresh_bar(progress, interval=0.3)

    # main loop
    for dataset_entry in datasets:
        if isinstance(dataset_entry, dict):
            dataset_name = dataset_entry.get("name")
            reader_type  = dataset_entry.get("reader", "hybea")
            subtype      = dataset_entry.get("subtype", "attribute_data")
        else:
            dataset_name = dataset_entry
            reader_type  = cfg.get("readers", {}).get(dataset_name, "hybea")
            subtype      = cfg.get("dataset_type", "attribute_data")

        dataset_dir = base_data / reader_type / subtype / dataset_name
        dataset_out_dir = base_output / dataset_name
        dataset_out_dir.mkdir(parents=True, exist_ok=True)

        reader = ReaderFactory.create_reader(reader_type)
        dataset = reader.read(str(dataset_dir))
        logger.info(f"→ {dataset_name} (reader={reader_type}, subtype={subtype})")

        for ratio in ratios:
            ratio_str = f"{ratio*100:.1f}%"
            progress.set_description_str(f"📦 {dataset_name} [{ratio_str}]")

            config = {"reduction": {"target_entities": int(len(dataset.aligned_entities) * ratio)}}
            reducer = ReducerClass(config)
            dataset_reduced = reducer.reduce(dataset)

            for aug_name in augmentations:
                AugClass = AUGMENTATION_REGISTRY.get(aug_name)
                augmenter = AugClass(config)
                dataset_augmented = augmenter.augment(dataset_reduced)

                for model_name in models:
                    ModelClass = MODEL_REGISTRY.get(model_name)
                    model = ModelClass(config)
                    result = model.evaluate(dataset_reduced, dataset_augmented)

                    ratio_dir = dataset_out_dir / f"{ratio*100:.1f}"
                    ratio_dir.mkdir(exist_ok=True)
                    out_file = ratio_dir / f"{model_name}_{aug_name}.json"
                    with open(out_file, "w") as f:
                        json.dump(result, f, indent=2)

            # update progress after ratio done
            progress.update(1)

    # stop refresh + close bar
    refresh_flag["stop"] = True
    progress.close()
    logger.info("=== All experiments completed successfully ===")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python experiments/run.py config/experiments/exp.yaml")
        sys.exit(1)
    logger.info(f"🕒 Started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    run_all_experiments(sys.argv[1])
    logger.info(f"🏁 Finished at {time.strftime('%Y-%m-%d %H:%M:%S')}")
