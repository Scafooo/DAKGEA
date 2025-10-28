#!/usr/bin/env python3
import sys
import json
import yaml
import time
import threading
from pathlib import Path
from tqdm import tqdm

from src.dataset.reader.ReaderFactory import ReaderFactory
from src.dataset.writer.WriterFactory import WriterFactory
from src.config.loader import Config
from src.reduction.registry import REDUCTION_REGISTRY
from src.augmentation.registry import AUGMENTATION_REGISTRY
from src.alignment_models.registry import MODEL_REGISTRY
from src.logger import logger

# --- autoload registries
REDUCTION_REGISTRY.autoload("src.reduction.methods")
AUGMENTATION_REGISTRY.autoload("src.augmentation.methods")
MODEL_REGISTRY.autoload("src.alignment_models.methods")


def auto_refresh_bar(bar, interval=0.3):
    stop_flag = {"stop": False}
    def refresher():
        while not stop_flag["stop"]:
            bar.refresh()
            time.sleep(interval)
    t = threading.Thread(target=refresher, daemon=True)
    t.start()
    return stop_flag

def dir_not_empty(path: Path) -> bool:
    return path.exists() and any(path.iterdir())


def run_all_experiments(config_path: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)["experiment"]

    writer_cfg = cfg.get("writer", None)

    global_cfg = Config().get()
    base_data = Path(global_cfg["paths"]["raw_data"])
    base_output = Path(global_cfg["paths"]["results"]) / cfg["name"]
    base_output.mkdir(parents=True, exist_ok=True)

    base_reduced = Path(global_cfg["paths"]["reduced_data"])
    base_augmented = Path(global_cfg["paths"]["augmented_data"])

    # resume = True → skip existing computations
    resume_mode = cfg.get("resume", True)

    logger.info(f"=== Starting experiment suite '{cfg['name']}' (resume={resume_mode}) ===")

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
        bar_format="{l_bar}{bar:40} | {n_fmt}/{total_fmt} ratios | ⏱️ {elapsed} | ETA {remaining}",
        file=sys.stdout,
    )
    refresh_flag = auto_refresh_bar(progress, interval=0.3)

    # --- MAIN LOOP ---
    for dataset_entry in datasets:
        if isinstance(dataset_entry, dict):
            dataset_name = dataset_entry.get("name")
            reader_type = dataset_entry.get("reader", "hybea")
            subtype = dataset_entry.get("subtype", "attribute_data")
            writer_conf = dataset_entry.get("writer", writer_cfg)
        else:
            dataset_name = dataset_entry
            reader_type = cfg.get("readers", {}).get(dataset_name, "hybea")
            subtype = cfg.get("dataset_type", "attribute_data")
            writer_conf = writer_cfg

        dataset_dir = base_data / reader_type / subtype / dataset_name
        dataset_out_dir = base_output / dataset_name
        dataset_out_dir.mkdir(parents=True, exist_ok=True)

        reader = ReaderFactory.create_reader(reader_type)
        dataset = reader.read(str(dataset_dir))
        logger.info(f"→ {dataset_name} (reader={reader_type}, subtype={subtype})")

        # --- Writer setup
        writer = None
        write_reduced = write_augmented = write_results = False
        if writer_conf:
            if isinstance(writer_conf, str):
                writer = WriterFactory.create_writer(writer_conf)
                write_reduced = write_augmented = write_results = True
            elif isinstance(writer_conf, dict):
                writer_type = writer_conf.get("type", reader_type)
                writer = WriterFactory.create_writer(writer_type)
                write_reduced  = writer_conf.get("write_reduced", False)
                write_augmented = writer_conf.get("write_augmented", False)
                write_results  = writer_conf.get("write_results", False)

        for ratio in ratios:
            ratio_str = f"{ratio*100:.1f}%"
            progress.set_description_str(f"📦 {dataset_name} [{ratio_str}]")

            config = {"reduction": {"target_entities": int(len(dataset.aligned_entities) * ratio)}}
            reducer = ReducerClass(config)

            # --- Output directories
            ratio_tag = f"{ratio * 100:.1f}"
            reduced_dir = base_reduced / dataset_name / ratio_tag
            augmented_dir = base_augmented / dataset_name / ratio_tag
            results_dir = base_output / dataset_name / ratio_tag

            # --- REDUCTION ---
            if resume_mode and dir_not_empty(reduced_dir):
                logger.info(f"⏭️ Skipping reduction for {dataset_name} ({ratio_str}) — already exists")
                dataset_reduced = reader.read(str(reduced_dir))
            else:
                dataset_reduced = reducer.reduce(dataset)
                if writer and write_reduced:
                    reduced_dir.mkdir(parents=True, exist_ok=True)
                    writer.write(dataset_reduced, str(reduced_dir))
                    logger.info(f"📝 Saved reduced dataset → {reduced_dir}")

            # --- AUGMENTATION + MODEL ---
            for aug_name in augmentations:
                aug_dir = augmented_dir / aug_name

                # AUGMENTATION skip check
                if resume_mode and dir_not_empty(aug_dir):
                    logger.info(f"⏭️ Skipping augmentation '{aug_name}' for {dataset_name} ({ratio_str}) — already exists")
                    dataset_augmented = reader.read(str(aug_dir))
                else:
                    AugClass = AUGMENTATION_REGISTRY.get(aug_name)
                    augmenter = AugClass(config)
                    dataset_augmented = augmenter.augment(dataset_reduced)
                    if writer and write_augmented:
                        aug_dir.mkdir(parents=True, exist_ok=True)
                        writer.write(dataset_augmented, str(aug_dir))
                        logger.info(f"📝 Saved augmented dataset → {aug_dir}")

                # MODEL EVALUATION skip check
                for model_name in models:
                    out_file = results_dir / f"{model_name}_{aug_name}.json"
                    if resume_mode and out_file.exists():
                        logger.info(f"⏭️ Skipping model '{model_name}' ({aug_name}) — results already exist")
                        continue

                    ModelClass = MODEL_REGISTRY.get(model_name)
                    model = ModelClass(config)
                    result = model.evaluate(dataset_reduced, dataset_augmented)

                    if writer and write_results:
                        results_dir.mkdir(parents=True, exist_ok=True)
                        with open(out_file, "w") as f:
                            json.dump(result, f, indent=2)
                        logger.info(f"💾 Saved results: {out_file}")

            progress.update(1)

    refresh_flag["stop"] = True
    progress.close()
    logger.info("=== All experiments completed successfully ===")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python experiments/run.py config/experiments/exp.yaml")
        sys.exit(1)

    start_time = time.strftime('%Y-%m-%d %H:%M:%S')
    logger.info(f"🕒 Started at {start_time}")
    run_all_experiments(sys.argv[1])
    end_time = time.strftime('%Y-%m-%d %H:%M:%S')
    logger.info(f"🏁 Finished at {end_time}")
