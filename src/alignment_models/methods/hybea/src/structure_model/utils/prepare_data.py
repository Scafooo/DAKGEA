import os
import logging
from ..reader.helper import read_triples
from ..reader.helper import write_triples
from ..reader.helper import read_entity_paris
from ..reader.helper import write_vocab_path
from ..reader.helper import load_vocab
from src.logger import get_logger, get_structured_logger

logger = get_logger(__name__)
slogger = get_structured_logger(__name__)


def prepare_entity_alignment_data(args, new_pairs):
    """Prepare entity alignment data for training."""
    slogger.section("Entity Alignment Data Preparation")

    sup_ents_path = os.path.join(args.dataset_root_path, args.dataset, "sup_ents.txt")
    ref_ents_path = os.path.join(args.dataset_root_path, args.dataset, args.ea_ref_file)
    ent_ILLs_path = os.path.join(args.dataset_root_path, args.dataset, args.ea_all_file)
    s_triples_path = os.path.join(args.dataset_root_path, args.dataset, args.ea_source_triples_file)
    t_triples_path = os.path.join(args.dataset_root_path, args.dataset, args.ea_target_triples_file)
    train_triples_path = os.path.join(args.dataset_root_path, args.dataset, "train.triples.txt")
    vocab_path = os.path.join(args.dataset_root_path, args.dataset, args.vocab_file)

    # Log data paths
    slogger.table("Input Data Paths", {
        "Source triples": s_triples_path,
        "Target triples": t_triples_path,
        "Reference entities": ref_ents_path,
        "Aligned entities": ent_ILLs_path
    })

    # Verify all required files exist
    required_files = {
        "Source triples": s_triples_path,
        "Target triples": t_triples_path,
        "Reference entities": ref_ents_path,
        "Aligned entities": ent_ILLs_path
    }

    missing_files = [name for name, path in required_files.items() if not os.path.exists(path)]
    if missing_files:
        logger.error(f"Missing required files: {', '.join(missing_files)}")
        raise FileNotFoundError(f"Missing files: {missing_files}")

    slogger.success("All required data files found")


    slogger.subsection("Triple Merging")

    if len(new_pairs) > 0:
        logger.info(f"Processing {len(new_pairs)} new entity pairs")

        sup_new_ents_path = os.path.join(args.dataset_root_path, args.dataset, "sup_new_ents.txt")
        train_new_triples_path = os.path.join(args.dataset_root_path, args.dataset, "train.new.triples.txt")

        sup_set = []
        with open(sup_ents_path, "r") as fp:
            for line in fp:
                sup_set.append((line.split("\t")[0], line .split("\t")[1].rstrip()))

        for pair in new_pairs:
            sup_set.append(pair)

        with open(sup_new_ents_path, "w") as fp:
            for pair in sup_set:
                fp.write(pair[0])
                fp.write("\t")
                fp.write(pair[1])
                fp.write("\n")

        def merge(train_new_triples_path_, s_triples_path_, t_triples_path_, sup_new_ents_path):
            logger.info("Reading source and target triples...")
            s_triples = read_triples(s_triples_path_)
            t_triples = read_triples(t_triples_path_)
            logger.debug(f"Source triples: {len(s_triples)}, Target triples: {len(t_triples)}")

            logger.info("Building entity mapping...")
            sup_ents = read_entity_paris(sup_new_ents_path)
            t_s_map = {}
            for s, t in sup_ents:
                t_s_map[t] = s
            logger.debug(f"Entity mapping size: {len(t_s_map)}")

            logger.info("Merging triples...")
            train_triples = s_triples.copy()
            for t_triple in t_triples:
                t_triple_h, t_triple_r, t_triple_t = t_triple
                t_triple_h = t_s_map.get(t_triple_h, t_triple_h)
                t_triple_t = t_s_map.get(t_triple_t, t_triple_t)
                train_triples.append((t_triple_h, t_triple_r, t_triple_t))

            assert len(train_triples) == len(s_triples) + len(t_triples)
            logger.info(f"Merged triples: {len(train_triples)}")
            write_triples(train_new_triples_path_, train_triples, is_add_mask=True)

        merge(train_new_triples_path, s_triples_path, t_triples_path, sup_ents_path)

    else:
        if not os.path.exists(train_triples_path):
            def merge(train_triples_path_, s_triples_path_, t_triples_path_, sup_ents_path_):
                logger.info("Reading source and target triples...")
                s_triples = read_triples(s_triples_path_)
                t_triples = read_triples(t_triples_path_)
                logger.debug(f"Source triples: {len(s_triples)}, Target triples: {len(t_triples)}")

                logger.info("Building entity mapping...")
                sup_ents = read_entity_paris(sup_ents_path_)
                t_s_map = {}
                for s, t in sup_ents:
                    t_s_map[t] = s
                logger.debug(f"Entity mapping size: {len(t_s_map)}")

                logger.info("Merging triples...")
                train_triples = s_triples.copy()
                for t_triple in t_triples:
                    t_triple_h, t_triple_r, t_triple_t = t_triple
                    t_triple_h = t_s_map.get(t_triple_h, t_triple_h)
                    t_triple_t = t_s_map.get(t_triple_t, t_triple_t)
                    train_triples.append((t_triple_h, t_triple_r, t_triple_t))

                assert len(train_triples) == len(s_triples) + len(t_triples)
                logger.info(f"Merged triples: {len(train_triples)}")
                write_triples(train_triples_path_, train_triples, is_add_mask=True)

            merge(train_triples_path, s_triples_path, t_triples_path, sup_ents_path)

    slogger.subsection("Vocabulary Extraction")

    logger.info("Extracting entities and relations from training triples...")
    entities_set = set()
    relations_set = set()

    train_triples = read_triples(train_triples_path)
    for triple in train_triples:
        for i, label in enumerate(triple):
            if label.startswith("MASK"):
                continue
            if i % 2 == 0:
                entities_set.add(label)
            else:
                relations_set.add(label)

    logger.debug(f"Extracted {len(entities_set)} entities and {len(relations_set)} relations from triples")

    logger.info("Adding reference entities...")
    ref_ents_path = os.path.join(args.dataset_root_path, args.dataset, "ref_ents.txt")
    with open(ref_ents_path, "r") as fp:
        for line in fp:
            entities_set.add(line.split("\t")[0])
            entities_set.add(line.split("\t")[1].rstrip())

    logger.info("Adding validation entities...")
    valid_ents_path = os.path.join(args.dataset_root_path, args.dataset, "valid_ents.txt")
    with open(valid_ents_path, "r") as fp:
        for line in fp:
            entities_set.add(line.split("\t")[0])
            entities_set.add(line.split("\t")[1].rstrip())

    logger.debug(f"Total entities after adding ref/valid: {len(entities_set)}")

    def custom_sort_key(entity):
        if 'http' in entity:
            return 0
        elif 'dbp:' in entity:
            return 1
        else:
            return 2
        
    def custom_sort_key_fr_en(entity):
        if 'http://fr.dbpedia' in entity:
            return 0
        elif 'http://dbpedia.org' in entity:
            return 1
        else:
            return 2
        
    def custom_sort_key_ja_en(entity):
        if 'http://ja.dbpedia' in entity:
            return 0
        elif 'http://dbpedia.org' in entity:
            return 1
        else:
            return 2
        
    def custom_sort_key_zh_en(entity):
        if 'http://zh.dbpedia' in entity:
            return 0
        elif 'http://dbpedia.org' in entity:
            return 1
        else:
            return 2
        
    def custom_sort_icew_wiki(entity):
        if 'https://en.wikipedia.org/wiki/' in entity:
            return 1
        else:
            return 0
        
    def custom_sort_icew_yago(entity):
        if 'http://yago-knowledge.org/resource/' in entity:
            return 1
        else:
            return 0

    logger.info("Sorting entities and relations...")
    entities_list = sorted(list(entities_set))
    relations_list = sorted(list(relations_set))

    # Apply dataset-specific sorting
    if args.dataset == "ICEW_WIKI":
        logger.debug("Applying ICEW_WIKI sorting")
        entities_list = sorted(list(entities_set), key=custom_sort_icew_wiki)
        relations_list = sorted(list(relations_set), key=custom_sort_icew_wiki)

    elif args.dataset == "ICEW_YAGO":
        logger.debug("Applying ICEW_YAGO sorting")
        entities_list = sorted(list(entities_set), key=custom_sort_icew_yago)
        relations_list = sorted(list(relations_set), key=custom_sort_icew_yago)

    elif args.dataset == "BBC_DB":
        logger.debug("Applying BBC_DB sorting")
        entities_list = sorted(list(entities_set), key=custom_sort_key)
        relations_list = sorted(list(relations_set), key=custom_sort_key)
        
    if args.dataset == "fr_en":
        entities_list = sorted(list(entities_set), key=custom_sort_key_fr_en)
        relations_list = sorted(list(relations_set), key=custom_sort_key_fr_en)
        
    if args.dataset == "ja_en":
        entities_list = sorted(list(entities_set), key=custom_sort_key_ja_en)
        relations_list = sorted(list(relations_set), key=custom_sort_key_ja_en)
        
    if args.dataset == "zh_en":
        entities_list = sorted(list(entities_set), key=custom_sort_key_zh_en)
        relations_list = sorted(list(relations_set), key=custom_sort_key_zh_en)

    args.vocab_size = (100 + len(entities_list) + len(relations_list))
    args.num_relations = len(relations_list)

    slogger.subsection("Vocabulary Statistics")

    # Log vocabulary statistics
    slogger.table("Vocabulary Composition", {
        "Special Tokens": 100,
        "Entities": len(entities_list),
        "Relations": len(relations_list),
        "Total Vocabulary Size": args.vocab_size,
        "Number of Relations": args.num_relations
    })

    # Check for extra tokens in vocab file
    if os.path.exists(vocab_path):
        vocab_tokens = set(load_vocab(vocab_path))
        expected_tokens = set(entities_list + relations_list)
        extra_tokens = vocab_tokens - expected_tokens

        if len(extra_tokens) > 100:
            logger.warning(f"Found {len(extra_tokens) -100} extra tokens in vocabulary file")
            # logger.debug(f"Extra tokens: {extra_tokens}")

    # Write or verify vocabulary
    if not os.path.exists(vocab_path):
        logger.info(f"Creating vocabulary file: {vocab_path}")
        write_vocab_path(vocab_path, entities_list, relations_list)
        slogger.success(f"Vocabulary file created successfully ({args.vocab_size} tokens)")
    else:
        vocab_size_file = len(load_vocab(vocab_path))
        if args.vocab_size == vocab_size_file:
            slogger.success(f"Vocabulary file verified (size: {vocab_size_file})")
        else:
            logger.error(f"Vocabulary size mismatch: expected {args.vocab_size}, got {vocab_size_file}")
            raise AssertionError(f"Vocabulary size mismatch: {args.vocab_size} != {vocab_size_file}")