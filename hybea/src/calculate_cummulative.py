import os
import pickle
import sys

from logger import get_logger

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.alignment_models.methods.hybea import legacy_config as cfg

logger = get_logger(__name__)

def safe_div(numer, denom):
    return numer / denom if denom != 0 else 0.0


def measures(DATASET, model, iteration, MYPATH):
    test_set = []
    with open(cfg.DATA_TARGET + "/knowformer_data/" + DATASET + "/ref_ents.txt", "r") as fp:
        for line in fp:
            e1 = line.split("\t")[0]
            e2 = line.split("\t")[1].rstrip()
            test_set.append((e1, e2))

    ents_1 = [e1 for e1, e2 in test_set]
    ents_2 = [e2 for e1, e2 in test_set]

    pickle_path = MYPATH + "rec_new_pairs_from_" + model + str(iteration) + ".pickle"
    if not os.path.exists(pickle_path):
        logger.error(f"The file {pickle_path} does not exist.")
        return set()

    with open(pickle_path, "rb") as file:
        try:
            newp = pickle.load(file)
        except Exception as e:
            logger.error(f"Error cannot read {pickle_path}: {e}")
            return set()

    if not newp:
        logger.error(f"The file {pickle_path} is empty.")
        return set()

    newp_list = [pair[0] for pair in newp]
    e1_proposed_pairs = set(newp_list)
    correct_proposed_pairs = e1_proposed_pairs.intersection(set(ents_1))

    logger.info(f"Iter {iteration} - total proposed: {len(e1_proposed_pairs)}  |  test entities: {len(ents_1)}")

    # false positives
    fp = {e1 for e1 in e1_proposed_pairs if e1 not in ents_1}

    e1_not_proposed = set(ents_1) - e1_proposed_pairs
    tn = set()
    fn = set()
    for e1 in e1_not_proposed:
        if e1 in ents_1:
            fn.add(e1)
        else:
            tn.add(e1)

    tp_count = len(correct_proposed_pairs)
    fp_count = len(fp)
    fn_count = len(fn)

    precision = safe_div(tp_count, tp_count + fp_count)
    recall = safe_div(tp_count, tp_count + fn_count)
    f1_score = safe_div(2 * precision * recall, precision + recall)

    logger.info(f"Iter {iteration} - tp: {tp_count}  fp: {fp_count}  fn: {fn_count}")
    logger.info(f"Iter {iteration} - precision: {precision:.6f}  recall: {recall:.6f}  f1_score: {f1_score:.6f}")

    return set(e1_proposed_pairs)


def calc_measures(DATASET, MYPATH):
    cpr = 0.0
    crec = 0.0

    test_set = []
    with open(cfg.DATA_TARGET + "/knowformer_data/" + DATASET + "/ref_ents.txt", "r") as fp:
        for line in fp:
            e1 = line.split("\t")[0]
            e2 = line.split("\t")[1].rstrip()
            test_set.append((e1, e2))

    ents_1 = [e1 for e1, e2 in test_set]
    ents_2 = [e2 for e1, e2 in test_set]

    models = ["structure", "attr"]
    for ch_model in models:
        e1_pr_pairs = set()

        # conta quanti file di iterazione esistono
        index = 0
        while os.path.exists(MYPATH + "/" + "rec_new_pairs_from_" + ch_model + str(index) + ".pickle"):
            index += 1
        logger.info(f"Iter {index}")

        logger.info(f"Model: {ch_model}")
        if ch_model == "attr" and (DATASET == "ICEW_WIKI" or DATASET == "ICEW_YAGO"):
            dif = 1
        else:
            dif = 0

        # se index - dif <= 0 non entri nel loop
        for i in range(0, max(0, index - dif)):
            iter_new_pairs = measures(DATASET, ch_model, i, MYPATH)
            e1_pr_pairs = e1_pr_pairs.union(iter_new_pairs)

        tp = e1_pr_pairs.intersection(set(ents_1))
        fp = {e1 for e1 in e1_pr_pairs if e1 not in ents_1}

        e1_not_pr_pairs = set(ents_1) - set(e1_pr_pairs)
        tn = set()
        fn = set()
        for e1 in e1_not_pr_pairs:
            if e1 not in ents_1:
                tn.add(e1)
            elif e1 in ents_1:
                fn.add(e1)

        tp_count = len(tp)
        fp_count = len(fp)
        fn_count = len(fn)

        # gestire casi di zero division
        precision = safe_div(tp_count, tp_count + fp_count)
        recall = safe_div(tp_count, tp_count + fn_count)
        f1_score = safe_div(2 * precision * recall, precision + recall)

        cpr += precision
        crec += recall

        logger.info(f"precision: {precision:.6f}  recall: {recall:.6f}  f1_score: {f1_score:.6f}")

    cpr = cpr / 2.0

    logger.info(f"Cummulative precision: {cpr:.6f}")
    logger.info(f"Cummulative recall: {crec:.6f}")

    denom = (cpr + crec)
    if denom == 0:
        logger.info("Cummulative f1_score: 0.0")
    else:
        logger.info("Cummulative f1_score: " + str((2 * cpr * crec) / denom))


if __name__ == "__main__":

    args = sys.argv
    if len(args) < 3:
        logger.info("Usage: python calculate_cummulative.py DATASET MYPATH")
        sys.exit(1)

    DATASET = args[1]
    MYPATH = args[2]

    calc_measures(DATASET, MYPATH)
