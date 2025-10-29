import os
import argparse
import torch

from .utils.args import ArgumentGroup
from .utils.args import print_arguments
from .model.knowformer import Knowformer
from .utils.tools import device
from .utils.swa import swa
from .utils.prepare_data import prepare_entity_alignment_data
from .utils.logger import get_logger
import numpy as np
import random
from .train.train_task import entity_alignment_train
from .valid.test_task import entity_alignment_test
from src.alignment_models.methods.hybea import runtime as cfg, path_for_KG
from .utils.tools import calculate_voc_lim
# from structure_model.graph_trans import *

def get_args(new_pairs):
    VOC_LIM_1, VOC_LIM_2 = calculate_voc_lim()

    parser = argparse.ArgumentParser()

    model_g = ArgumentGroup(parser, "model", "model configuration and paths")
    model_g.add_arg("hidden_size", int, cfg.HIDDEN_SIZE, "Knowformer model config: hidden size, default is 256")
    model_g.add_arg("num_hidden_layers", int, cfg.NUM_HIDDEN_LAYERS, "Knowformer model config: num_hidden_layers, default is 6")
    model_g.add_arg("num_attention_heads", int, cfg.NUM_ATTENTION_HEADS, "Knowformer model config: num_attention_heads, default is 4")
    model_g.add_arg("input_dropout_prob", float, cfg.INPUT_DROPOUT_PROB, "Knowformer model config: input_dropout_prob, default is 0.5")
    model_g.add_arg("attention_dropout_prob", float, cfg.ATTENTION_DROPOUT_PROB,
                    "Knowformer model config: attention_dropout_prob, default is 0.1")
    model_g.add_arg("hidden_dropout_prob", float, cfg.HIDDEN_DROPOUT_PROB, "Knowformer model config: hidden_dropout_prob, default is 0.1")
    model_g.add_arg("residual_dropout_prob", float, cfg.RESIDUAL_DROPOUT_PROB,
                    "Knowformer model config: residual_dropout_prob, default is 0.1")
    model_g.add_arg("initializer_range", float, cfg.INITIALIZER_RANGE, "Knowformer model config: initializer_range, default is 0.02")
    model_g.add_arg("intermediate_size", int, cfg.INTERMEDIATE_SIZE, "Knowformer model config: intermediate_size, default is 512")
    model_g.add_arg("residual_w", float, cfg.RESIDUAL_W, "Knowformer model config: residual_w, default is 0.5")
    model_g.add_arg("addition_loss_w", float, cfg.ADDITION_LOSS_W, "Knowformer model config: addition_loss_w, default is 0.1")
    model_g.add_arg("relation_combine_dropout_prob", float, cfg.RELATION_COMBINE_DROPOUT_PROB, "Knowformer model config: relation_combine_dropout_prob, default is 0.1")
    model_g.add_arg("use_gelu", bool, cfg.USE_GELU, "whether to use gelu in position combination of Knowformer, default is False")
    model_g.add_arg("q_dropout_prob", float, 0,
                    "Knowformer model config: q_dropout_prob, default is 0")
    model_g.add_arg("k_dropout_prob", float, 0,
                    "Knowformer model config: k_dropout_prob, default is 0")
    model_g.add_arg("v_dropout_prob", float, 0,
                    "Knowformer model config: v_dropout_prob, default is 0")

    train_g = ArgumentGroup(parser, "training and testing", "training and testing options")
    train_g.add_arg("epoch", int, cfg.EPOCH, "number of epochs for training, default is 500")
    train_g.add_arg("min_epochs", int, cfg.MIN_EPOCHS, "number of min epochs for training, default is 200")
    train_g.add_arg("early_stop_max_times", int, cfg.EARLY_STOP_MAX_TIMES,
                    "if in successive epochs, there is no improvement in valid, then early stop, default is 2")
    train_g.add_arg("eval_freq", int, cfg.EVAL_FREQ, "validation frequency, default is 5")
    train_g.add_arg("save_freq", int, 5, "save model frequency, default is 5") 
    train_g.add_arg("start_eval", int, cfg.START_EVAL, "if current epoch < start_eval, without eval, default is 0")
    train_g.add_arg("start_save", int, 100, "if current epoch < start_eval, without save model, default is 100") 
    train_g.add_arg("swa_pre_num", int, cfg.SWA_PRE_NUM, "load previous some saved models to do swa, default is 20")
    train_g.add_arg("learning_rate", float, cfg.LEARNING_RATE_STRUCTURE, "learning rate, default is 5e-4")
    train_g.add_arg("batch_size", int, cfg.BATCH_SIZE, "batch size for training, default is 2048")
    train_g.add_arg("eval_batch_size", int, cfg.EVAL_BATCH_SIZE, "batch size for evaluation, default is 4096")
    train_g.add_arg("soft_label", float, cfg.SOFT_LABEL, "soft label for computing loss, default is 0.25")
    train_g.add_arg("lp_eval_type", str, "ordinal", "eval type of LP, default is ordinal. All options: top, bottom, random, ordinal")
    train_g.add_arg("loaded_param_folder", str, None, "to load the param folder")

    data_g = ArgumentGroup(parser, "data", "basic data options")
    data_g.add_arg("dataset", str, cfg.DATASET, "dataset name, default is fb15k237")
    data_g.add_arg("dataset_root_path", str, cfg.DATA_TARGET + "/knowformer_data", "dataset_root_path, default is data")
    data_g.add_arg("task", str, cfg.TASK,
                   "task (entity-alignment), default is entity-alignment")
    kg1_name, kg2_name = path_for_KG(cfg.DATASET)
    data_g.add_arg("KG1_PATH_FOR_NAMES", str, kg1_name, "names for kg1")
    data_g.add_arg("KG2_PATH_FOR_NAMES", str, kg2_name, "names for kg2")
    
    data_g.add_arg("VOC_LIM_1", str, VOC_LIM_1, "voc lim 1")
    data_g.add_arg("VOC_LIM_2", str, VOC_LIM_2, "voc lim 2")

    data_g.add_arg("SIZE_AFTER_REDUCTION_IN_PERCENTAGE", str, str(round(cfg.SIZE_AFTER_REDUCTION_IN_PERCENTAGE,1)), "reduce after reduction")

    # if len(new_pairs)> 0:
    #     data_g.add_arg("vocab_file", str, "vocab_new.txt", "vocab file, default is vocab.txt")
    # else:
    data_g.add_arg("vocab_file", str, "vocab.txt", "vocab file, default is vocab.txt")
        
    data_g.add_arg("vocab_size", int, None, "size of vocab")
    data_g.add_arg("num_relations", int, None, "size of relations in vocab")

    lp_data_g = ArgumentGroup(parser, "data for link prediction", "data options for link prediction")
    lp_data_g.add_arg("lp_train_origin_file", str, "train.origin.txt",
                      "train origin file for link prediction, default is train.origin.txt")
    lp_data_g.add_arg("lp_valid_origin_file", str, "valid.origin.txt",
                      "valid origin file for link prediction, default is valid.origin.txt")
    lp_data_g.add_arg("lp_test_origin_file", str, "test.origin.txt",
                      "valid origin file for link prediction, default is valid.origin.txt")
    lp_data_g.add_arg("lp_train_file", str, "train.txt",
                      "train file for link prediction, default is train.txt")
    lp_data_g.add_arg("lp_valid_file", str, "valid.txt",
                      "valid file for link prediction, default is valid.txt")
    lp_data_g.add_arg("lp_test_file", str, "test.txt",
                      "valid file for link prediction, default is valid.txt")
    lp_data_g.add_arg("lp_all_true_file", str, "all.txt",
                      "all true triples for link prediction, default is all.txt")

    ea_data_g = ArgumentGroup(parser, "data for entity alignment", "data options for entity alignment")

    if len(new_pairs)> 0:
        ea_data_g.add_arg("ea_sup_file", str, "sup_new_ents.txt", "new entity pairs proposed by another model")
    else:
        ea_data_g.add_arg("ea_sup_file", str, "sup_ents.txt", "supervise entity pairs file for entity alignment, default is sup_ents.txt")

    ea_data_g.add_arg("ea_ref_file", str, "ref_ents.txt",
                      "reference entity pairs file for entity alignment, default is ref_ents.txt")
    
    ea_data_g.add_arg("ea_validation_file", str, "valid_ents.txt",
                      "reference entity pairs file for entity alignment, default is valid_ents.txt")
        
    ea_data_g.add_arg("ea_all_file", str, "ent_ILLs.txt",
                      "all entity pairs file for entity alignment, default is ent_ILLs.txt")
    ea_data_g.add_arg("ea_source_triples_file", str, "s_triples.txt",
                      "source triples file for entity alignment, default is s_triples.txt")
    ea_data_g.add_arg("ea_target_triples_file", str, "t_triples.txt",
                      "target triples file for entity alignment, default is t_triples.txt")

    if len(new_pairs)> 0:
        ea_data_g.add_arg("ea_train_triples_file", str, "train.new.triples.txt",
                      "train triples file for entity alignment, default is train.triples.txt")
    else:
        ea_data_g.add_arg("ea_train_triples_file", str, "train.triples.txt",
                      "train triples file for entity alignment, default is train.triples.txt")

    run_type_g = ArgumentGroup(parser, "run type", "running options")
    run_type_g.add_arg("do_train", bool, cfg.DO_TRAIN, "whether to perform training, default is True")
    run_type_g.add_arg("do_test", bool, cfg.DO_TEST, "whether to perform valid, default is True")
    run_type_g.add_arg("use_cuda", bool, True, "whether to use cuda, default is True")
    
    args_ = parser.parse_args([])
    assert args_.task is not None and (args_.task == "entity-alignment")

    return args_


def get_net_config(args_):
    config = dict()
    config["hidden_size"] = args_.hidden_size
    config["num_hidden_layers"] = args_.num_hidden_layers
    config["num_attention_heads"] = args_.num_attention_heads
    config["input_dropout_prob"] = args_.input_dropout_prob
    config["attention_dropout_prob"] = args_.attention_dropout_prob
    config["hidden_dropout_prob"] = args_.hidden_dropout_prob
    config["residual_dropout_prob"] = args_.residual_dropout_prob
    config["initializer_range"] = args_.initializer_range
    config["intermediate_size"] = args_.intermediate_size
    config["residual_w"] = args_.residual_w
    config["addition_loss_w"] = args_.addition_loss_w
    config["relation_combine_dropout_prob"] = args_.relation_combine_dropout_prob
    config["use_gelu"] = args_.use_gelu
    config["q_dropout_prob"] = args_.q_dropout_prob
    config["k_dropout_prob"] = args_.k_dropout_prob
    config["v_dropout_prob"] = args_.v_dropout_prob

    config["vocab_size"] = args_.vocab_size
    config["num_relations"] = args_.num_relations
    return config


def run(args, logger_, stamp_):
    if not (args.do_train or args.do_test):
        assert 0
    net_config = get_net_config(args)
    
    my_model = Knowformer(net_config, args)
    
    
    if args.use_cuda:
        my_model = my_model.to("cuda:0")
    
    
    # saved_model_folder = os.path.join("./saved_model_Knowformer_{}/".format(args.task), stamp_)
    
    if args.do_train:
        # if not os.path.exists(saved_model_folder):
            # os.makedirs(saved_model_folder)
        if args.task == "entity-alignment":
            export_sim_mat = entity_alignment_train(args, my_model, logger_)

    # if args.do_test:
    #     # load saved model
    #     if (not args.do_train) and (args.loaded_param_folder is not None):
    #         to_load_param_folder = args.loaded_param_folder
    #     else:
    #         to_load_param_folder = saved_model_folder
    #     if os.path.exists(os.path.join(to_load_param_folder, "params_best.ckpt")):
    #         # for EA
    #         my_model.load_state_dict(torch.load(os.path.join(to_load_param_folder, "params_best.ckpt")))

        # if args.task == "entity-alignment":
            # entity_alignment_test(args, my_model, logger_)
    return export_sim_mat

def run_structure_model(new_pairs):

    ########### set seed ###########
    torch.backends.cudnn.deterministic = True
    seed = 2020
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(seed)
    ################################

    args = get_args(new_pairs)
    print(args)
    prepare_entity_alignment_data(args, new_pairs)
    logger, stamp = get_logger(args)
    print_arguments(logger, args)
    return run(args, logger, stamp)
