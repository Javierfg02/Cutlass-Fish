# coding: utf-8
"""
Collection of helper functions
"""
import copy
import glob
import os
import errno
import shutil
import random
import logging
from logging import Logger
from typing import Callable, Optional, List
import numpy as np

import tensorflow as tf

import yaml

from dtw import dtw # TODO

class ConfigurationError(Exception):
    """ Custom exception for misspecifications of configuration """

def make_model_dir(model_dir: str, overwrite=False, model_continue=False) -> str:
    """
    Create a new directory for the model.

    :param model_dir: path to model directory
    :param overwrite: whether to overwrite an existing directory
    :param model_continue: whether to continue from a checkpoint
    :return: path to model directory
    """
    if os.path.isdir(model_dir):
        if model_continue:
            return model_dir
        if not overwrite:
            raise FileExistsError("Model directory exists and overwriting is disabled.")
        for file in os.listdir(model_dir):
            file_path = os.path.join(model_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        shutil.rmtree(model_dir, ignore_errors=True)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    return model_dir

def make_logger(model_dir: str, log_file: str = "train.log") -> Logger:
    """
    Create a logger for logging the training process.

    :param model_dir: path to logging directory
    :param log_file: path to logging file
    :return: logger object
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.DEBUG)
    fh = logging.FileHandler("{}/{}".format(model_dir, log_file))
    fh.setLevel(level=logging.DEBUG)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(message)s')
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    logging.getLogger("").addHandler(sh)
    logger.info("Progressive Transformers for End-to-End SLP")
    return logger

def log_cfg(cfg: dict, logger: Logger, prefix: str = "cfg") -> None:
    """
    Write configuration to log.

    :param cfg: configuration to log
    :param logger: logger that defines where log is written to
    :param prefix: prefix for logging
    """
    for k, v in cfg.items():
        if isinstance(v, dict):
            p = '.'.join([prefix, k])
            log_cfg(v, logger, prefix=p)
        else:
            p = '.'.join([prefix, k])
            logger.info("{:34s} : {}".format(p, v))

# TODO: MISSING CLONES, subsequent_mask, uneven_subsequent_mask
def set_seed(seed: int) -> None:
    """
    Set the random seed for modules tensorflow, numpy and random.

    :param seed: random seed
    """
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def load_config(path="configs/default.yaml") -> dict:
    """
    Loads and parses a YAML configuration file.

    :param path: path to YAML configuration file
    :return: configuration dictionary
    """
    with open(path, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg

def bpe_postprocess(string) -> str:
    """
    Post-processor for BPE output. Recombines BPE-split tokens.

    :param string:
    :return: post-processed string
    """
    return string.replace("@@ ", "")

def get_latest_checkpoint(ckpt_dir, post_fix="_every" ) -> Optional[str]:
    """
    Returns the latest checkpoint (by time) from the given directory, of either every validation step or best
    If there is no checkpoint in this directory, returns None

    :param ckpt_dir: directory of checkpoint
    :param post_fix: type of checkpoint, either "_every" or "_best"

    :return: latest checkpoint file
    """
    list_of_files = glob.glob("{}/*{}.ckpt".format(ckpt_dir,post_fix))
    latest_checkpoint = None
    if list_of_files:
        latest_checkpoint = max(list_of_files, key=os.path.getctime)
    return latest_checkpoint

def load_checkpoint(path: str, use_cuda: bool = True) -> dict:
    """
    Load model from saved checkpoint.

    :param path: path to checkpoint
    :param use_cuda: using cuda or not
    :return: checkpoint (dict)
    """
    assert os.path.isfile(path), "Checkpoint %s not found" % path
    if use_cuda:
        checkpoint = tf.train.load_checkpoint(path)
    else:
        checkpoint = tf.train.load_checkpoint(path)
    return checkpoint

def freeze_params(model) -> None:
    """
    Freeze the parameters of this model,
    i.e. do not update them during training

    :param model: freeze parameters of this model
    """
    for layer in model.layers:
        layer.trainable = False

def symlink_update(target, link_name):
    try:
        os.symlink(target, link_name)
    except FileExistsError as e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise e

def calculate_dtw(references, hypotheses):
    """
    Calculate the DTW costs between a list of references and hypotheses

    :param references: list of reference sequences to compare against
    :param hypotheses: list of hypothesis sequences to fit onto the reference

    :return: dtw_scores: list of DTW costs
    """
    euclidean_norm = lambda x, y: np.sum(np.abs(x - y))
    dtw_scores = []
    hypotheses = hypotheses[:, 1:]
    for i, ref in enumerate(references):
        _ , ref_max_idx = tf.reduce_max(ref[:, -1], axis=0)
        if ref_max_idx == 0: ref_max_idx += 1
        ref_count = ref[:ref_max_idx,:-1].numpy()
        hyp = hypotheses[i]
        _, hyp_max_idx = tf.reduce_max(hyp[:, -1], axis=0)
        if hyp_max_idx == 0: hyp_max_idx += 1
        hyp_count = hyp[:hyp_max_idx,:-1].numpy()
        d, cost_matrix, acc_cost_matrix, path = dtw(ref_count, hyp_count, dist=euclidean_norm)
        d = d/acc_cost_matrix.shape[0]
        dtw_scores.append(d)
    return dtw_scores
