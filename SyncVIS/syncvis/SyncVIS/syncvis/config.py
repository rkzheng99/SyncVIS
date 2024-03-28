# -*- coding: utf-8 -*-
from detectron2.config import CfgNode as CN


def add_syncvis_config(cfg):
    cfg.DATASETS.DATASET_RATIO = []

    # DataLoader
    cfg.INPUT.SAMPLING_FRAME_NUM = 2
    cfg.INPUT.SAMPLING_FRAME_RANGE = 20
    cfg.INPUT.SAMPLING_FRAME_SHUFFLE = False
    cfg.INPUT.AUGMENTATIONS = [] # "brightness", "contrast", "saturation", "rotation"

    # Pseudo Data Use
    cfg.INPUT.PSEUDO = CN()
    cfg.INPUT.PSEUDO.AUGMENTATIONS = ['rotation']
    cfg.INPUT.PSEUDO.MIN_SIZE_TRAIN = (480, 512, 544, 576, 608, 640, 672, 704, 736, 768)
    cfg.INPUT.PSEUDO.MAX_SIZE_TRAIN = 768
    cfg.INPUT.PSEUDO.MIN_SIZE_TRAIN_SAMPLING = "choice_by_clip"
    cfg.INPUT.PSEUDO.CROP = CN()
    cfg.INPUT.PSEUDO.CROP.ENABLED = False
    cfg.INPUT.PSEUDO.CROP.TYPE = "absolute_range"
    cfg.INPUT.PSEUDO.CROP.SIZE = (384, 600)

    # LSJ
    cfg.INPUT.LSJ_AUG = CN()
    cfg.INPUT.LSJ_AUG.ENABLED = False
    cfg.INPUT.LSJ_AUG.IMAGE_SIZE = 1024
    cfg.INPUT.LSJ_AUG.MIN_SCALE = 0.1
    cfg.INPUT.LSJ_AUG.MAX_SCALE = 2.0

    # syncvis
    cfg.MODEL.syncvis = CN()
    cfg.MODEL.syncvis.NHEADS = 8
    cfg.MODEL.syncvis.DROPOUT = 0.0
    cfg.MODEL.syncvis.DIM_FEEDFORWARD = 2048
    cfg.MODEL.syncvis.ENC_LAYERS = 6
    cfg.MODEL.syncvis.DEC_LAYERS = 3
    cfg.MODEL.syncvis.ENC_WINDOW_SIZE = 0
    cfg.MODEL.syncvis.PRE_NORM = False
    cfg.MODEL.syncvis.HIDDEN_DIM = 256
    cfg.MODEL.syncvis.NUM_OBJECT_QUERIES = 100
    cfg.MODEL.syncvis.ENFORCE_INPUT_PROJ = True

    cfg.MODEL.syncvis.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.syncvis.DEEP_SUPERVISION = True
    cfg.MODEL.syncvis.LAST_LAYER_NUM = 3
    cfg.MODEL.syncvis.MULTI_CLS_ON = True
    cfg.MODEL.syncvis.APPLY_CLS_THRES = 0.01

    cfg.MODEL.syncvis.SIM_USE_CLIP = True
    cfg.MODEL.syncvis.SIM_WEIGHT = 0.5

    cfg.MODEL.syncvis.FREEZE_DETECTOR = False
    cfg.MODEL.syncvis.TEST_RUN_CHUNK_SIZE = 18
    cfg.MODEL.syncvis.TEST_INTERPOLATE_CHUNK_SIZE = 5
