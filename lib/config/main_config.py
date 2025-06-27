import logging
logger = logging.getLogger(__name__)

from .config import Config, FieldStatus


class MainConfig(Config):
    def init_fields(self):
        Required = FieldStatus.Required # corresponding config.yaml file should contain this field
        Optional = FieldStatus.Optional
        Deprecated = FieldStatus.Deprecated
        Auto = FieldStatus.Auto # can be automatically set by the code
        Reserved = FieldStatus.Reserved # should not be specified in the config file
        NotImplemented = FieldStatus.NotImplemented # not implemented yet
        
        GENERAL = Config()
        GENERAL.LOCAL_RANK = Reserved # set by torch.distributed.launch
        GENERAL.DEVICE = Reserved
        GENERAL.DEBUG = False
        GENERAL.WORK_DIR = Auto
        GENERAL.CONFIG_NAME = Auto
        GENERAL.TITLE = Required
        GENERAL.BASE_SEED = 0 # Note: seeds in different processes will be different based on this.
        GENERAL.CUDNN_BENCHMARK = True
        GENERAL.USE_LMDB = False
        GENERAL.USE_WANDB = False
        GENERAL.DESCRIPTION = ""
        GENERAL.MOVE_DATA_TO_GPU = True
        GENERAL.PRINT_STATS = None # legacy code. still remains unknown.
        # Note: we do not specify logfile. 
        # Future support:
        GENERAL.DISTILL = Config()
        GENERAL.DISTILL.ENABLED = False
        GENERAL.DISTILL.SCRIPT_TEACHER = Optional
        GENERAL.DISTILL.CONFIG_TEACHER = Optional
        self.GENERAL = GENERAL
        
        MODEL = Config()
        MODEL.PRETRAIN_FILE = ""
        MODEL.EXTRA_MERGER = False
        MODEL.RETURN_INTER = False
        MODEL.RETURN_STAGES = []

        # MODEL.BACKBONE
        MODEL.BACKBONE = Config()
        MODEL.BACKBONE.TYPE = "vit_base_patch16_224"
        MODEL.BACKBONE.STRIDE = 16
        MODEL.BACKBONE.MID_PE = False
        MODEL.BACKBONE.SEP_SEG = False
        MODEL.BACKBONE.CAT_MODE = 'direct'
        MODEL.BACKBONE.MERGE_LAYER = 0
        MODEL.BACKBONE.ADD_CLS_TOKEN = False
        MODEL.BACKBONE.CLS_TOKEN_USE_MODE = 'ignore'

        MODEL.BACKBONE.CE_LOC = []
        MODEL.BACKBONE.CE_KEEP_RATIO = []
        MODEL.BACKBONE.CE_TEMPLATE_RANGE = 'ALL'  # choose between ALL, CTR_POINT, CTR_REC, GT_BOX

        # MODEL.HEAD
        MODEL.HEAD = Config()
        MODEL.HEAD.TYPE = "CENTER"
        MODEL.HEAD.NUM_CHANNELS = 256
        self.MODEL = MODEL

        # TRAIN
        TRAIN = Config()
        TRAIN.PROMPT = Config()
        TRAIN.PROMPT.TYPE = 'vipt_deep'  # vipt_deep vipt_shaw
        TRAIN.LR = 0.0001
        TRAIN.WEIGHT_DECAY = 0.0001
        TRAIN.EPOCH = 500
        TRAIN.LR_DROP_EPOCH = 400
        TRAIN.BATCH_SIZE = 16
        TRAIN.NUM_WORKER = 8
        TRAIN.OPTIMIZER = "ADAMW"
        TRAIN.BACKBONE_MULTIPLIER = 0.1
        TRAIN.GIOU_WEIGHT = 2.0
        TRAIN.L1_WEIGHT = 5.0
        TRAIN.FREEZE_LAYERS = [0, ]
        TRAIN.PRINT_INTERVAL = 50
        TRAIN.VAL_EPOCH_INTERVAL = 20
        TRAIN.GRAD_CLIP_NORM = 0.1
        TRAIN.AMP = False
        ## TRAIN save cfgs
        TRAIN.FIX_BN = True
        TRAIN.SAVE_EPOCH_INTERVAL = 1 # 1 means save model each epoch
        TRAIN.SAVE_LAST_N_EPOCH = 1 # besides, last n epoch model will be saved

        TRAIN.CE_START_EPOCH = 20  # candidate elimination start epoch
        TRAIN.CE_WARM_EPOCH = 80  # candidate elimination warm up epoch
        TRAIN.DROP_PATH_RATE = 0.1  # drop path rate for ViT backbone

        # TRAIN.SCHEDULER
        TRAIN.SCHEDULER = Config()
        TRAIN.SCHEDULER.TYPE = "step"
        TRAIN.SCHEDULER.DECAY_RATE = 0.1
        self.TRAIN = TRAIN

        # DATA
        DATA = Config()
        DATA.SAMPLER_MODE = "causal"  # sampling methods
        DATA.MEAN = [0.485, 0.456, 0.406]
        DATA.STD = [0.229, 0.224, 0.225]
        DATA.MAX_SAMPLE_INTERVAL = 200
        # DATA.TRAIN
        DATA.TRAIN = Config()
        DATA.TRAIN.DATASETS = []
        # DATA.TRAIN.DATASETS.append(dict(
        #     NAME = "LasHeR_train",
        #     PATH = "datasets/LasHeR/TrainingSet/trainingset",
        #     RATIO = 1.0,
        # ))
        DATA.TRAIN.SAMPLE_PER_EPOCH = 60000
        # DATA.VAL
        DATA.VAL = Config()
        DATA.VAL.DATASETS = []
        DATA.VAL.SAMPLE_PER_EPOCH = 10000
        # DATA.TEST
        DATA.TEST = Config()
        DATA.TEST.DATASETS = []
        # DATA.SEARCH
        DATA.SEARCH = Config()
        DATA.SEARCH.SIZE = 320
        DATA.SEARCH.FACTOR = 5.0
        DATA.SEARCH.CENTER_JITTER = 4.5
        DATA.SEARCH.SCALE_JITTER = 0.5
        DATA.SEARCH.NUMBER = 1
        # DATA.TEMPLATE
        DATA.TEMPLATE = Config()
        DATA.TEMPLATE.NUMBER = 1
        DATA.TEMPLATE.SIZE = 128
        DATA.TEMPLATE.FACTOR = 2.0
        DATA.TEMPLATE.CENTER_JITTER = 0
        DATA.TEMPLATE.SCALE_JITTER = 0
        self.DATA = DATA

        # TEST
        TEST = Config()
        TEST.TEMPLATE_FACTOR = 2.0
        TEST.TEMPLATE_SIZE = 128
        TEST.SEARCH_FACTOR = 5.0
        TEST.SEARCH_SIZE = 320
        TEST.EPOCH = 500
        TEST.INPUT_VIDEO = NotImplemented
        TEST.CHECKPOINT = Auto
        TEST.METRICS = []
        TEST.ONLY_METRICS = False # only evaluate metrics without inferrence
        TEST.SAVE_ALL_BOXES = False # save all boxes for each frame
        self.TEST = TEST
        
