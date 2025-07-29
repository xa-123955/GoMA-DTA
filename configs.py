from yacs.config import CfgNode as CN

_C = CN()

# GoMADTA model settings

_C.CONV_CHANNELS = [[768, 512, 3],
                    [512, 256, 3],
                    [640, 512, 3],
                    [512, 256, 5],
                    [768, 512, 3],
                    [512, 256, 3]]
_C.FUSION = CN()
_C.FUSION.HIDDEN1 = 256
_C.FUSION.HIDDEN2 = 128
_C.FUSION.DROPOUT = 0.1

_C.ATTN = CN()
_C.ATTN.DIM = 256
_C.ATTN.HEADS = 2

_C.DIM = CN()
_C.DIM.MAMBA = 256
_C.DIM.LINEAR = 256

_C.GRAPH_ENCODER = CN()
_C.GRAPH_ENCODER.ENCODER_TYPE = "TransConv"
_C.GRAPH_ENCODER.LAYER_NUM = 10
_C.GRAPH_ENCODER.IN_CHANNELS = 74
_C.GRAPH_ENCODER.BASIC_CHANNELS = 32
_C.GRAPH_ENCODER.DROPOUT = 0.3

_C.CLASSIFIER = CN()
_C.CLASSIFIER.HIDDEN1 = 256
_C.CLASSIFIER.HIDDEN2 = 128
_C.CLASSIFIER.HIDDEN3 = 64
_C.CLASSIFIER.DROPOUT = 0.1


# SOLVER
_C.SOLVER = CN()
_C.SOLVER.MAX_EPOCH = 100
_C.SOLVER.BATCH_SIZE = 64
_C.SOLVER.NUM_WORKERS = 4
_C.SOLVER.LR = 5e-5
_C.SOLVER.SEED = 2048

# RESULT
_C.RESULT = CN()
_C.RESULT.OUTPUT_DIR = "./result"
_C.RESULT.SAVE_MODEL = True

# Comet config, ignore it If not installed.
_C.COMET = CN()
# Please change to your own workspace name on comet.
_C.COMET.WORKSPACE = "pz-white"
_C.COMET.PROJECT_NAME = "GoMA-DTA"
_C.COMET.USE = False
_C.COMET.TAG = None

def get_cfg_defaults():
    return _C.clone()
