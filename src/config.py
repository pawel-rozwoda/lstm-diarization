""" 
data structure:
    wav | aac

"""

debug_mode=False

from data_path import DATA_PATH 

MODEL_FILE = DATA_PATH + 'model/model.pt'
TEST_PATH = DATA_PATH + 'test/'
OUT_TRAIN = DATA_PATH + 'train/'
OUT_EVAL = DATA_PATH + 'evaluate/'

VOX_1_TEST_PATH = DATA_PATH + 'vox_1_test_data/'
VOX_1_PATH = DATA_PATH + 'vox_1_data/'
VOX_2_PATH = DATA_PATH + 'vox_2_data/'
GMM_FILE = DATA_PATH + 'gmm/final_gmm.pkl'
GMM_TRAIN = DATA_PATH + 'vox_1_test/'


EVAL_WAV_DIRECTORY = DATA_PATH + 'callhome/wav/'
EVAL_CHA_DIRECTORY = DATA_PATH + 'callhome/eng/'

CALLHOME_ENG_10_SEC = DATA_PATH + 'CALLHOME_ENG_10_SEC/' #10 seconds files

"""Model-related"""
LOG_MEL_FILTERS = 40
INPUT_DIM = LOG_MEL_FILTERS
HIDDEN_DIM = 756
NUM_LAYERS = 3

""" audio preprocessing """
VAD_SHIFT_MS = 10
VAD_WIDTH_MS = 120

LOG_MEL_SHIFT_MS = 10
LOG_MEL_WIDTH_MS = 25

"""dataset"""
SEQ_LEN = 10
