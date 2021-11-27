from aux_evaluate import get_char_stamps
import sys
from config import TEST_PATH
sys.path.append('evaluate/')
import torch
from spectral_cluster import similarity, row_wise_threshold, symmetrize, diffuse, row_wise_normalize
# from ChaTool import ChaTool

# def test_get_char_stamps():
    # cha_file = TEST_PATH + '4065.cha'
    # chatool = ChaTool(cha_path_file=cha_file, sampling_rate=8000, wav_directory=EVAL_WAV_DIRECTORY, cha_directory=EVAL_CHA_DIRECTORY)
    # pass


def test_similarity():
    x = torch.Tensor([i-4 for i in range(8)]).reshape(4,2)
    s = similarity(x)
    print(s)
    print(x)
    
def test_row_wise_threshold():
    x = torch.Tensor([[0.2, 0.31, 0.4],
                        [0.5, 0.4, 0.31],
                        [0.3, 0.4, 0.3]])
    print(x)
    print(row_wise_threshold(s_matrix=x, percentile=40))

test_similarity()
test_row_wise_threshold()

