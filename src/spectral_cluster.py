import torch
import numpy as np
from tqdm import tqdm
from scipy.ndimage.filters import gaussian_filter
from sklearn.cluster import spectral_clustering
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from config import debug_mode

""" sequence as follows """

"""
similarity
blur
row_wise_threshold
symmetrize
diffuse
row_wise_normalize

""" 
 
class GaussianBlurConv(nn.Module):
    def __init__(self, channels=1):
        super(GaussianBlurConv, self).__init__()
        self.channels = channels
        kernel = [[0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633],
                  [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
                  [0.01330373, 0.11098164, 0.22508352, 0.11098164, 0.01330373],
                  [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
                  [0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
 
    def __call__(self, x):
        x = F.conv2d(x, self.weight, padding=2, groups=self.channels)
        return x

def similarity(d_vectors):
    """ input i.e (14000, 256) """
    nominator = torch.mm(d_vectors, d_vectors.T) 

    lengths = d_vectors.norm(dim=-1)
    denominator = torch.mm(lengths.unsqueeze(0).T, lengths.unsqueeze(0))
    return nominator/(denominator + 1e-6)


def blur(s_matrix): 
    gaussian_conv = GaussianBlurConv().to(s_matrix.device)
    return gaussian_conv(s_matrix.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)


def row_wise_threshold(*, s_matrix, percentile):
    """in place""" 
    for i in range(s_matrix.shape[0]):
        percentile_value = np.percentile(s_matrix[i].detach().cpu().numpy(), percentile)
        indexes = s_matrix[i]<=percentile_value 
        s_matrix[i][indexes] *= 0.01 
                
    return s_matrix


def symmetrize(s_matrix):
    """new array"""
    return torch.max( s_matrix, s_matrix.T)

def diffuse(s_matrix):
    return torch.mm(s_matrix, s_matrix.T)

def row_wise_normalize(s_matrix):
    aux = torch.max(s_matrix, dim=1).values.unsqueeze(0).T.to(s_matrix.device)
    return s_matrix/aux 

    
def cluster_affinity(affinity, dtype=np.bool):
    labels =  spectral_clustering(affinity, n_clusters=2, n_components=2, random_state=0) 
    # labels =  spectral_clustering(affinity, n_clusters=2, n_components=2, affinity='precomputed', random_state=0) 
    # labels =  spectral_clustering(n_clusters=2, n_components=2, affinity='precomputed', random_state=0) 
    if dtype == np.bool:
        speakers = np.unique(labels)
        result = np.zeros((speakers.shape[0], labels.shape[0]), dtype=np.bool)
        for spk in speakers:
            result[spk][np.where(labels==spk)] = True
        
        return result

    else:
        return labels

def get_affinity_matrix(embeddings):
    s = similarity(embeddings)
    s = blur(s)
    s = row_wise_threshold(s_matrix=s, percentile=10.)
    s = symmetrize(s)
    s = diffuse(s)
    s = row_wise_normalize(s)

    return s

def adjust_labels_to_signal(labels, destination_size):
    """ this method make labels and stretch them to shape of signal """
    """labels are clustered embeddings"""
    """currently for 2 speakers"""
    
    count = destination_size//labels.shape[0]
    remaining = destination_size % labels.shape[0]
    stretched_labels = None
    if labels.dtype == np.dtype('|S1'):
        stretched_labels = np.chararray(destination_size) 
    else:
        stretched_labels = np.zeros((destination_size,), dtype=labels.dtype)

    if remaining:
        stretched_labels[:-remaining] = np.repeat(labels, count)
        stretched_labels[-remaining:] = np.repeat(labels[-1], remaining)
    else:
        stretched_labels[:] = np.repeat(labels, count)

    return stretched_labels


def arr_to_areas(arr, sign):
    areas = []
    from_idx = 0
    flag=False
    for i in range(arr.shape[0]):
   
        if arr[i] == bytes(sign, 'utf8'):
            if flag == False:
                flag = True
                from_idx = i      
        else:
            if flag:
                areas.append((from_idx, i))
                flag=False
                
    if i == arr.shape[0] - 1 and flag == True:
            areas.append((from_idx, i))
            
    return areas

def get_char_labels(labels, mono_audio, speech_indexes):
  
    char_array = np.chararray(labels.shape[0])
    char_array[np.where(labels==True)] = 'A'
    char_array[np.where(labels==False)] = 'B' 
    
    stretched = adjust_labels_to_signal(char_array, np.sum(speech_indexes))
    
    result = np.chararray(mono_audio.shape[0])
    
    result[np.where(speech_indexes)] = stretched
    
    result[np.where(~speech_indexes)] = 'S'
    
    return result
