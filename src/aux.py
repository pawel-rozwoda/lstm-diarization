import sklearn.mixture as mixture
import numpy as np
from config import VAD_SHIFT_MS, VAD_WIDTH_MS, LOG_MEL_WIDTH_MS, LOG_MEL_SHIFT_MS
from kaldi_feats import plp_feats
import librosa
import torch
from torch.nn.functional import normalize

def int16_to_float(signal):
    return (signal/32768.).astype(np.float32)

def load_audio(filepath, sampling_rate, mono):
    audio, sampling_rate = librosa.load(filepath, sr=sampling_rate, mono=mono) 
    return sampling_rate, (audio * 32768).astype(np.int16)

class Cluster:
    def __init__(self):
        self.gmm = None
        self.speech_index = None 

    def fit(self, signal, sampling_rate): 
        feats = plp_feats(signal=signal, sampling_rate=sampling_rate)
        gmm = mixture.GaussianMixture(n_components=2, covariance_type='spherical', random_state=2 ) #full, spherical, diag, tied
        self.gmm = gmm.fit(feats)
        means = self.gmm.means_
        self.speech_index = False if np.mean(means[0]) > np.mean(means[1]) else True
    
    def filter_non_speech(self,* ,  signal, sampling_rate):
        speech = self.detect_speech(signal=signal, sampling_rate=sampling_rate, fit_to_audio=True)
        return signal[np.where(speech)]

    def detect_speech(self,* , signal, sampling_rate, fit_to_audio=False):
        labels = None
        if self.gmm is not None:
            feats = plp_feats(signal=signal, sampling_rate=sampling_rate)
            """gmm cant predict 180k feats at once"""
            if feats.shape[0]>int(1e5) :
                log_mel_arrays = [self.gmm.predict(feats[i: i+ int(1e5)]).astype(np.bool) for i in range(0,feats.shape[0], int(1e5))]
                labels = np.concatenate(log_mel_arrays, axis=0)
                # print('labels: ',labels.shape)
            else:
                labels = self.gmm.predict(feats).astype(np.bool)

        else:
            print('gmm is not fit yet')
            
        if not self.speech_index:
            labels = ~labels


        if fit_to_audio:
            """trim to seconds"""
            num_feats = int(signal.shape[0] //(sampling_rate*(VAD_SHIFT_MS/1e3)))
            # print('num_feats ', num_feats)
            # print('num_labels ', labels.shape)

            """fill beginning with first value n-times"""

            labels = np.repeat(labels, (sampling_rate * (VAD_SHIFT_MS/1e3) ))
            last_value = labels[-1]
            boundary_count = (signal.shape[0] - labels.shape[0])
            # print('boundary count ',boundary_count)

            left_boundary=0
            right_boundary=0
            if boundary_count%2==0:
                left_boundary = boundary_count//2
                right_boundary = boundary_count//2
            else:
                left_boundary = boundary_count//2 + 1
                right_boundary = boundary_count//2

            # print('left', left_boundary)
            # print('right', right_boundary)
            last_value = labels[-1]
            first_value = labels[0]
            for i in range(right_boundary):
                labels = np.insert(labels, labels.shape[0] - 1, last_value, axis=0) 

            for i in range(left_boundary):
                labels = np.insert(labels, 0, first_value, axis=0) 

        return labels
    """vad usage example"""

    # _, gmm_audio = load_audio(GMM_WAV,sampling_rate, mono=True/False)
    # vad_gmm = Cluster()
    # vad_gmm.fit(stereo_to_mono(gmm_audio[0], gmm_audio[1]))

    #OR

    
    """from file setup"""
    # vad_gmm=None
    # with open(DATA_PATH + 'gmm/gmm.pkl', 'rb') as fid:
        # vad_gmm = pickle5.load(fid)



def get_embeddings(d_vectors):
    """from aux_evaluate.py"""
    """ shape -> (batch, emb_count, emb_dim) """
    """performs norm, stack and average on d_vects"""
    stack_size = (LOG_MEL_WIDTH_MS//LOG_MEL_SHIFT_MS)
    
    last_vects_count = (d_vectors.shape[1] -1)% stack_size
    embeddings = torch.zeros((d_vectors.shape[0], 1+(d_vectors.shape[1]//stack_size), d_vectors.shape[2])).to(d_vectors.device) 
    """pure"""
    
    
    """norm"""
    norm = torch.norm(d_vectors, dim=2).unsqueeze(2)

    d_vectors = d_vectors/norm
    
    
    """create aux_arr that consist of 1 : -last_vects_count size of d_vects"""
    aux_arr = None
    if not last_vects_count:
        aux_arr = d_vectors[:, 1:] 
    else:
        aux_arr = d_vectors[:, 1:-last_vects_count]

    
    """stack"""
    aux_arr = aux_arr.reshape(d_vectors.shape[0], (d_vectors.shape[1]//stack_size) - last_vects_count, stack_size, aux_arr.shape[2])
    
    
    """average"""
    aux_arr = aux_arr.mean(dim=2)
    embeddings[:, 0] = normalize(d_vectors[:, 0])

    if not last_vects_count:
        embeddings[:, 1:] = aux_arr
        
        
    else:
        # print('last ', last_vects_count)
        # print('emb shape ', embeddings.shape)
        # print('dv shape ', d_vectors.shape)
        # print('into shape ', embeddings[:, -last_vects_count:].shape)
        # print('operation shape  ', d_vectors[:, -last_vects_count:].shape)
        # print('operation 1  ', d_vectors[:, -last_vects_count:].mean(1).shape)
        # print('operation 2  ', d_vectors[:, -last_vects_count:].mean(2).shape)
        embeddings[:, 1:-last_vects_count] = aux_arr
        # embeddings[:, -last_vects_count:] = d_vectors[:, -last_vects_count:].mean(dim=0)
        embeddings[:, -last_vects_count:] = d_vectors[:, -last_vects_count:].mean(dim=1).unsqueeze(1)
        
    
    
    return embeddings


