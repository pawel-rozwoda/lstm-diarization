from config import LOG_MEL_WIDTH_MS, LOG_MEL_SHIFT_MS, DATA_PATH
import numpy as np
from tqdm import tqdm
import torch
import os
import matplotlib.pyplot as plt
from kaldi_feats import logmel_feats
from sklearn.cluster import spectral_clustering
from spectral_cluster import similarity, cluster_affinity, get_affinity_matrix
import pylangacq

def cha_to_wav_extension(cha_path_file):
    file_id = os.path.basename(cha_path_file)
    file_id = os.path.splitext(file_id)[0]
    return file_id + '.wav'


def get_char_stamps(utterances, audio, draw=False):
    """ return char utterances"""
    """ S char stand for Silence""" 
    
    ch0 = audio[0]
    ch1 = audio[1]
    mono = (ch0//2) + (ch1//2)
    stamps_a, stamps_b = get_timestamps(utterances, mono )

    bottom, top = get_bounds( first_last(stamps_a), first_last(stamps_b) )
    stamps_a_bounded = stamps_a[bottom:top]
    stamps_b_bounded = stamps_b[bottom:top]
    mono_bounded = mono[bottom:top]
    
    
    overlap_filtered = filtered_overlapping_indexes(stamps_a_bounded, stamps_b_bounded)
    stamps_a_filtered = stamps_a_bounded[overlap_filtered]
    stamps_b_filtered = stamps_b_bounded[overlap_filtered]
    mono_filtered = mono_bounded[overlap_filtered]
    
    """new"""
    stamps_chars = np.chararray(stamps_a_filtered.shape)
    stamps_chars.fill('S')
    stamps_chars[np.where(stamps_a_filtered)] = 'A'
    stamps_chars[np.where(stamps_b_filtered)] = 'B'
     
    
    if draw:
        ch0_bounded = ch0[bottom:top]
        ch1_bounded = ch1[bottom:top]
        
        ch0_filtered = ch0_bounded[overlap_filtered]
        ch1_filtered = ch1_bounded[overlap_filtered] 
#         draw_signal(ch0_filtered, 'ch0')

                
        draw_speech_and_channel(stamps_a_filtered[:300000], ch0_filtered[:300000], 'channel 0','green')
        draw_speech_and_channel(stamps_b_filtered[:300000], ch1_filtered[:300000], 'channel 1','purple')
        
    
    assert  np.logical_and(stamps_a_filtered, stamps_b_filtered).any()==False,'intersection of speech ' 

    return stamps_chars, mono_filtered 


def first_last(bool_arr):
    result = np.where(bool_arr)
    return result[0][0], result[0][-1]

def get_bounds(b1, b2):
    bottom=np.min([b1[0], b2[0]])
    top = np.max([b1[1], b2[1]])
    
    return bottom, top

def filtered_overlapping_indexes(labels1, labels2):
    return np.where(np.logical_not(np.logical_and(labels1, labels2))) 

def draw_speech_and_channel(speech, channel, title, color):
    plt.figure(figsize=(14, 5))
    plt.ylim(-1., 1.)
    plt.plot(channel/32768)
    plt.title(title)
    plt.fill_between(range(speech.shape[0]), speech * .8,color=color, alpha=0.7)
    plt.show()

def draw_signal(signal, title ):
    plt.figure(figsize=(14, 5))
    plt.ylim(-1., 1.)
    plt.plot(signal/32768)
    plt.title(title)
    plt.show()

def draw_speech_bool(speech, title, color):
    """ input is expected to be boolean array """
    plt.figure(figsize=(14, 5))
    plt.ylim(-1., 1.)
    plt.fill_between(range(speech.shape[0]), speech * .8,color=color, alpha=0.7)
    plt.title(title)
    plt.show()


def prepare_mono_for_forward(*, filtered_mono_channel, sampling_rate):
    feats = logmel_feats(signal=filtered_mono_channel, sampling_rate=sampling_rate)
    feats = torch.from_numpy(feats)
    feats = feats.unsqueeze(0)
    return feats
