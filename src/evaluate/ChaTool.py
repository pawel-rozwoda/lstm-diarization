import sys
sys.path.append('../')
from config import DATA_PATH
import pylangacq
from aux import load_audio
from aux_evaluate import first_last, get_bounds, filtered_overlapping_indexes, prepare_mono_for_forward
import os
import numpy as np
from spectral_cluster import get_affinity_matrix, cluster_affinity, adjust_labels_to_signal, arr_to_areas
from pyannote.core import Segment, Timeline, Annotation
from pyannote.metrics.diarization import DiarizationErrorRate


def get_speakers_distribution(stamps):
    a =  stamps.count(b'A').sum()
    b =  stamps.count(b'B').sum()
    silence = stamps.count(b'S').sum()
    return {'A':a, 'B':b, 'silence':silence}


def first_last(bool_arr):
    result = np.where(bool_arr)
    return result[0][0], result[0][-1]

def get_bounds(b1, b2):
    bottom=np.min([b1[0], b2[0]])
    top = np.max([b1[1], b2[1]])
    
    return bottom, top

def filtered_overlapping_indexes(labels1, labels2):
    assert labels1.shape == labels2.shape
    return np.where(np.logical_not(np.logical_and(labels1, labels2)))

class ChaTool:
    def __init__(self, *, cha_path_file, wav_path_file, sampling_rate):
        self.sampling_rate = sampling_rate
        self.wav_filepath = wav_path_file
        _, self.mono_audio = load_audio(self.wav_filepath, sampling_rate, mono=True)
        reader = pylangacq.Reader.from_files([cha_path_file])
        self.utterances = reader.utterances()


    def get_timestamps(self):
        time_stamps_A = [utt.time_marks for utt in self.utterances if utt.participant=='A' and utt.time_marks is not None]
        time_stamps_B = [utt.time_marks for utt in self.utterances if utt.participant=='B' and utt.time_marks is not None]
        
        time_stamps_a = list(map(np.array, time_stamps_A))
            
        time_stamps_b = list(map(np.array, time_stamps_B))
        
        stamps_a = np.zeros_like(self.mono_audio, dtype=np.bool)
        stamps_b = np.zeros_like(self.mono_audio, dtype=np.bool)
            
        coeff = self.sampling_rate/1000
        
        for s in time_stamps_a:
            stamps_a[int(s[0]*coeff): int(s[1]*coeff)] = True
            
        for s in time_stamps_b:
            stamps_b[int(s[0]*coeff): int(s[1]*coeff)] = True

        return stamps_a, stamps_b

    def stamps_and_mono(self):
        stamps_a, stamps_b = self.get_timestamps()
        bottom, top = get_bounds( first_last(stamps_a), first_last(stamps_b) )
        stamps_a_bounded = stamps_a[bottom:top]
        stamps_b_bounded = stamps_b[bottom:top]
        mono_bounded = self.mono_audio[bottom:top]
        
        
        overlap_filtered = filtered_overlapping_indexes(stamps_a_bounded, stamps_b_bounded)
        stamps_a_filtered = stamps_a_bounded[overlap_filtered]
        stamps_b_filtered = stamps_b_bounded[overlap_filtered]
        mono_filtered = mono_bounded[overlap_filtered]
        
        """new"""
        stamps_chars = np.chararray(stamps_a_filtered.shape)
        stamps_chars.fill('S')
        stamps_chars[np.where(stamps_a_filtered)] = 'A'
        stamps_chars[np.where(stamps_b_filtered)] = 'B'


        assert  np.logical_and(stamps_a_filtered, stamps_b_filtered).any()==False,'intersection of speech ' 

        return stamps_chars, mono_filtered

class CharLabels:
    def __init__(self, labels, destination_size, speech_indexes ):

        char_array = np.chararray(labels.shape[1])
        char_array[np.where(labels[0])] = 'A'
        char_array[np.where(labels[1])] = 'B'
        
        stretched = adjust_labels_to_signal(char_array, np.sum(speech_indexes))
        
        result = np.chararray(destination_size)
        
        result[np.where(speech_indexes)] = stretched
        
        result[np.where(~speech_indexes)] = 'S'

        self.char_labels = result

class Hypothesis:
    def __init__(self, char_labels):
        hypothesis = Annotation()
        """pyannote.metrics does not require to have same label names"""
        a_areas_hypothesis = arr_to_areas(char_labels, 'A')
        b_areas_hypothesis = arr_to_areas(char_labels, 'B')


        for elem_a, elem_b in zip(a_areas_hypothesis, b_areas_hypothesis):
            hypothesis[Segment(elem_a[0], elem_a[1])] = 'A'
            hypothesis[Segment(elem_b[0], elem_b[1])] = 'B'

        self.hypothesis = hypothesis


class Reference:
    def __init__(self, char_stamps):

        reference = Annotation()
        areas_a = arr_to_areas(char_stamps, 'A')
        areas_b = arr_to_areas(char_stamps, 'B')

        reference = Annotation()

        for elem in areas_a:
            reference[Segment(elem[0], elem[1])] = 'A'
        for elem in areas_b:
            reference[Segment(elem[0], elem[1])] = 'B'

        self.reference = reference 
