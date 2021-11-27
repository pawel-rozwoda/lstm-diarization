import sys
import shutil
import os
sys.path.append('../')
sys.path.append('../train')
from tqdm import tqdm
import torch
import pickle5
from aux import load_audio
from config import DATA_PATH, GMM_FILE, OUT_EVAL, OUT_TRAIN, EVAL_WAV_DIRECTORY, EVAL_CHA_DIRECTORY, MODEL_FILE
from aux_evaluate import cha_to_wav_extension, prepare_mono_for_forward
import pylangacq
from ChaTool import ChaTool, get_speakers_distribution, Hypothesis, Reference, CharLabels
from kaldi_feats import logmel_feats
from config import GMM_FILE
from model import LSTM_Diarization
import pickle5
import gc
from spectral_cluster import get_affinity_matrix, cluster_affinity, arr_to_areas, get_char_labels, get_char_labels
import numpy as np
from datetime import datetime

from pyannote.core import Segment, Timeline, Annotation
from pyannote.metrics.diarization import DiarizationErrorRate
import warnings
warnings.filterwarnings("ignore")
dt = datetime.now()
output_dir = OUT_EVAL + dt.strftime('%m%d_') + dt.strftime('%H:%M') + '/'  
print('out dir ', output_dir)



vad=None                                                                                                                                                    
with open(GMM_FILE, 'rb') as fid:                                                                                                             
    vad = pickle5.load(fid) 

model_device = 'cpu'
affinity_device='cpu'
model = torch.load(MODEL_FILE, map_location=model_device)
model.window_size=40
model.shift=20
model.train=False

wav_directory = EVAL_WAV_DIRECTORY
cha_directory = EVAL_CHA_DIRECTORY

cha_files = [f for f in os.listdir(cha_directory) if f.endswith('.' + 'cha')]

import pandas as pd
import warnings 

total=0
missed_detection=0
confusion=0
correct=0
false_alarm=0

df = pd.DataFrame(columns=['filename', 'false alarm', 'total', 'missed detection', 'correct', 'confusion', 'diarization error rate'])

metric = DiarizationErrorRate()
sampling_rate=8000
counter = 0
for cha_file in tqdm(cha_files, leave=True):
# for cha_file in ['4289.cha']: # example of total silence on audio_splits 
    print(cha_file)
    wav_file = cha_to_wav_extension(cha_file)
    chatool = ChaTool(cha_path_file=cha_directory + cha_file,
                        wav_path_file=wav_directory + wav_file,
                        sampling_rate=sampling_rate)

    if chatool.utterances:
        counter += 1
        stamps_chars, mono_filtered = chatool.stamps_and_mono() 
        speech_indexes = vad.detect_speech(signal=mono_filtered, sampling_rate=sampling_rate, fit_to_audio=True)
        if speech_indexes.sum():
            audio_filtered = mono_filtered[speech_indexes]
            feats = prepare_mono_for_forward(filtered_mono_channel=mono_filtered, sampling_rate=sampling_rate)
            d_vectors = model.forward(feats)
            affinity = get_affinity_matrix(d_vectors.squeeze(0).to(affinity_device))

            pred_labels = cluster_affinity(affinity.detach().cpu().numpy())

            labels = CharLabels(pred_labels, mono_filtered.shape[0], speech_indexes)
            char_stamps = labels.char_labels
            hypothesis = Hypothesis(char_stamps).hypothesis
            reference = Reference(stamps_chars).reference

            metric_result = metric(reference, hypothesis, detailed=True)
            metric_result['filename'] = cha_file
            df = df.append(metric_result, ignore_index=True)
            print(metric_result)


if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open(output_dir + 'info.txt', "w") as myfile:
    myfile.write('model: ' +  MODEL_FILE) 
    myfile.write('width: ' +  str(model.window_size)) 
    myfile.write('shift: ' +  str(model.shift)) 

last_dir = OUT_EVAL + 'last/'
if os.path.exists(last_dir) and os.path.isdir(last_dir):
    shutil.rmtree(last_dir) 

os.makedirs(OUT_EVAL + 'last/')
with open(OUT_EVAL + 'last/' + 'info.txt', "w") as myfile:
    myfile.write('model: ' +  MODEL_FILE) 
    myfile.write('width: ' +  str(model.window_size)) 
    myfile.write('shift: ' +  str(model.shift)) 
df.to_csv(OUT_EVAL + 'last/' + 'result.csv')

print(counter , ' recordings contain corresponding annotations')
print('saving result.csv')
df.to_csv(output_dir + 'result.csv')
