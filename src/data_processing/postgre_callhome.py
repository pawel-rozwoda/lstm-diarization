"""
This script connects to database and stores MFCC feats of CALLHOME dataset which is used as test set in this project.
"""
import sys
sys.path.append('../')
from db_credentials import POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB, POSTGRES_PORT, POSTGRES_HOST
from config import DATA_PATH
import pandas as pd
import os
import shutil
import pickle5
from config import GMM_FILE
from aux import load_audio
from tqdm import tqdm
import torch
import numpy as np
from kaldi_feats import logmel_feats
from multiprocessing import cpu_count, Pool
from tqdm import tqdm
import warnings
import os
import argparse
from psycopg2.extensions import AsIs
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT


parser = argparse.ArgumentParser()

parser.add_argument("--threads", type=int, default=1)
parser.add_argument("--data_source", type=str)
args = parser.parse_args()
db_name = args.data_source[:-1]

print("Database opened successfully")
con = psycopg2.connect(database=POSTGRES_DB, user=POSTGRES_USER, password=POSTGRES_PASSWORD, host=POSTGRES_HOST, port=POSTGRES_PORT) 
con.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
cursor = con.cursor()

cursor.execute('drop database if exists '+ db_name )
cursor.execute('create database '+ db_name)
con.commit()
con.close()

data_source = DATA_PATH + args.data_source
warnings.filterwarnings("ignore")
print('PID: ', os.getpid()) 

extensions = os.listdir(data_source)
audio_type = None
if 'aac' in extensions:
    audio_type = 'aac/' 
elif 'wav' in extensions:
    audio_type = 'wav/'
else:
    sys.exit('no appropriate audio type')

SOURCE = data_source + audio_type +  '/'
DESTINATION = data_source + 'intermediate/'
sampling_rate=8000

print('src: ', SOURCE)
print('destination: ',DESTINATION)
print('audio type', audio_type)
print('threads: ', args.threads)
print('sampling_rate: ', sampling_rate)


print('data processing on: ', SOURCE)

def process_callhome_talk(filename): 

    base=os.path.basename(filename)
    base = os.path.splitext(base)[0]
    con = psycopg2.connect(database=db_name, user=POSTGRES_USER, password=POSTGRES_PASSWORD, host=POSTGRES_HOST, port=POSTGRES_PORT)
    cur = con.cursor()

    """vad_gmm setup"""
    vad_gmm=None
    with open(GMM_FILE, 'rb') as fid:
        # print('reading gmm file from ', GMM_FILE)
        vad_gmm = pickle5.load(fid) 
    """end setup"""

    table_name_0 = 'MFCC_' + base + '_a'
    table_name_1 = 'MFCC_'+ base + '_b'
    DROP_COMMAND_0 = 'drop table if exists ' + table_name_0
    DROP_COMMAND_1 = 'drop table if exists ' + table_name_1
    cur.execute(DROP_COMMAND_0)
    cur.execute(DROP_COMMAND_1)
    con.commit()
    cur.execute('''CREATE TABLE %(table_name)s
          (ID SERIAL PRIMARY KEY,
          feats float[10][40]);''', {"table_name":AsIs(table_name_0) })

    cur.execute('''CREATE TABLE %(table_name)s
          (ID SERIAL PRIMARY KEY,
          feats float[10][40]);''', {"table_name":AsIs(table_name_1) })
    con.commit()

    
    _, audio = load_audio(SOURCE + filename, sampling_rate, mono=False) 
    ch_0_filtered= vad_gmm.filter_non_speech(signal=audio[0], sampling_rate=sampling_rate)
    feats_0 = torch.Tensor(logmel_feats(signal=ch_0_filtered, sampling_rate=sampling_rate))
    feats_0_splitted = torch.split(feats_0, 10)

    for feat in feats_0_splitted:
        if feat.shape[0] == 10:
            feat = feat.tolist() 
            postgre_args ={"table_name": AsIs(table_name_0), "feat":feat} 
            cur.execute("""INSERT INTO %(table_name)s (feats) 
                    VALUES(%(feat)s)""", postgre_args)

    con.commit()



    ch_1_filtered= vad_gmm.filter_non_speech(signal=audio[1], sampling_rate=sampling_rate)
    feats_1 = torch.Tensor(logmel_feats(signal=ch_1_filtered, sampling_rate=sampling_rate))
    feats_1_splitted = torch.split(feats_1, 10)

    for feat in feats_1_splitted:
        if feat.shape[0] == 10:
            feat = feat.tolist() 
            postgre_args ={"table_name": AsIs(table_name_1), "feat":feat} 
            cur.execute("""INSERT INTO %(table_name)s (feats) 
                    VALUES(%(feat)s)""", postgre_args)

    con.commit()

    con.close()
    return None



filenames = os.listdir(SOURCE) 

if args.threads==1:
    for filename in tqdm(filenames):
        process_callhome_talk(filename)

else:
    with Pool(args.threads) as pool: 
        result = list(tqdm(pool.imap(process_callhome_talk, filenames), total=len(filenames)))
