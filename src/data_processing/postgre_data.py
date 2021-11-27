import sys
sys.path.append('../')
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
from db_credentials import DB, USER, PASSWORD, HOST, PORT


parser = argparse.ArgumentParser()

parser.add_argument("--threads", type=int, default=1)
parser.add_argument("--data_source", type=str)
args = parser.parse_args()
db_name = args.data_source[:-1]

print("Database opened successfully")
con = psycopg2.connect(database=DB, user=USER, password=PASSWORD, host=HOST, port=PORT) 
con.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
cursor = con.cursor()

cursor.execute('drop database if exists '+ db_name )
cursor.execute('create database '+ db_name) 

con.commit()
con.close()


con2 = psycopg2.connect(database=db_name, user=USER, password=PASSWORD, host=HOST, port=PORT) 
cursor2 = con2.cursor()
cursor2.execute('''CREATE TABLE if not exists speakers (ID smallint, speaker varchar(10) );''')
con2.commit()
con2.close()

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

SOURCE = data_source + audio_type
DESTINATION = data_source + 'intermediate/'
sampling_rate=16000

print('src: ', SOURCE)
print('destination: ',DESTINATION)
print('audio type', audio_type)
print('threads: ', args.threads)
print('sampling_rate: ', sampling_rate)


print('data processing on: ', SOURCE)


def process_speaker(speaker_with_id): 
    con = psycopg2.connect(database=db_name, user=USER, password=PASSWORD, host=HOST, port=PORT)
    cur = con.cursor()
    idx, speaker = speaker_with_id
    postgre_args ={"ID": idx, "speaker":speaker} 
    cur.execute("""INSERT INTO speakers (ID, speaker) VALUES(%(ID)s, %(speaker)s )""", postgre_args)

    """vad_gmm setup"""
    vad_gmm=None
    with open(GMM_FILE, 'rb') as fid:
        # print('reading gmm file from ', GMM_FILE)
        vad_gmm = pickle5.load(fid) 
    """end setup"""

    table_name = 'MFCC_'+str(idx)
    DROP_COMMAND = 'drop table if exists ' + table_name
    cur.execute(DROP_COMMAND)
    con.commit()
    cur.execute('''CREATE TABLE %(table_name)s
          (ID SERIAL PRIMARY KEY,
          feats float[10][40]);''', {"table_name":AsIs(table_name) })
    con.commit()

    performances = os.listdir(SOURCE + speaker)
    for p in performances: 
        utterances = os.listdir(SOURCE + speaker + '/' + p)

        for utt in utterances: 
            pass
            file_path = SOURCE +  speaker + '/' + p + '/' +  utt 
            _, audio = load_audio(file_path, sampling_rate, mono=True) 
            audio_filtered = vad_gmm.filter_non_speech(signal=audio, sampling_rate=sampling_rate)
            
            feats = torch.Tensor(logmel_feats(signal=audio_filtered, sampling_rate=sampling_rate))
            feats_splitted = torch.split(feats, 10)

            for feat in feats_splitted:
                if feat.shape[0]==10:
                    feat = feat.tolist() 
                    postgre_args ={"table_name": AsIs(table_name), "feat":feat} 
                    cur.execute("""INSERT INTO %(table_name)s (feats) 
                            VALUES(%(feat)s)""", postgre_args)


            con.commit()
    
    con.close()

    return None


speakers = os.listdir(SOURCE) 
speakers_with_id = [(i, speaker) for i, speaker in enumerate(speakers)]


if os.path.exists(DESTINATION) and os.path.isdir(DESTINATION):
    shutil.rmtree(DESTINATION) 

os.makedirs(DESTINATION)
os.makedirs(DESTINATION + 'files/')


if args.threads==1:
    for i, speaker in tqdm(speakers_with_id):
        process_speaker((i, speaker))

else:
    # with Pool(cpu_count() - 4) as pool: 
    with Pool(args.threads) as pool: 
        result = list(tqdm(pool.imap(process_speaker, speakers_with_id), total=len(speakers_with_id)))

    # or
    # for _ in tqdm(pool.imap(process_speaker, speakers)):
        #pass 

