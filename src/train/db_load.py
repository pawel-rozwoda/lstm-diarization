import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
from torch import randint 
import psycopg2
from db_credentials import POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_PORT, POSTGRES_HOST
import sys 
sys.path.append('../')
from threading import Thread


def dataset_split(*, dataset, train_partition):

    train_size = int(train_partition * len(dataset))
    validation_size = len(dataset) - train_size


    train_dataset, validation_dataset = torch.utils.data.random_split(dataset, [train_size, validation_size]) 

    return train_dataset, validation_dataset

class MFCC_Dataset(Dataset):
    def sorted_length_speakers(self):
        """
        returns list of speaker_ids in descending speech length like: 
        ['mfcc_5', 'mfcc_47', 'mfcc_12', ... etc]

        lengths = [4845, 4756, 4512, 4205, 4123, ... ] 
        batch_n = [0,0,0,0,0,1,1,1,1,2,2,3]
        """


        self.cursor.execute("SELECT table_name FROM information_schema.tables \
                WHERE table_name like 'mfcc_%' ORDER BY table_name;")
        speaker_ids=[i[0] for i in self.cursor.fetchall()]
        self.lengths=dict()
        for speaker in speaker_ids:
            self.cursor.execute("SELECT count(*) from " + speaker)
            speaker_length = self.cursor.fetchone()[0]
            self.lengths[speaker] = speaker_length
        aux = [i[0] for i in sorted(self.lengths.items(), key=lambda item: item[1], reverse=True)]
        return list(aux)

    def prepare_batches(self, sorted_speakers):
        s=0
        for i in range(len(sorted_speakers)//self.batch_size):
            bucket_speakers = sorted_speakers[i*self.batch_size : (i+1)*self.batch_size]
            self.bucket.append( bucket_speakers )
            last = self.bucket[-1][-1]
            self.bucket_bottom.append(s)
            last_idx = self.lengths[last] // self.occ_len
            s += last_idx
            # print(f"init -> bucket: {bucket_speakers}")
            # print(f"last: {last}, len = {last_idx} ")
            # print()

            for j in range(last_idx):
                self.batch_n.append(i)

        self.batch_count = len(self.batch_n)
        pass

    def _get_batch(self, idx):
        return self.batch_n[idx]

    def _get_bottom(self, idx):
        return self.bucket_bottom[self.batch_n[idx]]


    def __init__(self, db_name, batch_size, occ_len):
        """
        bucket: [
                 [mfcc_56, mfcc_41, mfcc_77, mfcc_21] # four speakers with longest speech
                 [ ...                              ]
                ]
        """

        con = psycopg2.connect(database=db_name, user=POSTGRES_USER, password=POSTGRES_PASSWORD, host=POSTGRES_HOST, port=POSTGRES_PORT) 
        self.db_name=db_name

        self.batch_size = batch_size
        self.occ_len = occ_len
        self.con = con
        self.cursor = con.cursor()
        self.batch_count = 0
        self.bucket = []
        self.bucket_bottom = []
        self.batch_n = []

        sorted_speakers = self.sorted_length_speakers()
        self.prepare_batches(sorted_speakers)
        print(f"bottoms: {self.bucket_bottom}")


    def __getitem__(self, idx):
        
        aux_con = psycopg2.connect(database=self.db_name, user=POSTGRES_USER, password=POSTGRES_PASSWORD, host=POSTGRES_HOST, port=POSTGRES_PORT) 
        aux_cursor = aux_con.cursor()

        if idx<0:
            idx = self.batch_count + idx
        b = self.bucket[self.batch_n[idx]]
        res = []
        bottom = self._get_bottom(idx)
        idx_from = str(1 + (idx*self.occ_len) - (bottom*self.occ_len))
        idx_to = str((idx*self.occ_len) + self.occ_len - (bottom*self.occ_len))
        # previous_bucket_len = sum(self.bucket_bottom(i) for i in range(self.batch_n[idx]))
        # print(f"previous_len: {previous_bucket_len}")
        # print(f"buckets: {self.bucket_bottom}")
        # print(f'DEBUG: bucket  {b}')
        # print(f"bottom: {bottom}")
                                
        # print(f"from: {idx_from}; to: {idx_to}")

        for i in b:

            # self.cursor.execute('select feats from ' + i + ' where ID Between ' + idx_from + ' and ' + idx_to)
            aux_cursor.execute('select feats from ' + i + ' where ID Between ' + idx_from + ' and ' + idx_to)

            # res.append(self.cursor.fetchall())
            res.append(aux_cursor.fetchall())

        # print(f"results {res}")
        r = torch.Tensor(res).reshape(self.batch_size, self.occ_len, -1)
        aux_con.close()
        return r



    def __len__(self):
        return self.batch_count

    def __del__(self):
        print("closing connection to db")
        self.con.close()
