import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import os
from torch import randint 
import psycopg2
from db_credentials import POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_PORT, POSTGRES_HOST
import sys 

class MFCC_Dataset(Dataset):
    def __init__(self, *, db_name, fixed_area_sample=False, feats_times_ten=10): 
        
        if feats_times_ten <= 0:
            sys.exit('feats_times_ten must be >= 0')

        self.chunks_count=feats_times_ten

        self.db_name=db_name
        conn = psycopg2.connect(dbname=db_name, user=POSTGRES_USER, password=POSTGRES_PASSWORD, host=POSTGRES_HOST, port=POSTGRES_PORT)#, sslmode='require')
        cur = conn.cursor()

        cur.execute("select table_name from information_schema.tables where table_name like 'mfcc%';")
        self.table_names = [i[0] for i in cur.fetchall()]
        self.lengths = []
        for t_name in self.table_names:
            cur.execute('select count(*) from ' + t_name)
            r_count = cur.fetchone()[0]
            self.lengths.append(r_count // self.chunks_count)

        cur.close()
        conn.close() 
        # print(self.table_names, end='\n\n')
        # print(self.lengths)


        self.fixed_area_sample=fixed_area_sample 
        
    
    def __len__(self):
        return len(self.table_names)


    def __getitem__(self, idx):
        conn = psycopg2.connect(dbname=self.db_name, user=POSTGRES_USER, password=POSTGRES_PASSWORD, host='localhost', port=POSTGRES_PORT)
        cur = conn.cursor()

        if self.fixed_area_sample:
            cur.execute('select feats from ' + self.table_names[idx] + ' where ID Between 0 and ' + str( self.chunks_count ) )

        else:
            r_idx = randint(1, self.lengths[idx], (1,)).item()
            cur.execute('select feats from ' + self.table_names[idx] + ' where ID Between ' + str(r_idx) + ' and ' + str(r_idx + self.chunks_count -1) )
        x = None
        try:
            x = cur.fetchall()
            
        except Exception as e:
            print('lengths ', self.lengths[idx])
            print('idx ', idx)
            print('row idx ', r_idx)
            print(type(x))

        res = torch.Tensor(x).reshape(-1, 40)
        cur.close()
        conn.close()
        
        return res


def dataset_split(*, dataset, train_partition):

    train_size = int(train_partition * len(dataset))
    validation_size = len(dataset) - train_size


    train_dataset, validation_dataset = torch.utils.data.random_split(dataset, [train_size, validation_size]) 

    return train_dataset, validation_dataset
