"""
This script provides batch generation tool from CALLHOME or VOX CELEB databases.
"""
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torch

class Get_Batch(Dataset):

    def sorted_length_speakers(self):
        self.cursor.execute("SELECT table_name FROM information_schema.tables \
                WHERE table_name like 'mfcc_%' ORDER BY table_name;")
        x=self.cursor.fetchall()
        self.d=dict()
        for i in x:
            self.cursor.execute("SELECT count(*) from " + i[0])
            res = self.cursor.fetchone()
            self.d[i[0]] = res[0]
        aux = {k: v for k, v in sorted(self.d.items(), key=lambda item: item[1], reverse=True)}
        return list(aux)

    def prepare_batches(self, sorted_speakers):
        s=0
        for i in range(len(sorted_speakers)//self.batch_size):
            print(i)
            self.bucket.append( sorted_speakers[i*self.batch_size : i * self.batch_size + self.batch_size] )
            last = self.bucket[-1][-1]
            self.bucket_bottom.append(s)
            last_idx = self.d[last] // self.occ_len
            s += last_idx

            for j in range(last_idx):
                self.batch_n.append(i)

        self.batch_count = len(self.batch_n)
        pass

    def __init__(self, con, batch_size, occ_len):
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



    def __getitem__(self, idx):
        b = self.bucket[self.batch_n[idx]]
        res = []
        bottom = self.bucket_bottom[self.batch_n[idx]]
        # print('bottom ', bottom)
        # print('bottom - idx', idx - bottom)
        for i in b:
            # self.cursor.execute('select feats from ' + i + ' where ID = ' + str(idx - bottom))
            self.cursor.execute('select feats from ' + i + ' where ID Between ' + str(idx) + ' and ' + str(idx + self.occ_len - 1) )
            res.append(self.cursor.fetchall())
            r = torch.Tensor(res).reshape(self.batch_size, self.occ_len, -1)
            # r = torch.Tensor(res).reshape(self.batch_size, -1, 40)
        return r
    
    def __len__(self):
        return self.batch_count

    def __del__(self):
        print("closing connection to db")
        self.con.close()
