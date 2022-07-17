"""
This script provides batch generation tool from CALLHOME or VOX CELEB databases.
"""
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torch

class Get_Batch(Dataset):

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
        print('speaker ids', speaker_ids)
        self.lengths=dict()
        for speaker in speaker_ids:
            self.cursor.execute("SELECT count(*) from " + speaker)
            speaker_length = self.cursor.fetchone()[0]
            self.lengths[speaker] = speaker_length
        # aux1 = {k: v for k, v in sorted(self.lengths.items(), key=lambda item: item[1], reverse=True)}
        aux = [i[0] for i in sorted(self.lengths.items(), key=lambda item: item[1], reverse=True)]
        return list(aux)

    def prepare_batches(self, sorted_speakers):
        s=0
        for i in range(len(sorted_speakers)//self.batch_size):
            print(i)
            self.bucket.append( sorted_speakers[i*self.batch_size : i * self.batch_size + self.batch_size] )
            last = self.bucket[-1][-1]
            self.bucket_bottom.append(s)
            last_idx = self.lengths[last] // self.occ_len
            s += last_idx

            for j in range(last_idx):
                self.batch_n.append(i)

        self.batch_count = len(self.batch_n)
        pass

    def __init__(self, con, batch_size, occ_len):
        """
        bucket: [
                 [mfcc_56, mfcc_41, mfcc_77, mfcc_21] # four speakers with longest speech
                 [ ...                              ]
                ]
        """
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
        print(f'bucket  bbb {b}')
        print(f'bucket should be {self.bucket[0]}')
        res = []
        bottom = self.bucket_bottom[self.batch_n[idx]]
        print(f'bottom: {bottom}')
        # print('bottom ', bottom)
        # print('bottom - idx', idx - bottom)
        for i in b:
            print(f'range: i: range = (idx - bottom {idx} ,  {idx + self.occ_len - 1})')
            self.cursor.execute('select feats from ' + i + ' where ID Between ' + str(idx - self.bucket_bottom[self.batch_n[idx]]) + ' and ' + str(idx-self.bucket_bottom[self.batch_n[idx]] + self.occ_len))
            # self.cursor.execute('select feats from ' + i + ' where ID Between ' + str(idx) + ' and ' + str(idx + self.occ_len -1) )
            res.append(self.cursor.fetchall())
            # r = torch.Tensor(res).reshape(self.batch_size, -1, 40)
        auxiliary_r = torch.Tensor(res)
        print('aux shape', auxiliary_r.shape)
        r = torch.Tensor(res).reshape(self.batch_size, self.occ_len, -1)
        # r = torch.tensor(res)
        return r
    
    def __len__(self):
        return self.batch_count

    def __del__(self):
        print("closing connection to db")
        self.con.close()
