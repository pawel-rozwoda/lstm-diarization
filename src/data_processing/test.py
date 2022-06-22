import sys
from tqdm import tqdm
sys.path.append('../')
from config import DATA_PATH
from db_credentials import POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_HOST, POSTGRES_PORT
import psycopg2
from get_batch import Get_Batch

BATCH_SIZE=16

con = psycopg2.connect(database='vox_1_test', user=POSTGRES_USER, password=POSTGRES_PASSWORD, host=POSTGRES_HOST, port=POSTGRES_PORT) 


# batch_size 
# batch_count
# bucket = [{4,2,9,0}, ...
# bucket_bottom = [0, 440, 520, ...
# batch_n [000000011111222233]

gb = Get_Batch(con, 8, 100)
assert len(gb.bucket) == 5
print(len(gb.batch_n))
print(gb.bucket_bottom)
# print(gb.bucket[-1][-1])
# print(gb.d['mfcc_28'])
# gb.get_batch(2600)
print(gb[10].shape)

# for x in gb:
    # print(x.shape)
print('batch count ' ,gb.batch_count)
# for batch_idx in tqdm(range(gb.batch_count)):
    # x = gb.get_batch(batch_idx)
