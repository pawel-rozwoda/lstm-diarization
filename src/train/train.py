
import sys
sys.path.append('../')
from model import LSTM_Diarization
import torch
from db_load import MFCC_Dataset, dataset_split, MFCC_Dataset
from tqdm import tqdm, trange
from config import DATA_PATH, OUT_TRAIN
from aux import *
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts, MultiplicativeLR
from ge2e import CELoss, similarity_per_speaker
import argparse
import os
import csv
import torch.utils.data as data_utils
from datetime import datetime
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

torch.manual_seed(2)

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=1000)
parser.add_argument("--num_workers", type=int)
parser.add_argument("--dataset", type=str)
parser.add_argument("--optional_validation", type=str, default=None)
parser.add_argument('--warm_restart', dest='warm_restart', action='store_true')
parser.add_argument("--init_w", type=float, default=10.)
parser.add_argument("--init_b", type=float, default=-5.)
args = parser.parse_args()
print('pid ', os.getpid())
print('initial w ', args.init_w)
print('initial b ', args.init_b)

input_dim = 40
hidden_dim = 768
num_layers = 3 

DB_NAME = args.dataset
OPTIONAL_VALIDATION_DB_NAME = args.optional_validation
print('db name ', DB_NAME)

dt = datetime.now()
out_dir = OUT_TRAIN + dt.strftime('%m%d_') + dt.strftime('%H:%M') + '/'  
print('out dir ', out_dir)

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

print('warm restart ', args.warm_restart)
print('num_workers: ', args.num_workers) 

train_dataset = MFCC_Dataset(db_name='vox_1_test', batch_size=8, occ_len=60)
split_index = 82
print(f"len: {len(train_dataset)}")


model = LSTM_Diarization(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, init_w=10., init_b=-5.)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.train=True
most_recent_validation = 10.
criterion = CELoss(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
print('Using device:', device)

with open(out_dir + 'info.txt', "w") as myfile:
    myfile.write('param w: ' + str(model.w) + '\n' ) 
    myfile.write('param b: ' + str(model.b) + '\n' ) 
    myfile.write('warm restarts ' +  str(args.warm_restart) + '\nepochs ' + str(args.epochs) + '\n') 
    myfile.write('train_dataset ' +  str(args.dataset) + '\n') 

if args.warm_restart:
    T_0 = 40
    T_mult = 2
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0, T_mult)
    iters = len(train_dataset)
    current_iter = 0


for epoch in tqdm(range(args.epochs)):

    for j in tqdm(range(split_index)):
        batch = train_dataset[j]
        batch = batch.to(device) 
        # batch = batch.squeeze(0)
        pred = model(batch)

        loss = criterion(pred) 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if args.warm_restart:
            scheduler.step((epoch + current_iter) / iters)

        # clipping_value = 3 # arbitrary value of your choosing
        # torch.nn.utils.clip_grad_norm(model.parameters(), clipping_value)

    if epoch==args.epochs-1:
        print('saving example pred in epoch = ', epoch)
        torch.save(pred.cpu(), out_dir + 's_matrix.pt')

    train_losses = []
    validation_losses = []
    with torch.no_grad():
        print("train loss")
        for i in range(split_index//4):
            batch = train_dataset[i]
            batch = batch.to(device).squeeze(0)
            pred = model(batch)
            loss = criterion(pred)
            train_losses.append(loss.item()) 

        print("validation loss")
        for i in range(split_index, len(train_dataset)):
            batch = train_dataset[i]
            batch = batch.to(device).squeeze(0)
            pred = model(batch)
            loss = criterion(pred)
            validation_losses.append(loss.item()) 

    if args.warm_restart:
        current_iter +=1

    with open(out_dir + 'train_losses.csv', "a") as myfile:
        myfile.write(','.join(map(str, (epoch, np.mean(train_losses)))) + '\n')
    with open(out_dir + 'validation_losses.csv', "a") as myfile:
        myfile.write(','.join(map(str, (epoch, np.mean(validation_losses)))) + '\n')

    if np.mean(validation_losses) < most_recent_validation:
        most_recent_validation = np.mean(validation_losses)
        print(f"current best model in epoch {epoch}")
        torch.save(model, out_dir + 'best_fit_model.pt')
        with open(out_dir + 'best_fit_update.txt', 'a') as myfile:
            myfile.write( str(epoch) + '\n')

    with open(out_dir + 'lrs.csv', "a") as myfile:
        last_lr = 0
        if args.warm_restart:
            last_lr = scheduler.get_last_lr()[0]
        else:
            last_lr = optimizer.param_groups[0]['lr']

        myfile.write(str(epoch) + ',' + str(last_lr) + '\n')
