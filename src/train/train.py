import sys
sys.path.append('../')
from model import LSTM_Diarization
import torch
from db_load import MFCC_Dataset, dataset_split
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

torch.manual_seed(2)

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=1)
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


input_dim = 40
hidden_dim = 768
num_layers = 3 
batch_size = 16


num_workers=args.num_workers


""" split dataset to train and test """
train_partition = 0.9 

# full_dataset = MFCC_Dataset(db_name=DB_NAME, feats_times_ten=40, fixed_area_sample=True)
full_dataset = MFCC_Dataset(db_name=DB_NAME, feats_times_ten=40)
train_dataset, validation_dataset = dataset_split(dataset=full_dataset, train_partition=train_partition)
""" end split """

print('train split: ', train_partition)
print('train_size', len(train_dataset))
print('validation size', len(validation_dataset))


# train_loader = DataLoader( train_dataset, batch_size=batch_size, sampler=RandomSampler(train_dataset), num_workers=num_workers)
train_loader = DataLoader( train_dataset, batch_size=batch_size, num_workers=num_workers)
validation_loader = DataLoader( validation_dataset, batch_size=batch_size, num_workers=num_workers)

if len(validation_dataset) % batch_size == 1 or len(train_dataset) % batch_size == 1:
    sys.exit('batch size cannot be 1 in any iteration') 

if args.optional_validation:
    optional_validation_dataset = MFCC_Dataset(db_name=OPTIONAL_VALIDATION_DB_NAME, fixed_area_sample=True)
    optional_validation_loader = DataLoader( optional_validation_dataset, batch_size=batch_size, num_workers=num_workers)



""" reduced """
reduced_size = len(validation_dataset)
reduced_train_dataset = data_utils.Subset(train_dataset, torch.arange(0,reduced_size ))
reduced_train_loader = torch.utils.data.DataLoader(reduced_train_dataset, batch_size=batch_size, shuffle=True) 
""" end reduced """


print('train: ' + DB_NAME + ': ' + str(len(train_dataset)))
print('train reduced ', str(len(reduced_train_dataset)))
print('optional validation ' + str(OPTIONAL_VALIDATION_DB_NAME))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

model = LSTM_Diarization(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, init_w=args.init_w, init_b=args.init_b)
model.train=True

with open(out_dir + 'info.txt', "w") as myfile:
    myfile.write('param w: ' + str(model.w) + '\n' ) 
    myfile.write('param b: ' + str(model.b) + '\n' ) 
    myfile.write('warm restarts ' +  str(args.warm_restart) + '\nepochs ' + str(args.epochs) + '\n') 
    myfile.write('train_dataset ' +  str(args.dataset) + '\n') 
    myfile.write('optional validation ' +  str(args.optional_validation) + '\n') 

print('model dtype', model.dtype)
model.to(device) 
model.train = True

scheduler = None
lr = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

if args.warm_restart:
    T_0 = 40
    T_mult = 2
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0, T_mult)

epochs = args.epochs
print(epochs, ' epochs')

iters = len(train_loader)

most_recent_validation = 10.
criterion = CELoss(device)
for epoch in trange(epochs):
    if not args.warm_restart:
        if  (epoch % 300) + 1 == 0:
            optimizer.param_groups[0]['lr'] /= 2
            # print('lr change:', scheduler.get_last_lr())

    """ train and validation are subsets of full_dataset """
    full_dataset.fixed_area_sample=False
    """ """
    grads_stacked = []
    for i, batch in enumerate(train_loader):
        """batch or "feats" shape: i.e (64, 10, 40)"""

        batch = batch.to(device) 
        batch = batch.squeeze(0)
        
        pred = model(batch)

        loss = criterion(pred) 

        optimizer.zero_grad()
        loss.backward()
        # clipping_value = 3 # arbitrary value of your choosing
        # torch.nn.utils.clip_grad_norm(model.parameters(), clipping_value)

        """ for analysis purposes """
        gradients_lengths = []
        for w in model.parameters():
            if w.grad is not None:
                n = torch.norm(w.grad)
                gradients_lengths.append(n) 
        gradients_lengths = torch.Tensor(gradients_lengths) 
        grads_stacked.append(gradients_lengths)
        """ end """


        optimizer.step() 

        if args.warm_restart:
            scheduler.step(epoch + i / iters)


    if epoch==epochs-1:
        print('saving example pred in epoch = ', epoch)
        torch.save(pred.cpu(), out_dir + 's_matrix.pt')

    st = torch.stack(grads_stacked)
    st = st.mean(axis=0)

    """ train and validation are subsets of full_dataset """
    full_dataset.fixed_area_sample=True
    """ """
    train_losses = []
    with torch.no_grad():
        for batch in reduced_train_loader:
            batch = batch.to(device).squeeze(0)
            pred = model(batch)

            loss = criterion(pred)
            train_losses.append(loss.item()) 

    validation_losses=[]
    with torch.no_grad():
        for batch in validation_loader:
            batch = batch.to(device).squeeze(0) 
            pred = model(batch)

            loss = criterion(pred) 
            validation_losses.append(loss.item())

    if args.optional_validation:
        optional_validation_losses=[]
        with torch.no_grad():
            for batch in optional_validation_loader:
                batch = batch.to(device).squeeze(0) 
                pred = model(batch)

                loss = criterion(pred) 
                optional_validation_losses.append(loss.item())

        with open(out_dir + 'optional_validation_losses.csv', "a") as myfile:
            myfile.write(','.join(map(str, (epoch, np.mean(optional_validation_losses)))) + '\n')




    with open(out_dir + 'train_losses.csv', "a") as myfile:
        myfile.write(','.join(map(str, (epoch, np.mean(train_losses)))) + '\n')

    with open(out_dir + "validation_losses.csv", "a") as myfile:
        myfile.write(','.join(map(str, (epoch, np.mean(validation_losses)))) + '\n') 

    with open(out_dir + 'grads.txt', 'a') as myfile:
        myfile.write( str(epoch) + ',' + ','.join(map(str, (st.flatten().tolist()))  ) + '\n')

    with open(out_dir + 'lrs.csv', "a") as myfile:
        last_lr = 0
        if args.warm_restart:
            last_lr = scheduler.get_last_lr()[0]
        else:
            last_lr = optimizer.param_groups[0]['lr']

        myfile.write(str(epoch) + ',' + str(last_lr) + '\n')

    if np.mean(validation_losses) < most_recent_validation:
        most_recent_validation = np.mean(validation_losses)
        torch.save(model, out_dir + 'best_fit_model.pt')
        with open(out_dir + 'best_fit_update.txt', 'a') as myfile:
            myfile.write( str(epoch) + '\n')




torch.save(model, out_dir + 'last_epoch_model.pt')
