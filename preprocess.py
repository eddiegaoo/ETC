import pandas as pd
import torch
import numpy as np
import csv
from tqdm import tqdm
import itertools
import argparse


parser=argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='dataset name')
parser.add_argument('--txt', type=str, help='txt file path')
args=parser.parse_args()


print('open txt file...')
with open(args.txt, "r") as f:
    data = f.readlines()
for i in range(len(data)):
    data[i] = data[i].split()

print('to DataFrame...')
df = pd.DataFrame(data[1:])
df.columns = ['src','dst','time']   

print('reindex...')
#reindex src & dst
src = df.src.astype(int)
dst = df.dst.astype(int)
cut = len(src)
total = np.concatenate((src,dst))
total,inv = np.unique(total,return_inverse=True)

new_data = np.arange(len(total))
df['src'] = new_data[inv][:cut]
df['dst'] = new_data[inv][cut:]

print('sort by time...')
# sort by time
df.time =  df.time.astype(int)
df.time -= df.time.min()
df = df.sort_values(by=['time'])
df = df.reset_index(drop=True)

print('split the data...')
#data split
values = np.zeros(len(df),dtype=int)
values[int(0.7*len(df)):int(0.85*len(df))] = 1
values[int(0.85*len(df)):] = 2
df['ext_roll'] = values

print('save csv...')
df.to_csv('./DATA/{}/edges.csv'.format(args.data))

print('generate graph...')
#gen_graph
df = pd.read_csv('./DATA/{}/edges.csv'.format(args.data))

num_nodes = max(int(df['src'].max()), int(df['dst'].max())) + 1
print('num_nodes: ', num_nodes)

int_train_indptr = np.zeros(num_nodes + 1, dtype=np.int)
int_train_indices = [[] for _ in range(num_nodes)]
int_train_ts = [[] for _ in range(num_nodes)]
int_train_eid = [[] for _ in range(num_nodes)]

int_full_indptr = np.zeros(num_nodes + 1, dtype=np.int)
int_full_indices = [[] for _ in range(num_nodes)]
int_full_ts = [[] for _ in range(num_nodes)]
int_full_eid = [[] for _ in range(num_nodes)]

ext_full_indptr = np.zeros(num_nodes + 1, dtype=np.int)
ext_full_indices = [[] for _ in range(num_nodes)]
ext_full_ts = [[] for _ in range(num_nodes)]
ext_full_eid = [[] for _ in range(num_nodes)]


for idx, row in tqdm(df.iterrows(), total=len(df)):
    src = int(row['src'])
    dst = int(row['dst'])
    ext_full_indices[src].append(dst)
    ext_full_ts[src].append(row['time'])
    ext_full_eid[src].append(idx)
    

for i in tqdm(range(num_nodes)):
    ext_full_indptr[i + 1] = ext_full_indptr[i] + len(ext_full_indices[i])

ext_full_indices = np.array(list(itertools.chain(*ext_full_indices)))
ext_full_ts = np.array(list(itertools.chain(*ext_full_ts)))
ext_full_eid = np.array(list(itertools.chain(*ext_full_eid)))

print('Sorting...')
def tsort(i, indptr, indices, t, eid):
    beg = indptr[i]
    end = indptr[i + 1]
    sidx = np.argsort(t[beg:end])
    indices[beg:end] = indices[beg:end][sidx]
    t[beg:end] = t[beg:end][sidx]
    eid[beg:end] = eid[beg:end][sidx]

for i in tqdm(range(int_train_indptr.shape[0] - 1)):
    tsort(i, ext_full_indptr, ext_full_indices, ext_full_ts, ext_full_eid)

print('saving...')

np.savez('./DATA/{}/ext_full.npz'.format(args.data), indptr=ext_full_indptr, indices=ext_full_indices, ts=ext_full_ts, eid=ext_full_eid)
