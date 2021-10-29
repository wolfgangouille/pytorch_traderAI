import torch
import pandas as pd
import os
all_files=[];

lepath='price_data/';
for x in os.listdir(lepath+'test/'):
    if x.endswith('.txt'):
        all_files.append(lepath+'test/'+x);


for file in all_files:
    data = pd.read_csv(file,delimiter=' ',header=None);
    data = torch.tensor(data.to_numpy(),dtype=torch.float)
    ley=data[:,6].unsqueeze(-1)
    exec(open('oracle.py').read())
