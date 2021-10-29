import torch
import pandas as pd
import os
train_files=[];
val_files=[];

lepath='price_data/';
for x in os.listdir(lepath+'training/'):
    if x.endswith('.txt'):
        train_files.append(lepath+'training/'+x);

for x in os.listdir(lepath+'validation/'):
    if x.endswith('.txt'):
        val_files.append(lepath+'validation/'+x);
#8 9-13
train_files=sorted(train_files);
val_files=sorted(val_files)

#ys=[];
#ys=torch.tensor(ys)
#trial=1;
#pattern=1



train_ind=random.randint(0,len(train_files)-1);
data = pd.read_csv(train_files[train_ind],delimiter=' ',header=None)
data = torch.tensor(data.to_numpy(),dtype=torch.float)
ytrain=data[:,6:8]#.unsqueeze(-1)
ley=ytrain;
exec(open('oracle.py').read())
maxrtrain=maxr;

# ystrain=[];
# maxsrtrain=[];
# for file in train_files:
#     data = pd.read_csv(file,delimiter=' ',header=None)
#     data = torch.tensor(data.to_numpy(),dtype=torch.float)
#     ytrain=data[:,6:8]#.unsqueeze(-1)
#     ley=ytrain;
#     ystrain.append(ytrain);
#     exec(open('oracle.py').read())
#     maxrtrain=maxr;
#     maxsrtrain.append(maxrtrain);







val_ind=random.randint(0,len(val_files)-1);
data = pd.read_csv(val_files[val_ind],delimiter=' ',header=None)
data = torch.tensor(data.to_numpy(),dtype=torch.float)
yval=data[:,6:8]#.unsqueeze(-1)
ley=yval;
exec(open('oracle.py').read())
maxrval=maxr;

ley=ytrain;
duration=ley.__len__()

# ysval=[];
# maxsrval=[];
# for file in val_files:
#     data = pd.read_csv(file,delimiter=' ',header=None)
#     data = torch.tensor(data.to_numpy(),dtype=torch.float)
#     yval=data[:,6:8]#.unsqueeze(-1)
#     ley=yval;
#     ysval.append(ytrain);
#     exec(open('oracle.py').read())
#     maxrval=maxr;
#     maxsrval.append(maxrval);

# data = pd.read_csv('price_data/test/continuous_chunk_104.txt',delimiter=' ',header=None)
# data = pd.read_csv(train_files[train_ind],delimiter=' ',header=None)
# data = torch.tensor(data.to_numpy(),dtype=torch.float)
# ytrain=data[:,6:8]#.unsqueeze(-1)
# ley=ytrain;
# exec(open('oracle.py').read())
# plt.plot(ley[:,0].numpy()),plt.show()
