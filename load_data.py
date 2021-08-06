import torch
import pandas as pd

#8 9-13

#ys=[];
#ys=torch.tensor(ys)
#trial=1;
#pattern=1

train = pd.read_csv('price_data/delay_p1p2.txt',delimiter=',')
train_tensor = torch.tensor(train.to_numpy(),dtype=torch.float)
ytrain=train_tensor[:,:]


train = pd.read_csv('price_data/delay_p1p2_bis.txt',delimiter=',')
train_tensor = torch.tensor(train.to_numpy(),dtype=torch.float)
yval=train_tensor[:,:]

ley=ytrain;
duration=ley.__len__()-1

#ys=torch.tensor(ys.clone().detach(),dtype=torch.float32)

#test = dd.read_csv("price_data/stockprices_4_30_17.txt", encoding = "UTF-8")
#with open("price_data/stockprices_4_30_17.txt") as players_data:
#    players_data.read()
#delimiter='\t'
