import torch
import pandas as pd

#8 9-13
#9 10-16
#9 25-30
#10 1-6
#10 10-16
month=10
#day=10
ys=[];
ys=torch.tensor(ys)

for day in range(10,16):
#day=torch.randint(10,16,(1,1))
    print(day)
    if day>=10:
        train = pd.read_csv('price_data/stockprices_'+str(int(month))+'_'+str(int(day))+'_17.txt',delimiter=' ')
        train_tensor = torch.tensor(train.to_numpy(),dtype=torch.float)
        ys=torch.cat((ys,train_tensor[:,7]),0)
    else:
        train = pd.read_csv('price_data/stockprices_'+str(int(month))+'_ '+str(int(day))+'_17.txt',delimiter=' ')
        train_tensor = torch.tensor(train.to_numpy(),dtype=torch.float)
        ys=torch.cat((ys,train_tensor[:,8]),0)



ys=ys[::3]


duration=ys.__len__()-Ilength-1

#ys=torch.tensor(ys.clone().detach(),dtype=torch.float32)

#test = dd.read_csv("price_data/stockprices_4_30_17.txt", encoding = "UTF-8")
#with open("price_data/stockprices_4_30_17.txt") as players_data:
#    players_data.read()
#delimiter='\t'
