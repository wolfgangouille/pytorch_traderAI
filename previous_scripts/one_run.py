ag1.iterations=0;
ag1.meanreward=0;
ag1.EUR=100;
ag1.XBT=0;
ag1.value=100;
print(ag1.epsilon)
#ys=(amplitude*(torch.cos(ts/4)+torch.cos(ts/2)+torch.cos(ts/10)))+100+torch.randn((ts.size()))*amplitude/10;

for k in range(duration):
    ag1.play(ys[k:k+Ilength])
