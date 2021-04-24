

ag1=torch.load('RESULTS/ag1_'+str(simid)+'.pt')
Rtrain=torch.load('RESULTS/Rtrain_'+str(simid)+'.pt')
Rval=torch.load('RESULTS/Rval_'+str(simid)+'.pt')
L=torch.load('RESULTS/L_'+str(simid)+'.pt')
ys=torch.load('RESULTS/ys_'+str(simid)+'.pt')
yval=torch.load('RESULTS/yval_'+str(simid)+'.pt')
time=torch.load('RESULTS/time_'+str(simid)+'.pt')

scoretrain=Rtrain[Rtrain.__len__()-1]
scoreval=Rval[Rval.__len__()-1]
