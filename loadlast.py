

ag1=torch.load('RESULTS/ag1_'+str(simid)+'.pt')
Rtrain=torch.load('RESULTS/Rtrain_'+str(simid)+'.pt')
Rval=torch.load('RESULTS/Rval_'+str(simid)+'.pt')
L=torch.load('RESULTS/L_'+str(simid)+'.pt')
ytrain=torch.load('RESULTS/ytrain_'+str(simid)+'.pt')
yval=torch.load('RESULTS/yval_'+str(simid)+'.pt')
time=torch.load('RESULTS/time_'+str(simid)+'.pt')
T=torch.load('RESULTS/T_'+str(simid)+'.pt')
RT=torch.load('RESULTS/RT_'+str(simid)+'.pt')
scoretrain=Rtrain[Rtrain.__len__()-1]
scoreval=Rval[Rval.__len__()-1]

TRAIN_MAXS=torch.load('RESULTS/TRAIN_MAXS_'+str(simid)+'.pt')
TRAIN_INDS=torch.load('RESULTS/TRAIN_INDS_'+str(simid)+'.pt')

TRAIN_MAXS=torch.load('RESULTS/VAL_MAXS_'+str(simid)+'.pt')
TRAIN_INDS=torch.load('RESULTS/VAL_INDS_'+str(simid)+'.pt')
