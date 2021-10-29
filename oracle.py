phase=np.sign(ley[0,0]-ley[1,0]);
if phase==0:
    phase=1

duration=ley.__len__()

lemax=ley[0,0];
maxind=1;
max_inds=[];
max_vals=[];
lemin=ley[0,0];
minind=1;
min_inds=[];
min_vals=[];
prices=[];
cost=0.003;
for k in range(1,duration):
    prices.append(ley[k,0]);
    if (phase==1): #if price goes up
        if (ley[k,0]>=lemax):
            lemax=ley[k,0];
            maxind=k;
        if (ley[k,0]<(1-cost)*(1-cost)*lemax): #distance from peak
            max_inds.append(maxind);
            max_vals.append(float(lemax))
            lemin=ley[k,0];
            minind=k;
            phase=-1;
    if (phase==-1):
        if (ley[k,0]<=lemin):
            lemin=ley[k,0];
            minind=k;
        if (ley[k,0]>(1+cost)*(1+cost)*lemin):
            min_inds.append(minind);
            min_vals.append(float(lemin))
            lemax=ley[k,0];
            maxind=k;
            phase=1;

#calculate cost
if (max_inds.__len__()==0):
    max_vals.append(ley.max());
    max_inds.append(ley.argmax());
    min_vals.append(ley.min());
    min_inds.append(ley.argmin());

if (max_inds[0]<min_inds[0]):
    max_inds.pop(0);
    max_vals.pop(0);

#calculate cost
if (max_inds.__len__()>0):
    min_inds=min_inds[0:max_inds.__len__()];
    min_vals=min_vals[0:max_inds.__len__()];

gain=100;

#


for i in range(max_inds.__len__()):
   gain=gain/ley[min_inds[i],0]*(1-cost);#buy
   gain=gain*ley[max_inds[i],0]*(1-cost);#sell

#plt.clf()
#plt.plot(prices,'k',linewidth=1)
#if (max_inds.__len__()>0):
#    plt.scatter(max_inds,max_vals,s=30,color='green')
#    plt.scatter(min_inds,min_vals,s=30,color='red')

#plt.show(block=False)
#print(gain)
#print(np.exp(np.log(gain/100)/duration)-1)
maxr=100*(np.exp(np.log(gain/100)/duration)-1)
