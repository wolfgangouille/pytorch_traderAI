A0=[]
A1=[]
A2=[]

prices=[]
colors=[]
times=[]
sizes=[]
lesr2=[]
lesv2=[]
ag1.iterations=0;
ag1.meanreward=0;
ag1.EUR=100;
ag1.XBT=0;
ag1.value=100;
olde=0.1#ag1.epsilon;
ag1.epsilon=0.00;
ag1.record_on=0;
ag1.position=-1;
ag1.lasth=torch.zeros(1,1,ag1.nhidden)
ag1.lastinput=torch.zeros_like(ag1.lastinput)
ag1.previous_inputs=torch.zeros_like(ag1.previous_inputs)+ley[0]
ag1.lastinput[0,0,0]=-1
duration = ley.__len__();
for k in range(duration):
    ag1.lasth=ag1.lasth.to(device)
    ag1.lastinput=ag1.lastinput.to(device)
    ag1.choose_a()
    prices.append(ag1.last_price)

    A0.append(ag1.lastoutput[0][0][0].detach()) # remove [0]
    A1.append(ag1.lastoutput[0][0][1].detach())
    A2.append(ag1.lastoutput[0][0][2].detach())


    ag1.compute_reward_and_new_state(ley[k,:].view(1,ag1.ninput)) #feed new data and play

    lesr2.append(ag1.lastr)
    lesv2.append(ag1.value)
    times.append(k)
    if ag1.lasta==0: #buy
        colors.append('r')
        sizes.append(30)
    if ag1.lasta==1: #sell
        colors.append('g')
        sizes.append(30)
    if ag1.lasta==2:
        colors.append('b')
        sizes.append(2)

nfutur=1
lesR=[]
for i in range(lesr2.__len__()-nfutur-1):
    lesR.append(0.0)
    for t in range(nfutur):
        lesR[i]+=(ag1.gamma**t)*lesr2[i+t][0][0]


ag1.epsilon=olde
ag1.record_on=1;

score=ag1.meanreward*100


#ag1.savebestSD('NNbest'+str(time)+'.sd')
'''
plt.clf()
plt.plot(A0,'r',linewidth=0.5)
plt.plot(A1,'g',linewidth=0.5)
plt.plot(A2,'b',linewidth=0.5)
plt.plot(lesR,'k',linewidth=0.5)
#plt.plot(lesr[:,0,0],'k',linewidth=1.5)
plt.show(block=False)
plt.xlim([300, 700])
plt.ylim([-0.15, 0.25])
plt.pause(0.01)
#plt.savefig('RESULTS/strategies/foo'+str(time)+'.png')


'''

plt.clf()
#plt.scatter(times,prices,s=sizes,color=colors )
plt.plot(times,prices,'k',linewidth=1)
prices=torch.tensor(prices)
plt.scatter(times,prices,s=sizes,color=colors )
plt.show(block=False)
#plt.xlim([0, 1000])
plt.pause(0.01)
