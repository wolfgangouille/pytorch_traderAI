
lesi=torch.zeros(duration,1,ag1.ninput*ag1.n_memory+1)
lesh=torch.zeros(duration,1,ag1.nhidden)

times=[]
sizes=[]
lesr=[]
lesa=[]
lesv=[]

ag1.EUR=100;
ag1.XBT=0;
ag1.value=100;
ag1.meanreward=0;
ag1.iterations=0;
ag1.record_on=0;
ag1.lasth=torch.zeros(1,1,ag1.nhidden)
ag1.lastinput=torch.zeros_like(ag1.lastinput)
ag1.lastinput[0,0,0]=-1

ag1.previous_inputs=torch.zeros_like(ag1.previous_inputs)+ys[0,:]

ag1.position=-1

#ag1.epsilon=0.1

#ag1.updateNN()


duration=ys.__len__()-Ilength-1

#duration=100

for t in range(duration):
    #ag1.play(ys[k:k+Ilength].view(1,Ilength))

    lesh[t,0,:]=ag1.lasth.clone().detach();
    lesi[t,0,:]=ag1.lastinput.clone().detach(); #get state before playing

    ag1.choose_a()

    lesa.append(ag1.lasta.squeeze(0).squeeze(0))
    ag1.compute_reward_and_new_state(ys[t,:].view(1,ag1.ninput)) #feed new data
    lesr.append(ag1.lastr.squeeze(0).squeeze(0)) #get action and reward
    lesv.append(ag1.value.squeeze(0).squeeze(0))




#plt.plot(lesr),plt.show()

lesr=torch.as_tensor(lesr).view(duration,1,1)
lesa=torch.as_tensor(lesa).view(duration,1,1)
