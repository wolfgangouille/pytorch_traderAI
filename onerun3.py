t0=tm.time();

device=torch.device("cpu");
ag1.NN=ag1.NN.to(device)
ag1.NN.mask2=ag1.NN.mask2.to(device)
ag1.NN.mask1=ag1.NN.mask1.to(device)






###For a particular ley in training data
duration = ley.__len__()-1;
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
ag1.lasta=2;
ag1.lasth=torch.zeros(1,1,ag1.nhidden)
ag1.lastinput=torch.zeros_like(ag1.lastinput)
ag1.lastinput[0,0,0]=-1 #no crypto
ag1.previous_inputs=torch.zeros_like(ag1.previous_inputs)+ley[0,:]
ag1.position=-1

ag1.last_price=ley[0,:].view(1,ag1.ninput)[0,0]; #initialize with firs time point


for t in range(0,duration):

    lesh[t,0,:]=ag1.lasth.clone().detach()
    lesi[t,0,:]=ag1.lastinput.clone().detach()

    ag1.lasth=ag1.lasth.to(device)
    ag1.lastinput=ag1.lastinput.to(device)

    ag1.choose_a()

    lesa.append(ag1.lasta.squeeze(0).squeeze(0))
    ag1.compute_reward_and_new_state(ley[t+1,:].view(1,ag1.ninput)) #feed new data skip first point
    lesr.append(ag1.lastr.squeeze(0).squeeze(0)) #get action and reward
    lesv.append(ag1.value.squeeze(0).squeeze(0))




#plt.plot(lesr),plt.show()

lesr=torch.as_tensor(lesr).view(duration,1,1)
lesa=torch.as_tensor(lesa).view(duration,1,1)

#
inds=range(duration-Lbatch);
inds=np.array(inds)
temp=np.matlib.reshape(np.matlib.repmat(inds,Lbatch,1).transpose(1,0)+np.matlib.repmat(np.arange(Lbatch),duration-Lbatch,1),[Lbatch*(duration-Lbatch),1])
small_inputbatch_lib=lesi[temp[:,0],:,:].reshape([duration-Lbatch,Lbatch,ag1.ninput+1]).transpose(1,0)
small_h0batch_lib=lesh[inds,0,:].unsqueeze(0)
small_abatch_lib=lesa[temp[:,0],0,0].reshape([duration-Lbatch,Lbatch,1]).transpose(1,0)
small_rbatch_lib=lesr[temp[:,0],0,0].reshape([duration-Lbatch,Lbatch,1]).transpose(1,0)

inputbatch_lib=torch.cat((inputbatch_lib,small_inputbatch_lib),dim=1)
h0batch_lib=torch.cat((h0batch_lib,small_h0batch_lib),dim=1)
abatch_lib=torch.cat((abatch_lib,small_abatch_lib),dim=1)
rbatch_lib=torch.cat((rbatch_lib,small_rbatch_lib),dim=1)

kkk=inputbatch_lib.size(1)
if (kkk>max_replay_size):
    inputbatch_lib=inputbatch_lib[:,kkk-max_replay_size:,:];
    h0batch_lib=h0batch_lib[:,kkk-max_replay_size:,:];
    abatch_lib=abatch_lib[:,kkk-max_replay_size:,:];
    rbatch_lib=rbatch_lib[:,kkk-max_replay_size:,:];

print(kkk)

#end of particular library
#concatenate all the lib until it reaches max value for replay.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ag1.NN=ag1.NN.to(device)
ag1.NN.mask2=ag1.NN.mask2.to(device)
ag1.NN.mask1=ag1.NN.mask1.to(device)

t1=tm.time();
print('onerun3 Elapsed time : '+str(t1-t0)+' s.')
