
less=torch.zeros(duration,1,ag1.inputlength+ag1.hiddenlength)
times=[]
sizes=[]
lesr=[]
lesa=[]
ag1.EUR=100;
ag1.XBT=0;
ag1.value=100;
ag1.meanreward=0;
ag1.iterations=0;
ag1.record_on=0;
ag1.lasth=torch.zeros(1,1,ag1.hiddenlength)
ag1.epsilon=0.25

Nbatch=150
Lbatch=20

duration=ys.__len__()-Ilength-1

for k in range(duration):
    #ag1.play(ys[k:k+Ilength].view(1,Ilength))
    ag1.play(ys[k:k+Ilength])
    lesr.append(ag1.lastr.squeeze(0).squeeze(0))
    less[k][0]=ag1.s[0][0].view(-1)
    lesa.append(ag1.lasta.squeeze(0).squeeze(0))

#plt.plot(lesr),plt.show()

lesr=torch.as_tensor(lesr).view(duration,1,1)
lesa=torch.as_tensor(lesa).view(duration,1,1)

lesi,lesh=torch.split(less,[ag1.inputlength,ag1.hiddenlength],dim=2)

inputbatch=torch.zeros(Lbatch,Nbatch,ag1.inputlength)
#nextinputbatch=torch.zeros(Lbatch,Nbatch,ag1.inputlength)
h0batch=torch.zeros(1,Nbatch,ag1.hiddenlength)
#nexth0batch=torch.zeros(1,Nbatch,ag1.hiddenlength)
abatch=torch.zeros(Lbatch,Nbatch,1).long()
rbatch=torch.zeros(Lbatch,Nbatch,1)
