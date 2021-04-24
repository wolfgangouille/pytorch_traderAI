less=torch.zeros(duration,1,ag1.inputlength+ag1.hiddenlength)
times=[]
sizes=[]
lesr=[]
lesa=[]
ag1.updateNN()
ag1.EUR=100;
ag1.XBT=0;
ag1.value=100;

ag1.record_on=0;
ag1.lasth=torch.zeros(1,1,ag1.hiddenlength)
for k in range(duration):
    #ag1.play(ys[k:k+Ilength].view(1,Ilength))
    ag1.play(ys[k:k+Ilength])
    lesr.append(ag1.lastr.squeeze(0).squeeze(0))
    less[k][0]=ag1.s[0][0].view(-1)
    lesa.append(ag1.lasta.squeeze(0).squeeze(0))

lesr=torch.as_tensor(lesr).view(30,100,1)
lesa=torch.as_tensor(lesa).view(30,100,1)



inputs,h0s=torch.split(less,[ag1.inputlength,ag1.hiddenlength],dim=2)
inputs=inputs.view(30,100,ag1.inputlength)
h0s=h0s.view(30,100,ag1.hiddenlength)

for t in range(10):

    outputs,hn=ag1.NNp(inputs[0:29],h0s[0].unsqueeze(0))
    actuals=outputs.gather(2, lesa[0:29]);

    outputs2,hn2=ag1.NN(inputs[1:30],h0s[1].unsqueeze(0))
    targets=lesr[0:29]+ag1.gamma*outputs2.max(2)[0].unsqueeze(2);


    ag1.optimizer.zero_grad()     # zeroes the gradient buffers of all parameters
    loss=ag1.criterion(actuals, targets)
    loss.backward(retain_graph=True)
    for param in ag1.NNp.parameters():
        param.grad.data.clamp_(-1, 1)
    ag1.optimizer.step()
    ag1.meanerror=math.sqrt(loss)
    print(ag1.meanerror)
    L.append(ag1.meanerror)
#print(actual)
