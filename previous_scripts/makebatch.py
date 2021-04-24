time=time+1

#exec(open('importData.py').read())


exec(open('onerun2.py').read())


for t in range(50):

    #sampling a batch
    for k in range(Nbatch):
        ind=torch.tensor(random.randrange(duration-Lbatch-1))
        inputbatch[:,k,:]=lesi[ind:ind+Lbatch,0,:]
        #nextinputbatch[:,k,:]=inputs[ind+1:ind+Lbatch+1,0,:]
        h0batch[:,k,:]=lesh[ind,0,:]
        #nexth0batch[:,k,:]=h0s[ind+1,0,:]
        abatch[:,k,0]=lesa[ind:ind+Lbatch,0,0]
        rbatch[:,k,0]=lesr[ind:ind+Lbatch,0,0]


    #Computing batch loss
    outputbatch,hn=ag1.NNp(inputbatch,h0batch)
    actualbatch=outputbatch.gather(2, abatch)
    actualbatch=actualbatch[0:Lbatch-1,:,:]
    outputbatch2,hn2=ag1.NN(inputbatch,h0batch)
    targetbatch=rbatch[0:Lbatch-1,:,:]+ag1.gamma*outputbatch2[1:Lbatch,:,:].max(2)[0].unsqueeze(2)


    ag1.optimizer.zero_grad()     # zeroes the gradient buffers of all parameters
    loss=ag1.criterion(actualbatch, targetbatch)
    loss.backward(retain_graph=True)
    for param in ag1.NNp.parameters():
        param.grad.data.clamp_(-1, 1)
    ag1.optimizer.step()
    ag1.meanerror=math.sqrt(loss)
    L.append(math.log10(ag1.meanerror))

ag1.updateNN()
#plt.clf(),plt.plot(L),plt.savefig('foo'+str(time)+'.png')
plt.clf(),plt.plot(L),plt.show(block=False),plt.pause(0.01)
#print(actual)
exec(open('plotpolicy.py').read())
score=ag1.meanreward*100

if score>bestscore:
    bestscore=score;
    ag1.savebestSD()
    torch.save(ag1,'monagent.pt')

#ag1.savebestSD()

print(time,ag1.meanerror,bestscore,score)

Rs.append(ag1.meanreward)
#params=list(ag1.NNp.state_dict().items())
#plt.imshow(params[0][1]),plt.show()
