time=time+1

#exec(open('importData.py').read())


#exec(open('onerun3.py').read())
Nbatch=150
Lbatch=100


inputbatch=torch.zeros(Lbatch,Nbatch,ag1.ninput*ag1.n_memory+1)
h0batch=torch.zeros(1,Nbatch,ag1.nhidden)
abatch=torch.zeros(Lbatch,Nbatch,1).long()
rbatch=torch.zeros(Lbatch,Nbatch,1)

ag1.NNp.draw_mask() #new dropout mask
ag1.NN.draw_mask() #new dropout mask

duration = ley.__len__();

for t in range(50):

    #sampling a batch
    for k in range(Nbatch):
        ind=torch.tensor(random.randrange(duration-Lbatch-100))
        ind
        inputbatch[:,k,:]=lesi[ind:ind+Lbatch,0,:]
        h0batch[:,k,:]=lesh[ind,0,:]
        abatch[:,k,0]=lesa[ind:ind+Lbatch,0,0]
        rbatch[:,k,0]=lesr[ind:ind+Lbatch,0,0]

    #ag1.NNp.draw_mask() #new dropout mask
    #ag1.NN.draw_mask() #new dropout mask
    #Computing batch loss
    outputbatch,hn=ag1.NNp(inputbatch.to(device),h0batch.to(device))
    actualbatch=outputbatch.gather(2, abatch.to(device))
    actualbatch=actualbatch[0:Lbatch-1,:,:]
    outputbatch2,hn2=ag1.NN(inputbatch.to(device),h0batch.to(device))
    targetbatch=rbatch[0:Lbatch-1,:,:].to(device)+ag1.gamma*outputbatch2[1:Lbatch,:,:].max(2)[0].unsqueeze(2)


    ag1.optimizer.zero_grad()     # zeroes the gradient buffers of all parameters
    loss=ag1.criterion(actualbatch, targetbatch)
    loss.backward(retain_graph=True)
    #loss.backward()
    for param in ag1.NNp.parameters():
        param.grad.data.clamp_(-1, 1)
    ag1.optimizer.step()
    ag1.meanerror=math.sqrt(loss)
    L.append(math.log10(ag1.meanerror))

plt.clf(),plt.plot(L),plt.savefig('RESULTS/strategies/foo'+str(time)+'.png')
#plt.clf(),plt.imshow(params[4][1]),plt.savefig('foo'+str(time)+'.png')
plt.clf(),plt.plot(L),plt.show(block=False),plt.pause(0.01)
#print(actual)
#exec(open('plotpolicy.py').read())



#ag1.savebestSD()


params=list(ag1.NNp.state_dict().items())
#plt.imshow(params[0][1]),plt.show()
