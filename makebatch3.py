t0=tm.time();

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ag1.NN=ag1.NN.to(device)
# ag1.NN.mask2=ag1.NN.mask2.to(device)
# ag1.NN.mask1=ag1.NN.mask1.to(device)
# ag1.NNp=ag1.NNp.to(device)
# ag1.NNp.mask2=ag1.NNp.mask2.to(device)
# ag1.NNp.mask1=ag1.NNp.mask1.to(device)
#optimizer_to(ag1.optimizer,device)
time=time+1
duration = ley.__len__();



#
# inputbatch=torch.zeros(Lbatch,Nbatch,ag1.ninput*ag1.n_memory+1)
# h0batch=torch.zeros(1,Nbatch,ag1.nhidden)
# abatch=torch.zeros(Lbatch,Nbatch,1).long()
# rbatch=torch.zeros(Lbatch,Nbatch,1)

ag1.NNp.draw_mask() #new dropout mask
ag1.NN.draw_mask() #new dropout mask

#lesi[0:8,:,:].transpose(1,0).reshape(2,4,2).transpose(1,0)
for t in range(10):
    #sampling batch
    inds=random.sample(range(duration-Lbatch-100), Nbatch)
    inds=np.array(inds)
    inputbatch=inputbatch_lib[:,inds,:]
    h0batch=h0batch_lib[:,inds,:]
    abatch=abatch_lib[:,inds,:]
    rbatch=rbatch_lib[:,inds,:]

    # for k in range(Nbatch):
    #     ind=torch.tensor(random.randrange(duration-Lbatch-100))
    #     inputbatch[:,k,:]=lesi[ind:ind+Lbatch,0,:]
    #     h0batch[:,k,:]=lesh[ind,0,:]
    #     abatch[:,k,0]=lesa[ind:ind+Lbatch,0,0]
    #     rbatch[:,k,0]=lesr[ind:ind+Lbatch,0,0]

    #Computing batch loss
    outputbatch,hn=ag1.NNp(inputbatch.to(device),h0batch.to(device))
    actualbatch=outputbatch.gather(2, abatch.to(device))
    actualbatch=actualbatch[0:Lbatch-1,:,:]
    outputbatch2,hn2=ag1.NN(inputbatch.to(device),h0batch.to(device))
    targetbatch=rbatch[0:Lbatch-1,:,:].to(device)+ag1.gamma*outputbatch2[1:Lbatch,:,:].max(2)[0].unsqueeze(2).to(device)


    ag1.optimizer.zero_grad()     # zeroes the gradient buffers of all parameters
    loss=ag1.criterion(actualbatch.to(device), targetbatch.to(device))
    #loss.backward(retain_graph=True)
    loss.backward()

    for param in ag1.NNp.parameters():
        param.grad.data.clamp_(-1, 1)
    ag1.optimizer.step()
    ag1.meanerror=math.sqrt(loss)
    L.append(math.log10(ag1.meanerror))

#plt.clf(),plt.plot(RT,Rtrain),plt.ylim(-0.005,max(TRAIN_MAXS)),plt.show(block=False)
#plt.clf(),plt.plot(L),plt.show(block=False)

plt.pause(0.01)



#ag1.savebestSD()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ag1.NN=ag1.NN.to(device)
# ag1.NN.mask2=ag1.NN.mask2.to(device)
# ag1.NN.mask1=ag1.NN.mask1.to(device)
# ag1.NNp=ag1.NNp.to(device)
# ag1.NNp.mask2=ag1.NNp.mask2.to(device)
# ag1.NNp.mask1=ag1.NNp.mask1.to(device)

t1=tm.time();
#print('makebatch3 Elapsed time : '+str(t1-t0)+' s.')
