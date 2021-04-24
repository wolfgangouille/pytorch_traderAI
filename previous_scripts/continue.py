c=10;

for ep in range(10000):
    #ys=(amplitude*(torch.cos(ts/4)+torch.cos(ts/2)+torch.cos(ts/10)))+100+torch.randn((ts.size()))*amplitude/10;

    plt.clf()
    print(ep)

    p=p+1;

    for t in range(50):
        ag1.trainbatch(100)
        L.append(math.log10(ag1.meanerror))

    if ep>c:
        c=c+10
        ag1.epsilon=max(ag1.epsilon*0.95,0.1)
        ag1.gamma=min(ag1.gamma+0.01,0.95)
        ag1.updateNN();
        print('boop')
        exec(open('one_run.py').read())

    if ag1.R.memory.__len__()==ag1.R.capacity:
        #ag1.cleanR(0.1);
        print('R cleaned')



    exec(open('plotpolicy.py').read())

    if ag1.meanreward>score:
        ag1.saveSD()
        #ag1.C=ag1.C+1;
        score=ag1.meanreward
        print('C increased to '+str(ag1.C))

    #if ag1.meanreward<score*0.9:
    #    ag1.loadSD()
    #    ag1.NNp.load_state_dict(ag1.NNbest.state_dict())

    T.append(p)
    Rs.append((ag1.meanreward.detach()))
    print(str(score*100)+' '+str(ag1.meanreward.detach()*100))
    #print(str(ag1.NNbest.state_dict()['fc4.bias'])+' '+str(ag1.NN.state_dict()['fc4.bias']))

    #print(ag1.best_sd)
    #print(ag1.value)
    params=list(ag1.NNp.state_dict().items())
    #plt.imshow(params[0][1])
    #plt.clf()
    #plt.plot(L)
    #plt.savefig('foo'+str(ep).zfill(4)+'.png')
    #plt.show(block=False)
    #plt.pause(0.01)


print(ag1.epsilon)
plt.figure()
plt.plot(T,Rs,'r')
plt.show(block=True)

plt.figure()
plt.plot(L,'b')
plt.show(block=True)
