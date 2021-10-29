previousdur=RT[-1]
start_time = tm.time()

while time<3000000:
    if time%100==0:
        #plt.clf(),plt.imshow(params[4][1].to("cpu"),vmin=-3,vmax=3),plt.savefig('RESULTS/weights/nfoo'+str(time)+'.png')
        #plt.clf(),plt.imshow(params[1][1].to("cpu"),vmin=-3,vmax=3),plt.savefig('RESULTS/weights/nfif'+str(time)+'.png')
        exec(open('savelast.py').read())
    if time%10==0:
        ag1.updateNN()
    if time%100==0: #no point measuring if no update
        ley=ytrain;
        exec(open('plotpolicy.py').read())
        scoretrain=score
        ley=yval;
        exec(open('plotpolicy.py').read())
        scoreval=score
        ley=ytrain;
        Rtrain.append(scoretrain)
        Rval.append(scoreval)
        T.append(time)
        RT.append(previousdur+(tm.time() - start_time)/3600)
        TRAIN_INDS.append(train_ind);
        VAL_INDS.append(val_ind);
        TRAIN_MAXS.append(maxrtrain);
        VAL_MAXS.append(maxrval);

        #L.append(math.log10(ag1.meanerror))
        #plt.plot(Rtrain,linewidth=1),plt.ylim(-0.0025,0.0025),plt.xlim(0,Rtrain.__len__()),plt.show(block=False),plt.pause(0.01)
    if time%10==0:
        exec(open('load_data.py').read())
        exec(open('onerun3.py').read())
        exec(open('plotall.py').read())
    exec(open('makebatch3.py').read())
    print(time,ag1.meanerror,scoretrain,scoreval,maxrtrain)





exec(open('savelast.py').read())

#plt.plot(Rtrain),plt.show() #plt.plot(Rval)
mtrain=torch.tensor(Rtrain[-100::1]).mean()
strain=torch.tensor(Rtrain[-100::1]).std()
mval=torch.tensor(Rval[-100::1]).mean()
sval=torch.tensor(Rval[-100::1]).std()
mL=torch.tensor(L[-100::1]).mean()
sL=torch.tensor(L[-100::1]).std()
print('Training gain : '+str(float(mtrain))+'%+-'+str(float(strain)))
print('Validation gain : '+str(float(mval))+'%+-'+str(float(sval)))
print('Loss : '+str(float(mL))+'%+-'+str(float(sL)))
print("--- %s seconds ---" % (tm.time() - start_time))
#
# plt.clf(),plt.plot(T,Rtrain),plt.plot([0,T[-1]],[maxrtrain,maxrtrain]),plt.ylim(-0.005,max(TRAIN_MAXS)*1.05),plt.xlim(0,time),plt.xlabel("Iterations"),plt.ylabel("Reward"),plt.title('Training'),plt.savefig('RESULTS/graphs/Rtrain'+str(simid)+'.png')
# plt.clf(),plt.plot(T,Rval),plt.plot([0,T[-1]],[maxrval,maxrval]),plt.ylim(-0.005,max(VAL_MAXS)*1.05),plt.xlim(0,time),plt.xlabel("Iterations"),plt.ylabel("Reward"),plt.title('Validation'),plt.savefig('RESULTS/graphs/Rval'+str(simid)+'.png')
# plt.clf(),plt.plot(RT,Rtrain),plt.plot([RT[0],RT[-1]],[maxrtrain,maxrtrain]),plt.ylim(-0.005,max(TRAIN_MAXS)*1.05),plt.xlim(RT[0],RT[-1]),plt.xlabel("Training time (h)"),plt.ylabel("Reward"),plt.title('Training'),plt.savefig('RESULTS/graphs/RTRtrain'+str(simid)+'.png')
# plt.clf(),plt.plot(RT,Rval),plt.plot([RT[0],RT[-1]],[maxrval,maxrval]),plt.ylim(-0.005,max(VAL_MAXS)*1.05),plt.xlim(RT[0],RT[-1]),plt.xlabel("Training time (h)"),plt.ylabel("Reward"),plt.title('Validation'),plt.savefig('RESULTS/graphs/RTRval'+str(simid)+'.png')
plt.clf(),plt.plot(L),plt.ylim(-5,-1),plt.xlim(0,time),plt.xlabel("Iterations"),plt.ylabel("Loss"),plt.title('Training Set'),plt.savefig('RESULTS/graphs/L'+str(simid)+'.png')

#plt.clf(),plt.imshow(params[4][1]),plt.savefig('foo'+str(time)+'.png')
#plt.imshow(params[4][1]),plt.show()
zzz= torch.FloatTensor(TRAIN_MAXS)
ttt= torch.FloatTensor(RT)
iii= torch.FloatTensor(TRAIN_INDS)
yyy= torch.FloatTensor(Rtrain)
plt.clf()
for i in range(train_files.__len__()):
    plt.plot(ttt[iii==i].numpy(),(yyy[iii==i]).numpy(),linewidth=0.8)

plt.xlim(0,RT[-1]),plt.ylim(-0.01,0.015),plt.xlabel("Training time (h)"),plt.ylabel("Reward"),plt.title('Training'),plt.savefig('RESULTS/graphs/RTRtrain'+str(simid)+'.png')


zzz= torch.FloatTensor(VAL_MAXS)
ttt= torch.FloatTensor(RT)
iii= torch.FloatTensor(VAL_INDS)
yyy= torch.FloatTensor(Rval)
plt.clf()
for i in range(val_files.__len__()):
    plt.plot(ttt[iii==i].numpy(),(yyy[iii==i]).numpy(),linewidth=0.8)
plt.xlim(0,RT[-1]),plt.ylim(-0.015,0.015),plt.xlabel("Training time (h)"),plt.ylabel("Reward / Max Reward"),plt.title('Training'),plt.savefig('RESULTS/graphs/RTRval'+str(simid)+'.png')
