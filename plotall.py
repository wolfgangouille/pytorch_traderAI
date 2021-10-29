zzz= torch.FloatTensor(TRAIN_MAXS)
ttt= torch.FloatTensor(RT)
iii= torch.FloatTensor(TRAIN_INDS)
yyy= torch.FloatTensor(Rtrain)
plt.clf()
for i in range(train_files.__len__()):
    plt.plot(ttt[iii==i].numpy(),(yyy[iii==i]/zzz[iii==i]).numpy(),linewidth=0.5)
plt.plot([RT[0],RT[-1]],[0,0],linewidth=2)
plt.xlim(0,RT[-1]),plt.ylim(-1.01,1.015),plt.xlabel("Training time (h)"),plt.ylabel("Reward / Max Reward"),plt.title('Training'),plt.show(block=False)

# plt.clf()
# for i in range(train_files.__len__()):
#     plt.plot(ttt[iii==i].numpy(),(yyy[iii==i]/zzz[iii==i]).numpy(),linewidth=0.8)
# plt.xlim(0,RT[-1]),plt.ylim(-1.01,1.015),plt.xlabel("Training time (h)"),plt.ylabel("Reward / Max Reward"),plt.title('Training'),plt.show(block=False)

#print(train_files)
# # #
zzz= torch.FloatTensor(VAL_MAXS)
ttt= torch.FloatTensor(RT)
iii= torch.FloatTensor(VAL_INDS)
yyy= torch.FloatTensor(Rval)
plt.clf()
for i in range(val_files.__len__()):
    plt.plot(ttt[iii==i].numpy(),(yyy[iii==i]/zzz[iii==i]).numpy(),linewidth=0.5)
plt.plot([RT[0],RT[-1]],[0,0],linewidth=2)
plt.xlim(0,RT[-1]),plt.ylim(-1.015,1.005),plt.xlabel("Training time (h)"),plt.ylabel("Reward / Max Reward"),plt.title('Training'),plt.show(block=False)

#max_replay_size=100000;

##added j k l at ~192 h
