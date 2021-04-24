ag1.iterations=0;
ag1.meanreward=0;
ag1.EUR=100;
ag1.XBT=0;
ag1.value=100;
loptimizer=optim.Adam(ag1.NN.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

print(ag1.epsilon)
for k in range(duration):
    ag1.play(ys[k:k+Ilength].view(1,Ilength))

loptimizer.zero_grad()     # zeroes the gradient buffers of all parameters
loss=-ag1.meanreward
loss.backward()
for param in self.NN.parameters():
    param.grad.data.clamp_(-1, 1)
loptimizer.step()
print(loss)
