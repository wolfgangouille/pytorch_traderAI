import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple
import random
import matplotlib
import matplotlib.pyplot as plt
import math
exec(open('init_functions.py').read())
simid=27;

#data

duration=10000;
Ilength=20;
ts=torch.arange((duration+Ilength-1));
amplitude=2;
ys=(amplitude*(torch.cos(ts/4)+torch.cos(ts/2)+torch.cos(ts/10)))+100;#+torch.randn((ts.size()))*amplitude/10;
yval=ys

exec(open('fakedata.py').read())
ley=ys;

class RNN(nn.Module):
    def __init__(self,la_inputsize,la_hiddensize,la_outputsize):
        super(RNN, self).__init__()
        self.inputsize=la_inputsize
        self.hiddensize=la_hiddensize
        self.outputsize=la_outputsize
        self.gru1 = nn.GRU(input_size=la_inputsize, hidden_size=la_hiddensize, num_layers=1,dropout=0)
        self.mask1=torch.ones(1,1,la_hiddensize)
        self.fc4 = nn.Linear(la_hiddensize, la_hiddensize, True)
        self.mask2=torch.ones(1,1,la_hiddensize)
        self.fc5 = nn.Linear(la_hiddensize, la_outputsize, True)
        self.do=0.5;
    def forward(self, x, h):
        x1b, h = self.gru1(x,h)
        x1b=x1b*self.mask1
        x2 = x1b+self.fc4(F.relu(x1b))
        x2=x2*self.mask2
        x = self.fc5(F.relu(x2))
        return x ,h
    def draw_mask(self):
        if self.training:
            self.mask1=torch.rand(1,1,self.hiddensize)>=self.do
            self.mask2=torch.rand(1,1,self.hiddensize)>=self.do
        if self.training==False:
            self.mask1=torch.ones(1,1,self.hiddensize)*(1-self.do)
            self.mask2=torch.ones(1,1,self.hiddensize)*(1-self.do)
    def reduce_weights(self):
        sd = self.state_dict()
        for a in sd:
            sd[a]=sd[a]/10
        self.load_state_dict(sd)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))




class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)



class TradingAgent(object):
    def __init__(self):
        self.ninput=2 #number of dimensions in inputs (eg: price 1, price2, vol1, vol2, time = 5)
        self.n_memory=1 #number of previous timepoints in buffer
        self.nhidden=64 #number of hidden units in each hidden layer
        self.na=3;
        self.position=0 #-1 if short, 1 if long
        self.previous_inputs=torch.zeros(self.n_memory,self.ninput);#Just easier to store like this first

        self.lastinput=torch.zeros(1,1,self.ninput*self.n_memory+1); #this dimensionning is because of how gru work, dim0 is time and dim1 is batch
        #+1 is because we add a variable to define state, -1 if having btc, +1 if having EUR

        self.lastr=0;
        self.lasth=torch.zeros(1,1,self.nhidden);
        self.lasta=0;


        #make neural networks
        self.NNp=RNN(self.ninput*self.n_memory+1,self.nhidden,self.na); #the net to be trained
        self.NNp.apply(weight_init)
        self.NNp.reduce_weights()

        #change reset gate bias
        #self.NNp.state_dict()['gru1.bias_ih_l0'][0:self.nhidden]=1
        #self.NNp.state_dict()['gru1.bias_hh_l0'][0:self.nhidden]=1
        #for i in range(0,self.nhidden):
            #self.NNp.state_dict()['gru1.weight_hh_l0'][i+2*self.nhidden,i]=1

        self.NN= RNN(self.ninput*self.n_memory+1,self.nhidden,self.na); #is only updated every so often for stability
        self.NN.load_state_dict(self.NNp.state_dict())
        self.NN.eval();
        self.NNbest= RNN(self.ninput*self.n_memory+1,self.nhidden,self.na); #is only updated every so often for stability
        self.NNbest.load_state_dict(self.NNp.state_dict())
        self.NNbest.eval();
        self.NNp.draw_mask()
        self.NN.draw_mask()

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.NNp.parameters(), lr=1e-04, betas=(0.9, 0.999), eps=1e-10, weight_decay=0, amsgrad=False)
        #self.optimizer = optim.SGD(self.NNp.parameters(), lr=0.00001)


        #RL parameters
        self.record_on=1;
        self.gamma=0.95;
        self.epsilon=1;
        self.C=0;


        #Evaluate
        self.value=100;
        self.XBT=0;
        self.EUR=100;
        self.meanerror=0;
        self.meanreward=0;
        self.iterations=0;

        self.last_price=100;

        #self.savebestSD()

    def choose_a(self):

        self.lastoutput,self.lasth=self.NN(self.lastinput,self.lasth)
        if random.random() > self.epsilon:
            with torch.no_grad(): #makes it faster
                self.lasta=self.lastoutput.max(2)[1].view(1, 1)
        else:
            self.lasta=torch.tensor(random.randrange(self.na)).view(1,1)



    def compute_reward_and_new_state(self,I):
        #previous action already commited

        #implemente last action here (too late to change action)
        if self.lasta==0 : #buy
            self.XBT=0.95*self.XBT+0.997*self.EUR/self.last_price; #a transaction has a cost of 1% to have less actions
            self.EUR=0;
            self.position=1
        if self.lasta==1 : #sell
            self.EUR=0.95*self.EUR+0.997*self.XBT*self.last_price;
            self.XBT=0;
            self.position=-1
            #else nothing

        #define new price
        self.last_price=I[0,0];

        #calculate reward, relative change in value, due to price fluctuation and fees
        new_value=self.EUR+self.XBT*self.last_price
        self.lastr=((new_value-self.value)/self.value).view(1,1)
        self.value=new_value

        #oldbuff compute new state
        oldbuf=self.previous_inputs.clone()
        #first save data in buffer
        self.previous_inputs[1:self.n_memory,:]=self.previous_inputs[0:self.n_memory-1,:].clone()
        self.previous_inputs[0,:]=I.view(1,self.ninput).clone()

        #preproocessing data and reshaping it
        #substracting mean
        ppI=(self.previous_inputs-oldbuf)/oldbuf; #beware :removing vector from matrix only works in dim 0

        #updating lastininput (processed)
        self.lastinput[0][0][0]=self.position;
        self.lastinput[0][0][1:self.ninput*self.n_memory+1]=ppI[0:self.ninput*self.n_memory].view(1,1,self.ninput*self.n_memory)


        if self.XBT>0:
            self.XBT=max(self.XBT,0.01/self.last_price)
        if self.EUR>0:
            self.EUR=max(self.EUR,0.01)

        self.value=self.EUR+self.XBT*self.last_price


        self.meanreward=(self.iterations*self.meanreward+self.lastr)/(self.iterations+1) #compute mean reward
        self.iterations+=1




    def updateNN(self):
        with torch.no_grad(): #makes it faster
            sd = self.NN.state_dict()
            sdp = self.NNp.state_dict()
            for a in sd:
                sd[a]=(self.C*sd[a]+sdp[a])/(self.C+1)
            self.NN.load_state_dict(sd)

    def savebestSD(self,filename):
        print('NN saved')
        self.NNbest.load_state_dict(self.NN.state_dict())
        torch.save(self.NNbest.state_dict(), filename)


    def importbestSD(self,path):
        self.NNbest.load_state_dict(torch.load(path))
        self.NNbest.eval()
        self.loadbestSD()


    def loadbestSD(self):
        self.NNp.load_state_dict(self.NNbest.state_dict())
        self.NN.load_state_dict(self.NNbest.state_dict())
        self.NN.eval()
        self.NNp.train()
        self.optimizer = optim.Adam(self.NNp.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-07, amsgrad=False)

    def resetOptimizer(self,lelr):
        self.optimizer = optim.Adam(self.NNp.parameters(), lr=lelr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-07, amsgrad=False)
        #self.NN.load_state_dict(self.NNbest.state_dict())



# Main script Here
ag1=TradingAgent();
time=0


ag1.epsilon=1;
ag1.gamma=0.0


time=0
T=[];
L=[];
Rs=[];
score=0;
bestscore=0;


exec(open('onerun3.py').read())
exec(open('makebatch3.py').read())
exec(open('onerun3.py').read())
exec(open('makebatch3.py').read())
exec(open('onerun3.py').read())
exec(open('makebatch3.py').read())
exec(open('plotpolicy.py').read())
ag1.updateNN()

time=0
T=[];
L=[];
Rtrain=[];
Rval=[];

ag1.epsilon=0.1;
ag1.gamma=0.95

score=0;
bestscore=0;

#intitialisation
#for t in range(5):
#    exec(open('onerun3.py').read())
#    exec(open('makebatch3.py').read())
#    ag1.updateNN()

#exec(open('loadlast.py').read())
ag1.NNp.do=0.0
ag1.NN.do=0.0


while time<100000:
    if time%100==0:
        plt.clf(),plt.imshow(params[4][1],vmin=-3,vmax=3),plt.savefig('RESULTS/weights/nfoo'+str(time)+'.png')
        plt.clf(),plt.imshow(params[1][1],vmin=-3,vmax=3),plt.savefig('RESULTS/weights/nfif'+str(time)+'.png')
        exec(open('savelast.py').read())
    if time%10==0:
        ag1.updateNN()
        ley=ys;
        exec(open('plotpolicy.py').read())
        scoretrain=score
        ley=yval;
        exec(open('plotpolicy.py').read())
        scoreval=score
        ley=ys;
    if time%10==0:
        exec(open('onerun3.py').read())
    exec(open('makebatch3.py').read())
    print(time,ag1.meanerror,scoretrain,scoreval)
    Rtrain.append(scoretrain)
    Rval.append(scoreval)


exec(open('savelast.py').read())

#plt.plot(Rtrain),plt.plot(Rval),plt.show()

print(torch.tensor(Rtrain[Rtrain.__len__()-501:Rtrain.__len__()-1:10]).mean())
print(torch.tensor(Rval[Rval.__len__()-501:Rval.__len__()-1:10]).mean())
print(torch.tensor(L[L.__len__()-5001:L.__len__()-1:10]).mean())

#plt.imshow(params[4][1]),plt.show()
