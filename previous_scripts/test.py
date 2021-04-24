import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple
import random
import matplotlib
import matplotlib.pyplot as plt
import math

duration=3000;
Ilength=100;
ts=torch.arange((duration+Ilength-1));
amplitude=2;
ys=(amplitude*(torch.cos(ts/4)+torch.cos(ts/2)+torch.cos(ts/10)))+100;#+torch.randn((ts.size()))*amplitude/10;

T=[];
L=[];
Rs=[];


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(100, 20, True)
        #self.do1 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(20, 20, True)
        #self.do2 = nn.Dropout(p=0.5)
        #self.fc3 = nn.Linear(10, 10, True)
        #self.fc4 = nn.Linear(10, 10, True)
        self.fc5 = nn.Linear(20, 3, True)
    def forward(self, x):
        x1 = F.relu(self.fc1(x))
        #x1=self.do1(x1)
        x2 = x1+F.relu(self.fc2(x1))
        #x2=self.do2(x2)
        #x3 = x2+F.relu(self.fc3(x2))
        #x4 = x3+F.relu(self.fc4(x3))
        x = self.fc5(x2)
        return x
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
        self.NNp= Net(); #the net to be trained
        self.NNp.reduce_weights();
        self.NNp.reduce_weights();
        self.NNp.reduce_weights();
        self.hiddenlength=20;
        self.NN= Net(); #is only updated every so often for stability
        self.NN.load_state_dict(self.NNp.state_dict())
        self.NN.eval();
        self.NNbest= Net(); #is only updated every so often for stability
        self.NNbest.load_state_dict(self.NNp.state_dict())
        self.NNbest.eval();
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.NNp.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-6, amsgrad=False)
        #self.optimizer = optim.SGD(self.NNp.parameters(), lr=0.00001)
        self.R = ReplayMemory(100000);
        self.gamma=0.95;
        self.epsilon=1;
        self.meanerror=0;
        self.value=100;
        self.XBT=0;
        self.EUR=100;
        self.na=3;
        self.meanreward=0;
        self.iterations=0;
        self.lastr=0;
        self.record_on=1;
        self.C=0;
        self.lastoutput=[];
        self.lasta=0;
        self.best_sd=[];
        self.saveSD()

    def play(self,I):
        I=I.view(1,Ilength)
        inputlength=I.size()[1]

        #compute s
        #compute next state
        current_price=I[0][inputlength-2];
        next_price=I[0][inputlength-1];
        #rI=I;
        rI=I/torch.mean(I)-1;
        old_value=self.EUR+self.XBT*current_price;
        s=torch.zeros(1,inputlength);
        s[0][0]=self.EUR/old_value-self.XBT*current_price/old_value;
        s[0][1:inputlength]=rI[0][0:inputlength-1]
        #print(s)
        self.lastoutput=self.NN(s)
        if random.random() > self.epsilon:
            with torch.no_grad(): #makes it faster
                a=self.lastoutput.max(1)[1].view(1, 1)
        else:
            a=torch.tensor(random.randrange(self.na)).view(1,1)
        #implemente actions ici
        #print(a[0][0])
        self.lastoutput=self.lastoutput.unsqueeze(0)
        self.lasta=a;
        if a==0 :
            self.XBT=0.9*self.XBT+0.99*self.EUR/current_price; #a transaction has a cost of 0.3% to have less actions
            self.EUR=0;
        if a==1 : #sell
            self.EUR=0.9*self.EUR+0.99*self.XBT*current_price;
            self.XBT=0;


        #else wait
        #calculate new state
        #to make sure it's not clipped
        current_price=next_price;
        self.value=self.EUR+self.XBT*current_price
        sp=torch.zeros(1,inputlength);
        sp[0][0]=(self.EUR/self.value-self.XBT*current_price/self.value)
        sp[0][1:inputlength]=rI[0][1:inputlength]
        #calculate reward
        lar=((self.value-old_value)/old_value).view(1,1)
        self.lastr=lar
        if self.XBT>0:
            self.XBT=max(self.XBT,0.000000000001)
        if self.EUR>0:
            self.EUR=max(self.EUR,0.000000000001)
        self.meanreward=(self.iterations*self.meanreward+lar)/(self.iterations+1) #compute mean reward
        self.iterations+=1
        if self.record_on==1:
            self.R.push(s,a,sp,lar)

    def trainbatch(self,n):
        transitions=self.R.sample(n);
        batch = Transition(*zip(*transitions))
        s_b=torch.cat(batch.state);
        a_b=torch.cat(batch.action);
        r_b=torch.cat(batch.reward);
        sp_b=torch.cat(batch.next_state);
        actual=self.NNp(s_b).gather(1, a_b);
        #print(actual)
        targets=r_b+self.gamma*self.NN(sp_b).max(1)[0].view(n,1)
        #print(targets)
        self.optimizer.zero_grad()     # zeroes the gradient buffers of all parameters
        loss=self.criterion(actual, targets)
        loss.backward()
        for param in self.NNp.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        self.meanerror=math.sqrt(loss)
        #print(loss)
    def updateNN(self):
        sd = self.NN.state_dict()
        sdp = self.NNp.state_dict()
        for a in sd:
            sd[a]=(self.C*sd[a]+sdp[a])/(self.C+1)
        self.NN.load_state_dict(sd)

    def saveSD(self):
        print('NN saved')
        self.NNbest.load_state_dict(self.NN.state_dict())

    def loadSD(self):
        self.NN.load_state_dict(self.NNbest.state_dict())

    def cleanR(self,maxtresh):
        transitions=self.R.memory;
        batch = Transition(*zip(*transitions))
        s_b=torch.cat(batch.state);
        a_b=torch.cat(batch.action);
        r_b=torch.cat(batch.reward);
        sp_b=torch.cat(batch.next_state);
        actual=self.NNp(s_b).gather(1, a_b).detach();
        #print(actual)
        targets=r_b+self.gamma*self.NN(sp_b).max(1)[0].view(self.R.memory.__len__(),1).detach()
        #print(targets)
        delta=torch.sqrt((actual-targets)**2)
        tresh=torch.mean(delta)/2
        for k in reversed(range(self.R.memory.__len__())):
            if delta[k]<min(tresh,maxtresh):
                self.R.memory.pop(k)
        self.R.position=self.R.__len__();




# Main script Here

ag1=TradingAgent();
p=0;
ag1.C=0;

exec(open('importData.py').read())


ag1.epsilon=1;

exec(open('one_run.py').read())

ag1.gamma=0.95;

for t in range(1000):
    ag1.trainbatch(1000)

ag1.updateNN();

ag1.epsilon=0.5

for k in range(10):
    exec(open('one_run.py').read())

ag1.epsilon=0.2
for k in range(10):
    exec(open('one_run.py').read())



score=ag1.meanreward;

#print(ag1.R.memory)
