import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple
import random
import matplotlib
import matplotlib.pyplot as plt
import math

#data
duration=3000;
Ilength=10;
ts=torch.arange((duration+Ilength-1));
amplitude=2;
ys=(amplitude*(torch.cos(ts/4)+torch.cos(ts/2)+torch.cos(ts/10)))+100;#+torch.randn((ts.size()))*amplitude/10;
T=[];
L=[];
Rs=[];

class RNN(nn.Module):
    def __init__(self,la_inputsize,la_hiddensize,la_outputsize):
        super(RNN, self).__init__()
        self.gru1 = nn.GRU(input_size=la_inputsize, hidden_size=la_hiddensize, num_layers=1,dropout=0)
        self.fc4 = nn.Linear(la_hiddensize, la_hiddensize, True)
        self.fc5 = nn.Linear(la_hiddensize, la_outputsize, True)
    def forward(self, x, h):
        x1, h = self.gru1(x,h)
        x2 = x1+self.fc4(F.relu(x1))
        x = self.fc5(F.relu(x2))
        return x ,h

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
        self.inputlength=10
        self.hiddenlength=20
        self.na=3;
        self.s=torch.zeros(1,1,self.inputlength+self.hiddenlength);
        self.sp=torch.zeros(1,1,self.inputlength+self.hiddenlength);
        self.NNp= RNN(self.inputlength,self.hiddenlength,self.na); #the net to be trained
        self.NNp.reduce_weights();
        self.NNp.reduce_weights();
        self.NNp.reduce_weights();
        self.NN= RNN(self.inputlength,self.hiddenlength,self.na); #is only updated every so often for stability
        self.NN.load_state_dict(self.NNp.state_dict())
        self.NN.eval();
        self.NNbest= RNN(self.inputlength,self.hiddenlength,self.na); #is only updated every so often for stability
        self.NNbest.load_state_dict(self.NNp.state_dict())
        self.NNbest.eval();

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.NNp.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        #self.optimizer = optim.SGD(self.NNp.parameters(), lr=0.00001)
        self.R = ReplayMemory(100000);
        self.gamma=0.95;
        self.epsilon=1;
        self.meanerror=0;
        self.value=100;
        self.XBT=0;
        self.EUR=100;
        self.meanreward=0;
        self.iterations=0;
        self.record_on=1;
        self.C=0;
        self.lastoutput=[];
        self.lastr=0;
        self.lasth=torch.zeros(1,1,self.hiddenlength);

        self.lasta=0;
        self.best_sd=[];
        self.savebestSD()

    def play(self,I):
        #compute s
        #compute next state
        current_price=I[self.inputlength-2];
        next_price=I[self.inputlength-1];
        #rI=I;
        rI=I/torch.mean(I)-1;

        old_value=self.EUR+self.XBT*current_price;

        input=torch.zeros(1,1,self.inputlength);
        input[0][0][0]=self.EUR/old_value-self.XBT*current_price/old_value;
        input[0][0][1:self.inputlength]=rI[0:self.inputlength-1]

        self.s=torch.cat((input,self.lasth),2) #save state


        #print(s)
        self.lastoutput,h=self.NN(input,self.lasth)

        if random.random() > self.epsilon:
            with torch.no_grad(): #makes it faster
                a=self.lastoutput.max(2)[1].view(1, 1)
        else:
            a=torch.tensor(random.randrange(self.na)).view(1,1)
        #implemente actions ici
        #print(a[0][0])

        self.lasta=a;
        if a==0 :
            self.XBT=0.9*self.XBT+0.997*self.EUR/current_price; #a transaction has a cost of 1% to have less actions
            self.EUR=0;
        if a==1 : #sell
            self.EUR=0.9*self.EUR+0.997*self.XBT*current_price;
            self.XBT=0;


        #else wait
        #calculate new state
        #to make sure it's not clipped
        current_price=next_price;
        self.value=self.EUR+self.XBT*current_price
        next_input=torch.zeros(1,1,self.inputlength);
        next_input[0][0][0]=(self.EUR/self.value-self.XBT*current_price/self.value)
        next_input[0][0][1:self.inputlength]=rI[1:self.inputlength]

        self.sp=torch.cat((next_input,h),2) #save state
        self.lasth=h

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
            self.R.push(self.s,self.lasta,self.sp,self.lastr)

    def trainbatch(self,n):
        transitions=self.R.sample(n);
        batch = Transition(*zip(*transitions))
        s_b=torch.cat(batch.state);
        a_b=torch.cat(batch.action);
        r_b=torch.cat(batch.reward);
        sp_b=torch.cat(batch.next_state);

        inputs,h0s=torch.split(s_b,[self.inputlength,self.hiddenlength],dim=2)
        inputs=inputs.permute(1,0,2) #second dimension is batch
        h0s=h0s.permute(1,0,2)

        outputs,hns=self.NNp(inputs,h0s)
        actuals=outputs.gather(2, a_b.view(1,n,1));
        #print(actual)

        inputsp,h0sp=torch.split(sp_b,[self.inputlength,self.hiddenlength],dim=2)
        inputsp=inputsp.permute(1,0,2) #second dimension is batch
        h0sp=h0sp.permute(1,0,2)

        outputsp,hnsp=self.NN(inputsp,h0sp) #use Hn or H0p ???

        targets=r_b.view(1,n,1)+self.gamma*outputsp.max(2)[0].clone().view(1,n,1)

        #print(targets)
        self.optimizer.zero_grad()     # zeroes the gradient buffers of all parameters
        loss=self.criterion(actuals, targets)
        loss.backward(retain_graph=True)
        for param in self.NNp.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        self.meanerror=math.sqrt(loss)
        #loss.backward(retain_graph=False)

        #print(loss)
    def updateNN(self):
        with torch.no_grad(): #makes it faster
            sd = self.NN.state_dict()
            sdp = self.NNp.state_dict()
            for a in sd:
                sd[a]=(self.C*sd[a]+sdp[a])/(self.C+1)
            self.NN.load_state_dict(sd)

    def savebestSD(self):
        print('NN saved')
        self.NNbest.load_state_dict(self.NN.state_dict())
        torch.save(self.NNbest.state_dict(), 'NNbest.sd')


    def importbestSD(self,path):
        self.NNbest.load_state_dict(torch.load(path))
        self.NNbest.eval()
        self.loadbestSD()


    def loadbestSD(self):
        self.NNp.load_state_dict(self.NNbest.state_dict())
        self.NN.load_state_dict(self.NNbest.state_dict())
        self.NN.eval()
        self.NNp.train()
        self.optimizer = optim.Adam(self.NNp.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

        #self.NN.load_state_dict(self.NNbest.state_dict())

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


time=0

score=0;
bestscore=0;
ag1.epsilon=0.1;

ag1.gamma=0.5;
exec(open('importData.py').read())

exec(open('makebatch.py').read())
Rs=[];
