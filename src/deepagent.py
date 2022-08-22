import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical
import random
import time

import envmanager as em

class TromisNet(nn.Module):
    def __init__(self,device="cuda"):
        super(TromisNet, self).__init__()

        self.device = torch.device(device)

        self.a1 = nn.Conv2d(1,32,2) #2x2 kernel on a 7x12 input, 6x11x16 out
        self.relu = nn.ReLU()
        #pad tensor to 7x12x32
        self.a2 = nn.Conv2d(32,64,2) #2x2 kernel on 7x12x32 input,6x11x64 out
        #pad tensor to 7x12x64
        self.a3 = nn.Conv2d(64,128,2) #6x11x128 output
        #pad tensor to 6x12x128
        self.mp2 = nn.MaxPool2d(2) # 3x6x128 out
        self.a4 = nn.Conv2d(128,256,2) #2x5x256 output
        #pad tensor to 2x6x256
        self.mp2 = nn.MaxPool2d(2) # 1x3x256 out
        self.flat = nn.Flatten() #768 features
        self.a5 = nn.Linear(768, 576)

        self.b1 = nn.Linear(581, 290)

        self.v1 = nn.Linear(580,290)
        self.v2 = nn.Linear(290,1)
        self.sig = nn.Sigmoid()
        self.p1 = nn.Linear(580, 440)
        self.p2 = nn.Linear(440,440)

    def a(self, x):
        x = torch.unsqueeze(x,1)
        x = x.float()
        x = self.a1(x)
        x = self.relu(x)
        x = F.pad(x,(0,1,0,1), "constant",0)
        x = self.a2(x)
        x = self.relu(x)
        x = F.pad(x,(0,1,0,1),"constant",0)
        x = self.a3(x)
        x = self.relu(x)
        x = F.pad(x,(0,0,0,1),"constant",0)
        x = self.mp2(x)
        x = self.a4(x)
        x = self.relu(x)
        x = F.pad(x,(0,0,0,1),"constant",0)
        x = self.mp2(x)
        x = self.flat(x)
        x = self.a5(x)
        return x

    def forward(self, states):

        x1 = torch.tensor([state.boards[0] for state in states], device=self.device)
        x2 = torch.tensor([state.boards[1] for state in states], device=self.device)
        q1 = torch.tensor([[state.type[0]]+state.queue[0] for state in states], device=self.device)
        q2 = torch.tensor([[state.type[1]]+state.queue[1] for state in states], device=self.device)

        x1 = self.a(x1)
        x2 = self.a(x2)
        x1 = torch.cat((x1,q1),dim=1)
        x2 = torch.cat((x2,q2),dim=1)
        x1 = self.b1(x1)
        x2 = self.b1(x2)

        x = torch.cat((x1,x2),dim=1)

        pi = self.p1(x)
        pi = self.p2(pi)

        v = self.v1(x)
        v = self.v2(v)
        v = self.sig(v)

        return v, pi


class ActorCritic():
    def __init__(self, device="cuda"):
        self.nn = TromisNet(device=device)
        self.device = torch.device(device)
        self.nn.to(self.device)
        self.tbuffers = [[]]
        self.loss = torch.tensor([0],dtype=torch.float32, device=self.device)
        self.optimizer = torch.optim.Adam(self.nn.parameters(), lr=0.0001)
        torch.autograd.set_detect_anomaly(True)

    def agent(self,states,moves,ID): #states is an array of boardStates
        mask = [[True] * 440 for i in range(len(states))]
        for i in range(len(states)):
            for j in range(len(moves[i])):
                mask[i][moves[i][j][0]] = False
        
        mask = torch.tensor(mask, dtype=torch.bool, device=self.device)

        out = self.nn.forward(states)

        v = out[0]
        pi = out[1]
        pi = pi.masked_fill_(mask,-float("Inf"))
        pi = F.softmax(pi,dim=1)

        d = Categorical(pi)

        action = d.sample()
        lp = d.log_prob(action)
        lp = torch.unsqueeze(lp,0)
        action = action.detach().tolist()

        for i in range(len(states)):
            for j in range(len(moves[i])):
                if (moves[i][j][0] == action[i]):
                    action[i] = moves[i][j]
                    break

        return action, v.detach()

    def optimize(self):

        print("optimizing")
        self.optimizer.zero_grad()
        self.loss.to(self.device)
        self.loss.backward()
        self.optimizer.step()

        print("done optimizing")

        with open("./losses.txt", 'a') as file:
            file.write("\n")
            file.write(str(self.loss.tolist()[0]))

        self.loss = torch.tensor([0],dtype=torch.float32, device=self.device)

    def add_loss(self, rbuffer, ID):

        for i in range(len(rbuffer.advantages)):
            self.loss += -torch.sum(self.tbuffers[ID][i][1] * rbuffer.advantages[i])

        for i in range(len(rbuffer.rewards)):
            #self.loss += F.mse_loss(self.tbuffers[ID][i][0], torch.tensor(rbuffer.rewards[i],dtype=torch.float32, device=self.device))
            self.loss += F.mse_loss(self.tbuffers[ID][i][0], torch.tensor(rbuffer.rewards[i],dtype=torch.float32, device=self.device))
            
        self.tbuffers[ID] = []

    def save(self, path):
        torch.save(self.nn.state_dict(), path)

    def load(self, path):
        self.nn.load_state_dict(torch.load(path))
        self.nn.eval()
        self.nn.to(self.device)