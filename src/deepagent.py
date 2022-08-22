import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical
import random
import time

import envmanager as em

def tstr(tensor):
    return str(tensor.tolist()[0])

class restate():
    def __init__(self, actions, lp, masks, t_state, rewards_t):
        self.actions = actions
        self.lp = lp
        self.masks = masks
        self.t_state = t_state
        self.rewards_t = rewards_t

    def pr(self):
        print("state: {")
        print("    "+tstr(self.actions[0]))
        print("    "+tstr(self.actions[1]))
        print("    "+tstr(self.lp[0]))
        print("    "+tstr(self.lp[1]))
        print("    "+tstr(self.t_state[0]))
        print("    "+tstr(self.t_state[1]))
        print("    "+tstr(self.t_state[2]))
        print("    "+tstr(self.t_state[3]))
        print("}")

    def graphical(self):
        if not type(self.rewards_t) == list:      
            print("reward: "+tstr(self.rewards_t))
        board1 = self.t_state[0].tolist()[0]
        board2 = self.t_state[1].tolist()[0]
        for i in range(12):
            s = ""
            for j in range(7):
                if (board1[i][j] == 0):
                    s += " "
                    continue
                s += str(int(board1[i][j]))
            s += "         "
            for j in range(7):
                if (board2[i][j] == 0):
                    s += " "
                    continue
                s += str(int(board2[i][j]))

            print(s)

    def to(self,device):
        self.actions[0].to(device)
        self.actions[1].to(device)
        self.lp[0].to(device)
        self.lp[1].to(device)
        self.t_state[0].to(device)
        self.t_state[1].to(device)
        self.t_state[2].to(device)
        self.t_state[3].to(device)
        if not type(self.rewards_t) == list:
            self.rewards_t.to(device)


class tensorReplay():
    def __init__(self):
        self.states = []
        self.pair = [None, None]

    def addState(self, actions, lp, masks, t_state, rewards_t):
        self.states.append(restate(actions,lp,masks,t_state,rewards_t))
        self.pair[0] = self.pair[1]
        self.pair[1] = self.states[len(self.states)-1]

    def randPair(self):
        i = random.randint(0,len(self.states)-3)
        self.states[i].to('cuda')
        self.states[i+1].to('cuda')
        return self.states[i], self.states[i+1]


def mask(tensor1, mask, tensor2):
    """mask is a boolean or integer tensor."""

    A = tensor1 * (~mask) #delete masked entries in tensor1
    B = tensor2 * mask #extract masked entries from tensor2

    return A+B #insert B into A

class ResConvNet(nn.Moduke):
    def __init__(self,device="cuda"):
        super(ResConvNet, self).__init__()

        self.device = torch.device(device)

        


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

    def tensify(self,states):

        x1 = torch.tensor([state.boards[0] for state in states],dtype=torch.float32, device=self.device)
        x2 = torch.tensor([state.boards[1] for state in states],dtype=torch.float32, device=self.device)
        q1 = torch.tensor([[state.type[0]]+state.queue[0] for state in states],dtype=torch.float32, device=self.device)
        q2 = torch.tensor([[state.type[1]]+state.queue[1] for state in states],dtype=torch.float32, device=self.device)

        return x1,x2,q1,q2

    def forward(self, inputs, side=0):

        x1, x2, q1, q2 = inputs

        if (side == 1):
            x1, x2 = x2, x1
            q1, q2 = q2, q1

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
    def __init__(self, device="cuda", lr=0.00001, entr=0.001):
        self.nn = TromisNet(device=device)
        self.device = torch.device(device)
        self.nn.to(self.device)
        self.tbuffers = [[]]
        self.loss = torch.tensor([0],dtype=torch.float32, device=self.device)
        self.optimizer = torch.optim.Adam(self.nn.parameters(), lr=lr)
        torch.autograd.set_detect_anomaly(False)
        self.back_counts = 0
        self.AC_weight = 1
        self.entropy_weight = entr

        self.nograd_loss = 0

    def agent(self,states,moves,ID,side=0, grad=True): #states is an array of boardStates
        mask = [[True] * 440 for i in range(len(states))]
        for i in range(len(states)):
            for j in range(len(moves[i])):
                mask[i][moves[i][j][0]] = False

        mask = torch.tensor(mask, dtype=torch.bool, device=self.device)

        t_states = self.nn.tensify(states)

        if not grad:
            with torch.no_grad():
                out = self.nn.forward(t_states,side=side)
        else:
            out = self.nn.forward(t_states,side=side)

        v = out[0]
        pi = out[1]
        pi = pi.masked_fill_(mask,-float("Inf"))
        pi = F.softmax(pi,dim=1)

        d = Categorical(pi)

        entropy = d.entropy()

        action = d.sample()
        lp = d.log_prob(action)
        lp = torch.unsqueeze(lp,0)
        move = action.clone().tolist()

        for i in range(len(states)): #can remove this later, if the env accepts legal placements instead of input sequences
            for j in range(len(moves[i])):
                if (moves[i][j][0] == move[i]):
                    move[i] = moves[i][j]
                    break

        return move, action, v, lp, mask, t_states, entropy

    def fast_agent(self,t_state,masks,actions,side=0,grad=True):
        """(also) for off policy training. Mask has already been computed for one state, so we reuse it without any CPU interaction / memory transfers required. """

        if not grad:
            with torch.no_grad():
                out = self.nn.forward(t_state,side=side)
        else:
            out = self.nn.forward(t_state,side=side)

        v = out[0]
        pi = out[1]

        pi = pi.masked_fill_(masks[side], -float("Inf"))
        pi = F.softmax(pi, dim=1)

        d = Categorical(pi)

        e = d.entropy()

        lp = d.log_prob(actions[side])
        lp = torch.unsqueeze(lp,0)

        return v, lp, e

    def optimize(self):

        #print(self.back_counts)
        self.back_counts += 1
        #print("optimizing")

        self.optimizer.zero_grad(set_to_none=True)

        self.nograd_loss += self.loss

        self.loss.to(self.device)
        self.loss.backward()
        self.optimizer.step()

        #print("done optimizing")

        #with open("./losses.txt", 'a') as file:
            #file.write("\n")
            #file.write(str(self.loss.tolist()[0]))

        torch.cuda.empty_cache()

        self.loss = torch.tensor([0],dtype=torch.float32, device=self.device)

    def v_trace(self, oldValues, newValues, rewards, lpi, lmu, entropy):
        #rewards is already a tensor

        #print("reward: " + tstr(rewards))
        rmask = torch.ge(rewards,0);
        nvm = [newValues[0].detach(),newValues[1].detach()]
        #print("p1 old eval: "+tstr(oldValues[0]))
        #print("p2 old eval: "+tstr(oldValues[1]))
        nvm[0] = mask(nvm[0], rmask, rewards)
        nvm[1] = mask(nvm[1], rmask, (-1)*torch.add(rewards,-1)) #invert rewards for p2
        #print("p1 new eval: "+tstr(nvm[0]))
        #print("p2 new eval: "+tstr(nvm[1]))

        impr = [torch.clamp(torch.exp(lpi[0] - lmu[0]),max=1), torch.clamp(torch.exp(lpi[1] - lmu[1]),max=1)]

        #print("p1 imp ratio: "+tstr(impr[0]))
        #print("p2 imp ratio: "+tstr(impr[1]))

        impr[0] = impr[0].detach();
        impr[1] = impr[1].detach();

        td = [impr[k]*(nvm[k] - oldValues[k].detach()) for k in range(2)]

        #print("p1 TD error: "+tstr(td[0]))
        #print("p2 TD error: "+tstr(td[1]))

        self.loss -= torch.sum(impr[0] * lpi[0] * td[0])
        self.loss -= torch.sum(impr[1] * lpi[1] * td[1])

        self.loss += F.mse_loss(oldValues[0] + td[0],oldValues[0]) * 0.5 * self.AC_weight #oldValues[0] + td[0] is the v-trace target.
        self.loss += F.mse_loss(oldValues[1] + td[1],oldValues[1]) * 0.5 * self.AC_weight #0.5 corrects for the gradient of the square.

        self.loss -= (torch.sum(entropy[0]) + torch.sum(entropy[1])) * self.entropy_weight

    def rtensor(self, rewards):
        rewards = torch.tensor(rewards,dtype=torch.float32, device=self.device,requires_grad=False)
        rewards = torch.unsqueeze(rewards, 1)
        return rewards

    def add_loss(self, oldValues, newValues, rewards, lprobs, entropy, imp=1, selfplay=True):

        rewards = torch.tensor(rewards,dtype=torch.float32, device=self.device,requires_grad=False)
        rewards = torch.unsqueeze(rewards, 1)
        rmask = torch.ge(rewards,0);
        nvm = [newValues[0].detach(),newValues[1].detach()]
        nvm[0] = mask(nvm[0], rmask, rewards)
        nvm[1] = mask(nvm[1], rmask, (-1)*torch.add(rewards,-1)) #invert rewards for p2

        self.loss -= torch.sum(lprobs[0] * (nvm[0] - oldValues[0].detach()))

        self.loss += F.mse_loss(nvm[0], oldValues[0]) * 0.5 * self.AC_weight

        self.loss -= torch.sum(entropy[0]) * self.entropy_weight

        if selfplay:
            self.loss -= torch.sum(lprobs[1] * (nvm[1] - oldValues[1].detach()))
            self.loss += F.mse_loss(nvm[1], oldValues[1]) * 0.5 * self.AC_weight
            self.loss -= torch.sum(entropy[1]) * self.entropy_weight

        if (not imp == 1):
            self.loss = self.loss * torch.tensor(imp,dtype=torch.float32, device=self.device)
        
        return rewards

    def reset_loss(self):
        self.nograd_loss = torch.tensor([0],dtype=torch.float32, device=self.device, requires_grad=False)

    def write_loss(self):
        with open("./losses.txt", 'a') as file:
            file.write("\n")
            file.write(str(self.back_counts) + " " + str(self.nograd_loss.tolist()[0]))

    def save(self, path):
        torch.save(self.nn.state_dict(), path)

    def load(self, path):
        self.nn.load_state_dict(torch.load(path))
        self.nn.eval()
        self.nn.to(self.device)