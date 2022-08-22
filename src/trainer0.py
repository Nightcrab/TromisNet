from subprocess import Popen, PIPE, STDOUT
import time
import random
import math
import sys
import pickle

import warnings

warnings.simplefilter("ignore", UserWarning)

import envmanager as em
import deepagent as da

MAX_ROLLOUT = 100
max_rolls = 1
PATH = "./checkpoints/conv_conv_0"

def encode_move(state, x):
    return int(state.type[state.active])*5*11*4 + int(x[0])*11*4 + int(x[1])*4 + int(x[2])

def decode_move(m):
    r = m % 4
    y = ((m - r)/4) % 11
    x = (((m - r)/4) - y)/11 % 5
    p = math.floor(m/220)
    return int(x), int(y), int(r)

def random_agent (states, moves):
    action = []
    for i in range(len(moves)):
        action.append(random.choice(moves[i]))
    value = [0.5] * len(states)
    return action, value

def defMM ():
    return random_agent, random_agent

outcomes = [0,0,0]

env_time = 0

def nextTurn (batch_size, p, agents, terminate=False):
    global outcomes
    global env_time

    states = []
    moves = ([],[])
    rewards = []

    for i in range(batch_size):
        t = time.perf_counter()
        state = em.boardState(p.stdout.readline().decode().strip())
        if state.terminal:
            rewards.append(state.outcome)
            outcomes[int(state.outcome*2)] += 1
            state = em.boardState(p.stdout.readline().decode().strip())
        elif not terminate:
            rewards.append(-1)
        env_time += time.perf_counter() - t
        states.append(state)

        for k in range(2):
            m = p.stdout.readline().decode().strip().split("|")
            m.pop()

            for j in range(len(m)):
                m[j] = m[j].split(":")
                m[j] = [encode_move(state,[m[j][0],m[j][1],m[j][2]]), m[j][3]]

            moves[k].append(m)

    actions = [0,0]
    values = [0,0]

    actions[0], values[0] = agents[0](states, moves[0])
    actions[1], values[1] = agents[1](states, moves[1])

    if terminate:
        rewards = values[0]

    for i in range(batch_size):
        for k in range(2):
            move = actions[k][i]
            move[1] = " ".join(list(move[1])) + " 6"
            p.stdin.write((move[1]+'\n').encode())

    p.stdin.flush()

    return values[0], rewards

states = 0

def rollout (depth, batch_size, match_maker=defMM):

    global states

    p = Popen(["tritris.exe", str(batch_size)], stdout=PIPE, stdin=PIPE, stderr=PIPE)

    players = match_maker()

    rbuffer = em.replayBuffer(batch_size, depth)

    for i in range(depth):
        values, rewards = nextTurn(batch_size, p, players, terminate=(i==depth-1))
        states += batch_size
        rbuffer.rewards.append(rewards)
        rbuffer.addValue(values)

    rbuffer.advantages.pop()

    rbuffer.propRewards()

    p.kill()

    return rbuffer

agent_time = 0

def main (a2c):
    global outcomes
    def deep_agent(state, moves):
        global agent_time
        t = time.perf_counter()
        out = a2c.agent(state,moves,0)
        agent_time += time.perf_counter() - t
        return out

    def mm1():
        return deep_agent, random_agent

    tic = time.perf_counter()

    for i in range(1):
        rbuffer = rollout(100, 1, match_maker=mm1)

    #print(rbuffer.rewards)
    #print(rbuffer.advantages)
    #for x in a2c.tbuffers[0]:
        #print(x[0].clone().detach().tolist())

    print(outcomes[2]/(outcomes[0]+outcomes[2]))

    print(str(states) + " states in " + str(time.perf_counter() - tic) + "s")
    print(str(states/(time.perf_counter() - tic)) + " per second")

    tic = time.perf_counter()

    #a2c.add_loss(rbuffer, 0)
    #a2c.optimize()
    a2c.save(PATH+"main.pickle")

    print("time to optimise: "+str(time.perf_counter()-tic))
    print("agent time: "+str(agent_time))
    print("env time: "+str(env_time))

def train():
    global outcomes
    a2c = da.ActorCritic()
    a2c.load(PATH+"main.pickle")
    for k in range(1000):
        for i in range(1):
            main(a2c)
            print(outcomes)
            states = 0
            agent_time = 0
            env_time = 0
            outcomes = [0,0,0]

        a2c.save(PATH+str(random.choice(range(0,1000)))+".pickle")

def cpu_vs_gpu():
    import matplotlib.pyplot as plt

    a2c0 = da.ActorCritic(device="cpu")
    ct = []

    def ag2(state, moves):
        out = a2c0.agent(state,moves,0)
        return out

    def mm2():
        return ag2, random_agent

    for x in range(11):
        t = time.perf_counter()
        rollout(40,2 ** x,match_maker=mm2)
        ct.append(time.perf_counter() - t)

    a2c1 = da.ActorCritic(device="cuda")

    def ag3(state, moves):
        out = a2c1.agent(state,moves,0)
        return out
        
    def mm3():
        return ag3, random_agent

    gt = []
    for x in range(11):
        t = time.perf_counter()
        rollout(40,2 ** x,match_maker=mm3)
        gt.append(time.perf_counter() - t)

    plt.plot(ct, label="CPU")
    plt.plot(gt, label="GPU")
    plt.ylabel("time (s)")
    plt.xlabel("log_2 batch size")
    plt.legend()
    plt.show()

def count_params():
    a2c = da.ActorCritic()
    return sum(p.numel() for p in a2c.nn.parameters() if p.requires_grad)

train()