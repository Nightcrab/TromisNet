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
PATH = "./checkpoints/conv_conv_elo/"

def decode_move(m):
    r = m % 4
    y = ((m - r)/4) % 11
    x = (((m - r)/4) - y)/11 % 5
    p = math.floor(m/220)
    return int(x), int(y), int(r)

def random_agent (states, moves, side=0):
    action = []
    for i in range(len(moves)):
        action.append(random.choice(moves[i]))
    value = [0.5] * len(states)
    return action, [], value, [0] * len(states), [], [], 0

def defMM ():
    return random_agent, random_agent

outcomes = [0,0,0]

env_time = 0

states = 0
opt_time = 0

def ex_wr(r0, r1): #expected winrate
    return 1/(1+math.pow(10,(r0-r1)/400))

def rollout (depth, batch_size, step, redo, offpol, match_maker=defMM, evaluation=False):

    global states
    global opt_time

    menv = em.multiEnv()
    menv.startall(2, batch_size)

    players = match_maker()

    oldValues = [[0.5]*batch_size,[0.5]*batch_size] #tensor pair
    newValues = [[],[]] #tensor pair
    rewards = [] #list
    lprobs_old = [[],[]] #tensor pair
    lprobs_new = [[],[]] #tensor pair
    entropy = [[],[]]

    replay = da.tensorReplay()

    wins = 0
    losses = 0

    for i in range(depth):

        actions = []
        mask = []
        t_state = []

        newValues, rewards, lprobs_new, masks, actions, t_state = menv.nextTurn(batch_size, players, terminate=(i==depth-1))
        
        if (evaluation):
            for j in range(len(rewards)):
                if rewards[j] == 1:
                    wins += 1
                if rewards[j] == 0:
                    losses += 1
            continue

        replay.addState(actions,lprobs_new,masks,t_state,[])

        if (i > 0):
            t = time.perf_counter()
            rewards_t = step(oldValues, newValues, rewards, lprobs_old, entropy)
            replay.pair[1].rewards_t = rewards_t
            offpol(replay.pair,k=10)

            opt_time += time.perf_counter() - t

        if i < (depth-1):
            oldValues, lprobs_old, entropy = redo(replay.pair[1])

        newValues = None
        lprobs_new = None

        states += batch_size

    menv.killall()

    if (evaluation):
        return wins, losses

    print("on policy done")

    for j in range(0):
        pair = replay.randPair()
        offpol(pair,k=2)
        pair[0].to('cpu')
        pair[1].to('cpu')

    print("off policy done")


agent_time = 0

def main (a2c):
    global outcomes

    def deep_agent(state, moves, side=0):
        global agent_time
        t = time.perf_counter()
        out = a2c.agent(state,moves,0,side=side,grad=False)
        agent_time += time.perf_counter() - t
        return out

    def mm1():
        return deep_agent, random_agent
    def mm2():
        return deep_agent, deep_agent #both are the same agent (same parameters).

    def step(oldValues, newValues, rewards, lprobs,i):
        rewards_t = a2c.add_loss(oldValues, newValues, rewards, lprobs, i)
        a2c.optimize()
        return rewards_t

    def redo(state):
        values = [[],[]]
        lprobs = [[],[]]
        entropy = [[],[]]
        values[0], lprobs[0], entropy[0] = a2c.fast_agent(state.t_state, state.masks, state.actions, side=0)
        values[1], lprobs[1], entropy[1] = a2c.fast_agent(state.t_state, state.masks, state.actions, side=1)
        return values, lprobs, entropy

    def offpol(restates,k=1):
        v0 = [None]*2
        lp0 = [None]*2
        v1 = [None]*2
        lp1 = [None]*2
        entr0 = [None]*2
        entr1 = [None]*2
        states = restates
        #states[0].graphical()
        #states[1].graphical()
        for i in range(k):
            v0[0], lp0[0], entr0[0] = a2c.fast_agent(states[0].t_state, states[0].masks, states[0].actions, side=0)
            v0[1], lp0[1], entr0[1] = a2c.fast_agent(states[0].t_state, states[0].masks, states[0].actions, side=1)
            v1[0], lp1[0], entr1[0] = a2c.fast_agent(states[1].t_state, states[1].masks, states[1].actions, side=0, grad=False)
            v1[1], lp1[1], entr1[1] = a2c.fast_agent(states[1].t_state, states[1].masks, states[1].actions, side=1, grad=False)
            a2c.v_trace(v0, v1, states[1].rewards_t, lp0, states[0].lp, entr0)
            a2c.optimize()

    tic = time.perf_counter()

    for i in range(1):

        #wl = rollout(50,128, step, redo, offpol, match_maker=mm4, evaluation=True)
        #print(str(wl[0]/(wl[0]+wl[1])))
        wl = rollout(50,128, step, redo, offpol, match_maker=mm1, evaluation=True)
        print(str(wl[0]/(wl[0]+wl[1])))


        #with open("./wr.txt", 'a') as file:
            #file.write("\n")
            #file.write(str(wl))

        #rollout(25,2048, step, redo, offpol, match_maker=mm3)

        rollout(50,1024, step, redo, offpol, match_maker=mm2)

    #a2c.save(PATH+"blank.pickle")

    print(str(states) + " states in " + str(time.perf_counter() - tic) + "s")
    print(str(states/(time.perf_counter() - tic)) + " per second")

    print("agent time: "+str(agent_time))
    print("env time: "+str(env_time))

    with open("./rewards0.txt", 'a') as file:
        file.write("\n")
        file.write(str(a2c.back_counts)+" "+str(wl[0]/(wl[0]+wl[1])))

    #a2c.write_loss()
    a2c.reset_loss()

def train():
    global outcomes
    a2c = da.ActorCritic(lr=0.000008*1.4, entr=0.0025)
    a2c.load(PATH+"0.pickle")
    for k in range(1000):
        for i in range(5):
            main(a2c)
            print(outcomes)
            states = 0
            agent_time = 0
            env_time = 0
            outcomes = [0,0,0]

def count_params():
    a2c = da.ActorCritic()
    return sum(p.numel() for p in a2c.nn.parameters() if p.requires_grad)

train()