from subprocess import Popen, PIPE, STDOUT
import multiprocessing as mp
import time
import random
import math
import sys
import pickle

import warnings

warnings.simplefilter("ignore", UserWarning)

import envmanager as em

MAX_ROLLOUT = 100
max_rolls = 1

def encode_move(state, x):
    return int(state.type[state.active])*5*11*4 + int(x[0])*11*4 + int(x[1])*4 + int(x[2])

def decode_move(m):
    r = m % 4
    y = ((m - r)/4) % 11
    x = (((m - r)/4) - y)/11 % 5
    p = math.floor(m/220)
    return int(x), int(y), int(r)

def random_agent (state, moves):
    v = 0.5
    lp = 0
    x = random.choice(moves)
    return x, v, lp

def rollout (p,agent,adversary,rbuffer):

    local_buffer = []

    reward = 0.5
    cur_state = em.boardState(p.stdout.readline().decode().strip())
    old_state = 0
    old_value = 0.5
    turns = 0
    first_turn = True

    coin = random.choice([0,1])

    while (True):
        if (turns % 2 == coin):
            player = agent
        else:
            player = adversary

        if (turns % 2 == coin and first_turn):
            old_state = cur_state

        p.stdin.write(b'9\n')
        p.stdin.flush()
        moves = p.stdout.readline().decode().strip().split("|")
        moves.pop()

        for i in range(len(moves)):
            moves[i] = moves[i].split(":")
            moves[i] = [encode_move(cur_state,[moves[i][0],moves[i][1],moves[i][2]]), moves[i][3]]

        x = player(cur_state, moves) # if player is a deep agent, the NN call goes into a queue and we have to wait for it to come back.

        new_value = x[1]

        move = x[0][1]

        for c in move:    
            p.stdin.write((c+'\n').encode())

        if move != "0":
            if (turns > MAX_ROLLOUT):
                p.stdin.write(b'0\n')
                p.stdin.flush()
            else:
                p.stdin.write(b'6\n')
                p.stdin.flush()
        else:            
                p.stdin.flush()

        cur_state = p.stdout.readline().decode().strip()

        if (cur_state == 'game over'):
            if (turns % 2 == coin):
                reward = 0
            else:
                reward = 1
            break

        cur_state = em.boardState(cur_state)
        if (turns % 2 == coin and not first_turn):
            local_buffer.append(em.Replay(old_state, cur_state, old_value, new_value, x[0], x[2], 0.5))
            old_state = cur_state
            old_value = new_value

        if (turns % 2 == coin and first_turn):
            first_turn = False

        turns += 1

    for i in range(len(local_buffer)):
        rbuffer.addReplay(local_buffer[i], reward)

    return turns, reward

def rolloutBatch(ID, queue, returns, batchsize, save):
    p = Popen(["tritris.exe", str(ID), str(save)], stdout=PIPE, stdin=PIPE, stderr=PIPE)

    rbuffer = em.ReplayBuffer(ID) #rbuffer for this batch

    total_states = 0

    reward = 0

    adversary = random_agent

    def agent(state, moves):
        queue.put(("state",ID,state,moves))
        return returns.recv()

    for x in range(0,batchsize):
        result = rollout(p, agent, adversary, rbuffer)
        total_states += result[0]
        reward += result[1]

    p.kill()

    print(reward/batchsize)

    queue.put((total_states,"replay", ID, rbuffer))

def donothing(a,b,c,d):
    pass

def parallelRollout(proc_count, batchsize, save, epoch):
    import deepagent as da

    a2c = da.ActorCritic()

    #a2c.load("./checkpoints/main.pickle")

    tic = time.perf_counter()

    manager = mp.Manager()

    queue = mp.Queue();

    moves = []
    processes = []

    num = proc_count

    for i in range(num):
        m = mp.Pipe(False)
        moves.append(m)
        p = mp.Process(target=rolloutBatch, args=(i,queue, moves[i][0], batchsize, save))
        p.start()
        processes.append(p)

    total_states = 0
    checked = 0

    agent_time = 0
    opt_time = 0

    def fast_agent(state,moves,id):
        return random_agent(state,moves)

    while True:
        if checked == num:
            break
        job = queue.get()
        if job[0] == "state":
            t = time.perf_counter()
            m = a2c.agent(job[2],job[3],job[1])
            moves[job[1]][1].send(m)
            agent_time += time.perf_counter() - t
            continue
        rbuffer = job[3]
        t = time.perf_counter()
        batch = a2c.sync_batch(rbuffer)
        a2c.add_batch(batch)
        opt_time += time.perf_counter() - t
        total_states += job[0]
        checked += 1

    t = time.perf_counter()
    a2c.optimize()
    opt_time += time.perf_counter() - t
    a2c.save("./checkpoints/epoch"+str(epoch)+".pickle")
    a2c.save("./checkpoints/main.pickle")

    print("done")

    toc = time.perf_counter()

    print(str(toc-tic)+" seconds to run "+str(total_states)+" states")
    print(str(agent_time)+" seconds spent calling agent")
    print(str(total_states/(toc-tic)) + " states / sec")

def event_test():

    tic = time.perf_counter()

    event = mp.Event()
    for i in range(100000):
        event.set()
        event.clear()

    toc = time.perf_counter()

    print(str(toc-tic))

def pworker(p1,p2):
    for i in range(10000):
        arr = [1]*1000
        p1.send(arr)
        p2.recv()
    p1.send(0)

def ptest():
    tic = time.perf_counter()
    pipe1 = mp.Pipe(True)
    pipe2 = mp.Pipe(True)
    num = 1
    processes = []
    for i in range(num):
        p = mp.Process(target=pworker, args=(pipe1[1],pipe2[0]))
        p.start()
        processes.append(p)

    while True:
        val = pipe1[0].recv()
        if val == 0:
            break
        pipe2[1].send(0)

    print("seconds taken "+str(time.perf_counter()-tic))
    pipe1[0].close()
    pipe1[1].close()
    pipe2[0].close()
    pipe2[1].close()

def qworker(q1,q2):
    for i in range(1000):
        arr = [1]*1000
        q1.put(arr)
        q2.get()
    q1.put(0)

def qtest():
    tic = time.perf_counter()
    q1 = mp.Queue()
    q2 = mp.Queue()
    num = 1
    processes = []
    for i in range(num):
        p = mp.Process(target=qworker, args=(q1,q2))
        p.start()
        processes.append(p)

    while True:
        val = q1.get()
        if val == 0:
            break
        q2.put(0)

    print("seconds taken "+str(time.perf_counter()-tic))


def main():
    proc_count = int(sys.argv[1])
    batchsize = int(sys.argv[2])
    save = int(sys.argv[3])

    epoch = 0

    for i in range(100):
        parallelRollout(proc_count, batchsize, save, epoch)
        epoch += 1

if __name__ == '__main__':
    import torch.cuda as cuda

    if (not cuda.is_available()):
        print("machine does not have CUDA installed or available.")
    main()
    #event_test()
    #ptest()