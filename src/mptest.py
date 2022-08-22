from subprocess import Popen, PIPE, STDOUT
import multiprocessing as mp
import time
import random

def rollout (p):
    game_state = p.stdout.readline().decode()
    #print(game_state)

    states = 0;

    while (True):
        p.stdin.write(b'9\n')
        p.stdin.flush()
        moves = p.stdout.readline().decode().strip().split("|")
        moves.pop()
        #print(len(moves))
        for i in range(len(moves)):
            #print(moves[i])
            moves[i] = moves[i].split(":")
            moves[i][0] = int(moves[i][0])
            moves[i][1] = int(moves[i][1])
            moves[i][2] = int(moves[i][2])
        x = random.choice(moves)
        for c in x[3]:    
            #print((c+'\n').encode()) 
            p.stdin.write((c+'\n').encode())
            #p.stdin.flush()
        p.stdin.write(b'6\n')
        p.stdin.flush()
        game_state = p.stdout.readline()
        #print("game_state :" + game_state.decode())
        states += 1
        if (game_state == b'game over\r\n'):
            #print("game over");
            break
        if (states > 100):
            p.stdin.write(b'0\n')
            p.stdin.flush()
            #print("game too long")
            break

    return states

def rollouts(id):
    p = Popen(["tritris.exe", str(id)], stdout=PIPE, stdin=PIPE, stderr=PIPE);

    total_states = 0

    for x in range(0,100):
        total_states += rollout(p)

    p.terminate()
    return total_states

def donothing(id):
    return id+1

def main():
    pool = mp.Pool(mp.cpu_count())
    print(str(mp.cpu_count())+" cores found")
    result = pool.map(rollouts,[0,1,2,3,4])

    return result[0]+result[1]+result[2]+result[3]+result[4]

def singlethread():
    tic = time.perf_counter()
    total = rollouts(0)
    secs = time.perf_counter()-tic
    print(str(total)+" gamestates in "+str(secs)+"s")
    print(str(total/secs) + " gamestates per second")

def multithread():
    if __name__ == "__main__":
        tic = time.perf_counter()
        total = main()
        secs = time.perf_counter()-tic
        print(str(total)+" gamestates in "+str(secs)+"s")
        print(str(total/secs) + " gamestates per second")

#singlethread()
multithread()