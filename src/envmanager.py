import torch
from subprocess import Popen, PIPE, STDOUT

def diff(a,b):
    try:
        return a - b
    except:
        c = []
        for i in range(len(a)):
            c.append(a[i]-b[i])
        return c

def encode_move(state, x):
    return int(state.type[state.active])*5*11*4 + int(x[0])*11*4 + int(x[1])*4 + int(x[2])

class replayBuffer:
    def __init__(self,batch_size,depth):
        self.batch_size = batch_size
        self.depth = depth
        self.advantages = [] #detached 1D tensors. array of batch_size-vectors.
        self.rewards = [] #non tensors

    def addValue(self, v):
        try:
            values = v.copy()
            self.advantages.append(values.copy())
        except:
            values = v.clone()
            self.advantages.append(values.clone())
        if not len(self.advantages) == 1:        
            for i in range(self.batch_size):
                if self.rewards[len(self.rewards)-1][i] != -1:
                    values[i] = self.rewards[len(self.rewards)-1][i]
            self.advantages[len(self.advantages)-2] = diff(values, self.advantages[len(self.advantages)-2])

    def propRewards(self):
        reward = -1000
        for i in range(self.batch_size):
            for j in range(self.depth):
                k = self.depth - j - 1
                if self.rewards[k][i] != -1:
                    reward = self.rewards[k][i]
                else:
                    self.rewards[k][i] = reward

class boardState:
    def __init__(self, raw):
        if len(raw) < 5:
            self.terminal = True
            if raw == "win":
                self.outcome = 1
            elif raw == "loss":
                self.outcome = 0
            elif raw == "draw":
                self.outcome = 0.5
            return
        self.terminal = False
        raw = raw.split("&")
        raw[0] = raw[0].split("|")
        raw[1] = raw[1].split("|")
        raw[2] = raw[2][0]
        self.boards = [[],[]]
        self.queue = []
        self.type = [0,0]
        self.htype = [0,0]
        self.garbage = []
        for i in range(2):
            for y in range(12):
                self.boards[i].append([])
                for x in range(7):
                    self.boards[i][y].append(int(raw[i][0][7*y + x] != "0"))
            self.queue.append([])
            for j in range(4):
                self.queue[i].append(int(raw[i][1][j]))
            self.type[i] = int(raw[i][2])
            self.htype[i] = int(raw[i][3])
            self.garbage.append(raw[i][5])
        self.active = int(raw[2])

        self.reward = 0.5
        self.value = 0.5

    def print(self):
        print(self.boards)
        print(self.queue)
        print("active pieces "+str(self.type))
        print(self.garbage)
        print("playing as "+str(self.active))

class multiEnv:
    def __init__(self):
        self.pool = []

    def startall(self, pool_size, batch_size):
        self.batch_size = int(batch_size/pool_size)
        for i in range(pool_size):
            self.pool.append(Popen(["tritris.exe", str(self.batch_size)], stdout=PIPE, stdin=PIPE, stderr=PIPE))

    def killall(self):

        for i in range(len(self.pool)):
            self.pool[i].kill()

        self.pool = []

    def nextTurn (self, batch_size, agents, terminate=False):

        states = []
        moves = ([],[])
        rewards = [-1]*batch_size

        for pi in range(len(self.pool)):

            p = self.pool[pi]

            for i in range(self.batch_size):
                state = boardState(p.stdout.readline().decode().strip())
                if state.terminal:
                    rewards[i] = state.outcome
                    state = boardState(p.stdout.readline().decode().strip())

                states.append(state)

                for k in range(2):
                    m = p.stdout.readline().decode().strip().split("|")
                    m.pop()

                    for j in range(len(m)):
                        m[j] = m[j].split(":")
                        m[j] = [encode_move(state,[m[j][0],m[j][1],m[j][2]]), m[j][3]]

                    moves[k].append(m)
            #print("state got")

        
        mvs = [[],[]]
        actions = [[],[]]
        values = [[],[]]
        lprob = [[],[]]
        masks = [[],[]]
        entr = []

        mvs[0], actions[0], values[0], lprob[0], masks[0], t_state, entr = agents[0](states, moves[0],side=0)

        t_state = []

        mvs[1], actions[1], values[1], lprob[1], masks[1], t_state, entr = agents[1](states, moves[1],side=1)

        #print(mvs)

        for pi in range(len(self.pool)):
            p = self.pool[pi]
            #print("making moves on process "+str(pi))
            for i in range(self.batch_size):
                #print("move #"+str(i))
                for k in range(2):
                    move = mvs[k][self.batch_size*pi+i]
                    move[1] = " ".join(list(move[1])) + " 6"
                    p.stdin.write((move[1]+'\n').encode())
            p.stdin.flush()

        return values, rewards, lprob, masks, actions, t_state