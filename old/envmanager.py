import torch

def diff(a,b):
    try:
        return a - b
    except:
        c = []
        for i in range(len(a)):
            c.append(a[i]-b[i])
        return c

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