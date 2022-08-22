
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
