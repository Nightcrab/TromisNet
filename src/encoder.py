
def encode_move(state, x):
    return int(state.type[state.active])*5*11*4 + int(x[0])*11*4 + int(x[1])*4 + int(x[2])

def decode_move(m):
    r = m % 4
    y = ((m - r)/4) % 11
    x = (((m - r)/4) - y)/11 % 5
    p = math.floor(m/220)
    return int(x), int(y), int(r)
