
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Unsupported value encountered.')

def find_last(string, str):
    last_position=-1
    while True:
        position=string.find(str, last_position+1)
        if position == -1:
            return last_position
        last_position = position