
def to_list(tensor): # Tddo move to utils
    return tensor.detach().cpu().tolist()
