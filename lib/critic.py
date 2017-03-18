
def init_params(p):

    return p

'''
    probs is (time x example x token)
    rewards is (example)

    Network goes over probs and estimates final reward using an RNN.  
'''
def network(probs,rewards, params):


