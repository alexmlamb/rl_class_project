from nn_layers import param_init_lngru
from nn_layers import lngru_layer
from nn_layers import ff_layer
from nn_layers import param_init_fflayer

def init_params(p):

    p = param_init_lngru(options, params, prefix='lngru', nin=200, dim=512)

    p = param_init_fflayer({}, p, 'ff',512,1)

    return p

'''
    probs is (time x example x token)
    rewards is (example)

    Network goes over probs and estimates final reward using an RNN.  

    Initial architecture: one layer lngru followed by mean-pooling followed by one fully-connected layer.  

'''
def network(probs,rewards, params):
    
    h1 = lngru_layer(probs)

    h2 = ff_layer(h1)

    estimated_rewards = h2

    loss = T.mean(T.sqr(rewards - estimated_rewards))

    return estimated_rewards, loss

if __name__ == "__main__":
    probs = theano.shared(rng.normal(size=(10,128,200)).astype('float32'))
    rewards = theano.shared(rng.normal(size=(128,)).astype('float32'))

    params = init_params({})

    est_r, loss = network(probs,rewards,params)

    f = theano.function([],outputs = [loss])



