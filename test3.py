from concrete import fhe
import numpy as np
import copy

# configuration = fhe.Configuration(comparison_strategy_preference=fhe.ComparisonStrategy.THREE_TLU_CASTED)
def homoLVQ1(weights, sample,l_sample,alpha):
    epochs = 1
    #weights = np.array(weights,dtype=object)
    # new = np.empty([len(weights),len(weights[0])])
    #weights_cp = copy.deepcopy(weights)
    weights_cp = fhe.identity(weights)
    weights = np.array(weights)
    

    # take the sample
    #distances = np.empty(weightscount,dtype=object)
    #distances = [0] * weightscount
    distances = []
    for w in range(len(weights) ):
        distance = 0
        for i in range(len(sample)):
            distance = distance + ((sample[i] * sample[i])-(2*sample[i]*weights[w][i])+(weights[w][i]*weights[w][i]))
            
        #distances[w] = distance 
        distances.append(distance)
    # distances = fhe.refresh(distances)
    

    min_value = distances[0]
    min_index = 0
    
    K = len(distances)
    # dtype object such that in different python objects can be stored
    # aka Tracer Object from concrete
    A = np.empty(K,dtype=object)

    A[0] = 1
    for i in range(1,K):
        A[i] = 0

    min_index_array=[]
    for k in range(0,K):
        C = min_value < distances[k] # result is an tracer object
        for r in range(1,k):
            A[r] = A[r] * C
        # negate C
        negate = 1 + C * (0-1)
        A[k] = negate
        min_value  = min_value + C * (distances[k]-min_value)
    

    
    min_index = fhe.zero()
    for i in range(len(A)):
        min_index +=  i * A[i]

    """
    #Komplexität dieser Implementierung höher
    for index, value  in enumerate(distances,start=0):
        comp = min_value>value
        min_value, min_index = min_value + (comp) * (value-min_value) , min_index + (comp) * (index-min_index)
    
    """
    """
    prototype = [0,0,0]
    for w in range(len(weights)):
                equal = (min_index == w)
                for i in range(len(weights[w])):
                    prototype[i] += (equal * weights[w][i])

    
    for w, weight in enumerate(weights):
        equal = (min_index == w)
        prototype = [p + equal * w_i for p, w_i in zip(prototype, weight)]
    """
    
    prototype = [0,0,0]
    all_indices = np.arange(len(weights))
    index_selection = min_index == all_indices # one hot encoded array
    #selection_and_zeros = weights[0][0] * index_selection
   
    
    for w,s in zip(weights_cp,index_selection):
        for t in range(len(w)):
            prototype[t] += w[t] * s
    

    """
    block_size = len(weights[0])
    #index_selection = np.repeat(index_selection,block_size) # [0,0,0,1,1,1]
    expanded = []
    for value in index_selection:
        for _ in range(block_size):
            expanded.append(value)
    print(len(expanded))
    components =[]
    for i in range(len(weights)):
        for j in range(len(weights[0])):
            components.append(weights[i][j])

    i = 0
    for c , s in zip(components,expanded):
        prototype[i%3] = c * s
        i+=1
    
    """
    """
    for w,s in zip(weights_cp,index_selection):
        prototype += w *s
    """
    #selection = np.sum(selection_and_zeros)
   

    # prototype = weights[0]

    winner_class = min_index
    
    equal = l_sample == winner_class
    
    
    # compute psi for the attraction repelling scheme, holds -1 or 1
    negate = 1 + (equal) * (0-1)
    psi = equal - negate
    
    for i in range(len(prototype)):
        nom = psi * (sample[i] - prototype[i])
        denom = alpha
        prototype[i] = prototype[i] + fhe.multivariate(lambda nom, denom: nom// denom)(nom, denom)
        # prototype[i] = fhe.refresh(prototype[i])
    
    """
    for w in range(len(weights)):
        mask = (min_index == w)
        for i in range(len(weights[w])):
            weights[w][i] = weights[w][i] + mask * (prototype[i]-weights[w][i])
    """
    alpha +=1
    

    return tuple(prototype)


if __name__=="__main__":
    
    #lvqCompiler = fhe.Compiler(homoLVQ1,{"weights":"encrypted","samples":"encrypted", "labels":"encrypted", "alpha":"encrypted"})
    #lvqCompiler = fhe.Compiler(homoLVQ1,{"weights":"encrypted","samples":"encrypted", "labels":"encrypted", "alpha":"clear"})   
    lvqCompiler = fhe.Compiler(homoLVQ1,{"weights":"encrypted","sample":"encrypted","l_sample":"encrypted","alpha":"encrypted"})
    samples = [[40, 152, 144], [16, 152, 136], [139, 0, 16], [131, 32, 22], [128, 32, 32], [0, 136, 160], [160, 0, 0], [27, 160, 128], [144, 16, 16], [136, 32, 32], [32, 144, 128], [48, 144, 144], [0, 160, 160], [16, 136, 144], [147, 24, 16], [160, 20, 32]]
    labels = [1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0]
    weights= [[160, 112, 16], [160, 160, 160]]


    configuration = fhe.Configuration(
        enable_unsafe_features=True,
        use_insecure_key_cache=True,
        insecure_key_cache_location=".keys",
        #compiler_debug_mode=True,
        #compiler_verbose_mode=True,
        #security_level = 128,
        #show_optimizer=True,
        # To enable displaying progressbar
        show_progress=True,
        # To enable showing tags in the progressbar (does not work in notebooks)
        progress_tag=True,
        # To give a title to the progressbar
        progress_title="Sample",
    )   
    """
    for i in range(len(samples)):
        for w in range(len(samples[i])):
            samples[i][w] = int(samples[i][w]/10)

    for i in range(len(weights)):
        for w in range(len(weights[i])):
            weights[i][w] = int(weights[i][w]/10)
    """
    samples = samples[:4]
    labels = labels[:4]
    
    """ works with two parameters 
    winner_inputset =[([[160, 112, 16],[160, 160, 160]],samples), ([[160, 160, 160],[170, 170, 170]],samples),([[170, 170, 170],[150, 150, 150]],samples),([[150, 150, 150],[170, 170, 170]],samples),([[160,160, 160],[180, 150, 140]],samples),
                    ([[160, 112, 16],[160, 160, 160]],samples), ([[160, 160, 160],[170, 170, 170]],samples),([[170, 170, 170],[150, 150, 150]],samples),([[150, 150, 150],[170, 170, 170]],samples),([[160, 160, 160],[180, 150, 140]],samples)]
    """
    winner_inputset =[([[160, 112, 16],[160, 160, 160]],max(samples),0,5), ([[160, 160, 160],[170, 170, 170]],max(samples),1,8),([[170, 170, 170],[150, 150, 150]],max(samples),0,6),([[150, 150, 150],[170, 170, 170]],max(samples),0,5),([[160,160, 160],[180, 150, 140]],max(samples),0,6),
                    ([[160, 112, 16],[160, 160, 160]],min(samples),1,10), ([[160, 160, 160],[170, 170, 170]],min(samples),0,9),([[170, 170, 170],[150, 150, 150]],min(samples),1,7),([[150, 150, 150],[170, 170, 170]],min(samples),1,10),([[160, 160, 160],[180, 150, 140]],min(samples),1,9)]


    #inputset = [(inputset_weights,samples,labels,5),(inputset_weights,samples,labels,10)]
    #inputsetWithoutAlpa=[(inputset_weights,samples,labels),(inputset_weights,samples,labels)]
    print("Starte Kompillieren")
    lvqCircuit = lvqCompiler.compile(winner_inputset,configuration=configuration)
    #lvqCircuit = lvqCompiler.compile(winner_inputset)
    #lvqCircuit = lvqCompiler.compile(inputset)
    print("Kompillieren erfolgreich")
    lvqCircuit.keygen()

    """
    weights_enc, samples_enc, labels_enc , alpha_enc =lvqCircuit.encrypt(weights,samples,labels,5)
    new_weights = lvqCircuit.run(weights_enc,samples_enc,labels_enc,alpha_enc)
    """
    epochs = 1
    alpha = 5
    j =0
    samples_enc=[]
    labels_enc=[]
    for i in range(len(samples)):
        weights_enc, tmp,tmp2,alpha_enc =lvqCircuit.encrypt(weights,samples[i],labels[i],alpha)
        samples_enc.append(tmp)
        labels_enc.append(tmp2)

    weights_enc=lvqCircuit.run(weights_enc,samples_enc[j],labels_enc[j],alpha_enc)
    
    """
    print("Starte Iteration")
    
    for i in range(epochs):
        
        # for each sample
        for j in range(len(samples_enc)):
    """
        