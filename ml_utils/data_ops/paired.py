import numpy as np
import random

def create_negative_sampler(possibles):
    """Function creator, create a function that exclude 1 type of label
       
       # this line returns a function
       choose_label_not = create_negative_sampler(['a','b','c','d'])

       # use the function to sample, anything but not 'c'
       [choose_label_not('c') for _ in range(10)]

       >> ['d', 'b', 'a', 'd', 'b', 'b', 'a', 'b', 'a', 'b']

       # possibles can also be an integer, like 
       choose_label_not = create_negative_sampler(7)
       [choose_label_not(6) for _ in range(25)]
       >>> [3, 2, 2, 3, 1, 3, 4, 5, 2, 5, 4, 2, 0, 2, 5, 2, 5, 3, 4, 4, 1, 4, 4, 2, 3]
    
    """
    if isinstance(possibles, int):
        possibles = list(range(possibles))
    def not_class(label):
        candidates = possibles.copy()
        candidates.remove(label)
        return random.choice(candidates)
    return not_class

def create_paired_samples(x, y):
    """When 2 sample in datasset (x, y), they get a label 1, else 0
    return pairs = [[x,y],....], labels = [0/1,....])
    """

    # a list of numpy array
    # [[indicies with label 0], [indices with label 1], [indices with label 2]....[indices with label 9]]
    digit_indices = [np.where(y == i)[0] for i in range(10)]

    # number of samples of (the class with least samples)
    n = min([len(digit_indices[d]) for d in range(10)]) - 1

    not_class = create_negative_sampler(10)

    pairs = []
    labels = []
    # for each class
    for d in range(10):
        # loop n iterations
        for i in range(n):
            # in each iteration, generate 2 samples
            z1, z2 = [digit_indices[d][i], digit_indices[d][i+1]]
            pairs += [x[z1],x[z2]]

            d_not = not_class(d)
            z1, z2 = [digit_indices[d][i], digit_indices[d_not][i+1]]
            pairs += [x[z1],x[z2]]

            labels += [1, 0]
    return np.array(pairs), np.array(labels).astype('float32')