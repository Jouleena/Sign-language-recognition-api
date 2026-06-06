import numpy as np 
import os 

words = ['hello' ,"please" , 'yes']
for word in words:
    sample = np.load(f'dataset/{word}/000.npy')
    print(f'---{word}---')
    print('shape:' , sample.shape)
    print('frame 0: ' , sample[0])
    print('frame 0: ' , sample[0])
    print()