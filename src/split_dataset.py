import os
import random
'''
randomly split the training set into 10 portions. 9 portions are for training, 
and the other is for development, which will be used for early
stopping or hyperparameter tuning.
'''
random.seed(1)
ratio = 0.9
with open('.././data/train_lc.txt', 'r') as f1:
    lines = f1.readlines()
print(type(lines))
random.shuffle(lines)
# lines = [line for line in lines if line.strip()]
lines_split = int(len(lines) * ratio)
open('.././data/train.txt', 'w').write(''.join(lines[:lines_split]))
open('.././data/dev.txt', 'w').write(''.join(lines[lines_split:]))


