import sys, os
sys.path.append(os.pardir)

import numpy as np
import math,random
import cv2
from scipy.special import softmax


# gene_length=4
iteration=21
population_size=100
test_label=np.array([0,1,0,0,0,0,0,0,0,0])
test_label=np.expand_dims(test_label,axis=0)


def calculateConfidence(person,predict):
    person=np.expand_dims(person,axis=0)
    confidence=softmax(predict(person))[0,np.argmax(test_label)]
    return confidence

# def calculateConfidence(person,label,predict):
#     person=np.expand_dims(person,axis=0)
#     confidence=softmax(predict(person))[0,np.argmax(label)]
#     return confidence


def selectParent(confidence_list):
    rand=random.random()
    current_confidence_sum=.0
    location=0
    confidence_sum=np.sum(confidence_list)
    for i in range(confidence_list.size):
        current_confidence_sum+=(confidence_list[i]/confidence_sum)
        if(rand<current_confidence_sum):
            location=i
            break
    return location

# under single cross, best loss
def crossover(father,mother,predict):
    best_confidence=-10000.
    child=np.zeros(784)
    # print('father_len',len(father))
    for i in range(father.size):
        current_child=np.zeros(784)
        current_child=np.append(father[0:i],mother[i:])
        # print('current_child:',current_child.shape)
        current_confidence=calculateConfidence(current_child, predict)
        if current_confidence>best_confidence:
            best_confidence=current_confidence
            # change:添加copy
            child=current_child.copy()
    return child

# under single cross, randomly loss
def crossover2(father,mother):
    location=int(random.random()*len(father))
    return np.append(father[0:location],mother[location:])

# under muti-cross, randomly loss
def crossover3(father,mother):
    child=[]
    for i in range(784):
        location=int(random.random()*gene_length)
        child.append(father[i*gene_length:i*gene_length+location]+mother[i*gene_length+location:(i+1)*gene_length])
    return child

# under muti-cross, best loss in specital loop
def crossover4(father,mother,loss):
    _loop=100
    loss=10000.
    child=[]
    for l in range(_loop):
        current_child=[]
        for i in range(784):
            location=int(random.random()*gene_length)
            current_child.append(father[i*gene_length:i*gene_length+location]+mother[i*gene_length+location:(i+1)*gene_length])
        current_loss=loss(child)
        if(current_loss<loss):
            loss=current_loss
            child=current_child
    return child

# all pixel to mutate
def muteChild(child):
    _mutate_rate=.05
    temp_child=child
    for i in range(temp_child.size):
        if(random.random()<_mutate_rate):
            temp_child[i] = 0. if temp_child[i]==1 else 1.
    return temp_child


# randomly single pixel to mutate
def muteChild2(child):
    _mutate_rate=.2
    temp_child=child
    if(random.random()<_mutate_rate):
        index=int(random.random()*len(temp_child))
        temp_child[index]= 0 if temp_child[index]==1 else 1
    return temp_child

def getPopulation():
    population=np.zeros((population_size, 28, 28, 1),dtype=np.float32)
    for i in range(population.shape[0]):
        for j in range(population.shape[1]):
            for k in range(population.shape[2]):
                if (random.random()>.5):
                    population[i][j]=1
    return population
