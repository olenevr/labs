import numpy as np
import random as rd
from scipy.special import softmax

epsilon = 1
population_size = 100

def getPopulation():
    population = np.zeros((population_size, 28, 28, 1))
    for i in range(population.shape[0]):
        for j in range(population.shape[1]):
            for k in range(population.shape[2]):
                population[i,j,k,0] = rd.random()*epsilon*((-1)**(rd.randint(0,1)))
    return population

def selectionTour(confidence_list,list):
    res = []
    for i in range(0,population_size-1,2):
        j0 = list[i]
        j1 = list[i+1]
        if confidence_list[j0] > confidence_list[j1]:
            res.append(j0)
            res.append(j1)
        else:
            res.append(j1)
            res.append(j0)
    return res

def calculateConfidence(predict):
    confidence=softmax(predict)
    return confidence

def crossover2(father,mother):
    location = int(len(father)/2)
    res = np.zeros((28,28,1))
    for i in range(location):
        res[i] = father[i]
    for i in range(location, len(mother)):
        res[i] = mother[i]

    return res

def crossover(father,mother):
    child = (father+mother)/2

    return child

def muteChild(child):
    _mutate_rate=0.05
    mutate_var = 1
    temp_child=child
    for i in range(len(temp_child)):
        for j in range(len(temp_child)):
            if(rd.random()<_mutate_rate):
                temp_child[i,j,0] += mutate_var*rd.random()*((-1)**(rd.randint(0,1)))
    return temp_child



