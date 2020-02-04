import pandas as pd
import numpy as nump
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_absolute_error,precision_score,recall_score,f1_score,roc_curve
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import random
from sklearn.metrics import roc_curve, roc_auc_score
import math

file=open("de_knn_auc.txt","a+")
path="C:\\Users\\Aman\\Documents\\Defect Prediction\\Datasets"
filelist=os.listdir(path)

file.write("N_Neighbours, Algo, Weight")
file.write("\n")

K=[i for i in range(1,11)] #k=1 to 10
algos=["brute","kd_tree","ball_tree"]
weight=["uniform","distance"]
algoParameters = [{'low': 1, 'high': 10}]

global global_best_score
global_best_score = []
global global_best
global_best= []

global global_worst_score
global_worst_score = []
global global_worst
global_worst= []


def initialisePopulation(np, noOfParameters):
    population = []

    for i in range(0, np):
        candidate = {}
        tunings = []

        n_neighbours = random.choice(K)
        tunings.append(n_neighbours)

        algorithm = random.choice(algos)
        tunings.append(algorithm)

        weights = random.choice(weight)
        tunings.append(weights)

        candidate['tunings'] = tunings
        candidate['score'] = 0

        population.append(candidate)

    '''
    print ("Population")
    print (population)
    print ("\n\n\n")
    '''

    return population

def knn(a, b, c, d, candidate):
  model = KNeighborsClassifier(n_neighbors=candidate['tunings'][0], weights=candidate['tunings'][2], algorithm=candidate['tunings'][1])
  model.fit(a,b)
  rocaucsc = roc_auc_score(d, model.predict(c))
  r=round(rocaucsc,2)
  return r


def score(candidate, datasets):
    df = pd.read_csv(datasets)

    y = df["bug"]
    l = y.tolist()
    for i in range(len(l)):
        if l[i] >= 1:
            l[i] = 1
    y = pd.DataFrame(l)
    y = y.values.ravel()

    X = df.iloc[:, 3:23]

    train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.80, test_size=0.20, random_state=0)

    r = knn(train_X, train_y, test_X, test_y, candidate)
    r = round(r, 2)
    return r


def threeOthers(pop, old, index, np):
    three = list(range(0, np, 1))
    three.remove(index)
    three = random.sample(three, 3)  # Array Formation
    # print (three)

    return pop[three[0]]['tunings'], pop[three[1]]['tunings'], pop[three[2]]['tunings']


def extrapolate(np, old, pop, cr, f, noOfParameters, index, dataset):
    a, b, c = threeOthers(pop, old, index, np)  # index is for the target row
    newf = []

    for i in range(0, noOfParameters):

        x = nump.random.uniform(0, 1)

        if cr < x:
            newf.append(old['tunings'][i])

        elif type(old['tunings'][i]) == bool:
            newf.append(not old['tunings'][i])

        elif old['tunings'][i] in algos:
            newf.append(random.choice(algos))

        elif old['tunings'][i] in weight:
            newf.append(random.choice(weight))

        else:
            lo = algoParameters[i]["low"]
            hi = algoParameters[i]["high"]

            value = a[i] + (f * (b[i] - c[i]))

            mutant_value = int(max(lo, min(value, hi)))
            newf.append(mutant_value)

    # print ("NewF = ",newf)

    dict_mutant = {'tunings': newf}

    score_mutant = score(dict_mutant, dataset)
    score_original = score(old, dataset)

    global global_best
    global global_best_score

    global global_worst
    global global_worst_score

    if score_mutant > score_original:
        global_best_score.append(score_mutant * 100)
        global_best.append({'score': score_mutant * 100, 'tunings': newf})

    else:
        global_best_score.append(score_original)
        global_best.append({'score': score_original * 100, 'tunings': old["tunings"]})

    if score_mutant < score_original:
        global_worst_score.append(score_mutant * 100)
        global_worst.append({'score': score_mutant * 100, 'tunings': newf})

    else:
        global_worst_score.append(score_original)
        global_worst.append({'score': score_original * 100, 'tunings': old["tunings"]})

    newCandidate = {'score': 0, 'tunings': newf}
    # print (newCandidate)
    return newCandidate


def getBestSolution(population):
    max = -1
    bestSolution = {}

    for i in range(0,len(population)):
        scores = population[i]['score']
        #print ("scores ",scores)

        if scores > max:
            max = scores
            bestSolution = population[i]

    return bestSolution


def getWorstSolution(population):
    min = 200
    worstSolution = {}

    for i in range(0,len(population)):
        scores = population[i]['score']
        #print ("scores ",scores)

        if scores < min:
            min = scores
            worstSolution = population[i]

    return worstSolution


def DE(np, f, cr, life, noOfParameters, dataset):
    population = initialisePopulation(np, noOfParameters)  # Intial population formation

    while life > 0:

        global global_best
        global_best = []
        global global_best_score
        global_best_score = []

        global global_worst_score
        global_worst_score = []
        global global_worst
        global_worst = []

        for i in range(0, np):
            extrapolate(np, population[i], population, cr, f, noOfParameters, i, dataset)

        # print ("Global Best :")
        # print (global_best)

        oldPopulation = []
        globalPopulation = []

        for row in population:
            oldPopulation.append(row['tunings'])

        for row in global_best:
            globalPopulation.append(row['tunings'])

        '''
        print("Old Population :")
        print(oldPopulation)
        print("\n")
        print("Global Population :")
        print(globalPopulation)
        '''

        if oldPopulation != globalPopulation:
            population = global_best
            # print (population)
        else:
            life -= 1

        s_Best = getBestSolution(global_best)
        s_Worst = getWorstSolution(global_worst)

    '''
    print ("\n")
    print (global_best)
    print (global_worst)
    print ("\n")
    '''

    scorelist = []
    for i in global_best:
        scorelist.append(i['score'])
    for i in global_worst:
        scorelist.append(i['score'])

    s_default = sum(scorelist) / len(scorelist)

    return s_Best, s_Worst, s_default


for f in tqdm(filelist):
    dataset = path + "\\" + str(f)

    a,b,c=DE(10,0.75, 0.4, 3, 3, dataset)

    file.write("\n")
    file.write(str(f))
    file.write("\n")

    file.write ("\nBest = ")
    for item in a['tunings']:
        file.write("%s, " % item)
    file.write("\n")
    file.write("Score = ")
    file.write(str(a['score']))
    file.write("\n")

    file.write ("\nWorst = ")
    for item in b['tunings']:
        file.write("%s ," % item)
    file.write("\n")
    file.write("Score = ")
    file.write(str(b['score']))
    file.write("\n")

    file.write("\nDefault Score = ")
    file.write(str(c))

    file.write("\n")
    file.write("\n")