import random
import numpy as nump
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import precision_score,recall_score,f1_score
from tqdm import tqdm
import os
from sklearn.metrics import roc_curve, roc_auc_score
import math

import time
start_time = time.time()

file=open("de auc2.txt","a+")
path="C:\\Users\\Aman\\Documents\\Defect Prediction\\Datasets"
filelist=os.listdir(path)

file.write("NEstimator, Learning Rate, Algorithm")
file.write("\n")

alpha = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
binarize = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
fit_prior = [True, False]

algoParameters = [{'low': 0.0001, 'high': 100}, {'low': 0.0, 'high': 1.0}]

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

        a = random.choice(alpha)
        tunings.append(a)

        b = random.choice(binarize)
        tunings.append(b)

        fp = random.choice(fit_prior)
        tunings.append(fp)

        candidate['tunings'] = tunings
        candidate['score'] = 0

        population.append(candidate)

    '''
    print ("Population")
    print (population)
    print ("\n\n\n")
    '''

    return population

def naive_bayes(a, b, c, d, candidate):

    model = BernoulliNB(alpha=candidate['tunings'][0], binarize=candidate['tunings'][1],
                               fit_prior=candidate['tunings'][2])
    model.fit(a, b)
    rocaucsc = roc_auc_score(d, model.predict(c))
    r = round(rocaucsc, 2)
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

    r = naive_bayes(train_X, train_y, test_X, test_y, candidate)
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
        # print "Random number for comparison with cr : " + str(x)

        if cr < x:
            # print "Old tuning Value for index " + str(index) + " : " + (str(old['tunings'][i]))
            newf.append(old['tunings'][i])

        elif type(old['tunings'][i]) == bool:
            newf.append(not old['tunings'][i])


        else:
            lo = algoParameters[i]["low"]
            hi = algoParameters[i]["high"]

            value = a[i] + (f * (b[i] - c[i]))
            # print "Value before trim : " + str(value)

            mutant_value = int(max(lo, min(value, hi)))
            # print "Mutant Value : " + str(mutant_value)

            newf.append(mutant_value)

    # print ("NewF = ",newf)

    dict_mutant = {'tunings': newf}
    # print ("Dict_mutant",dict_mutant)


    score_mutant = score(dict_mutant, dataset)
    score_original = score(old, dataset)

    # print ("Original Score : " + str(score_original))
    # print ("Mutant Score : " + str(score_mutant))

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

print (time.time()-start_time, "seconds")