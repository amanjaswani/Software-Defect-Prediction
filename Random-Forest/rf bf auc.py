import random
import numpy as nump
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, precision_score, recall_score, f1_score, roc_curve
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, roc_auc_score
import math

import time
start_time=time.time()

file = open("rf_bf_auc2.txt", "a+")
path = "C:\\Users\\Aman\\Documents\\Defect Prediction\\Datasets"
filelist = os.listdir(path)

n_estimators = [2, 4, 8, 10, 16, 32, 64, 100, 200]
max_depths = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None]
min_samples_splits = [2, 5, 10]
min_samples_leafs = [1, 2, 4]
criterion = ['gini', 'entropy']

file.write("NEstimator, Depth, Samples Split, Sample Leaves, Criterion")
file.write("\n")
file.write("Accuracy train, Accuracy test, Precision, Recall, F1, AUC")
file.write("\n")

for f in tqdm(filelist):
    dataset = path + "\\" + str(f)
    df = pd.read_csv(dataset)

    y = df["bug"]
    l = y.tolist()
    for i in range(len(l)):
        if l[i] >= 1:
            l[i] = 1
    y = pd.DataFrame(l)
    y = y.values.ravel()

    X = df.iloc[:, 3:23]

    train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.80, test_size=0.20, random_state=0)

    count = 1
    minauc = 2
    maxauc = -1
    best = []
    worst = []
    defaultres = []
    bestconfig = ""
    worstconfig = ""
    defaultconfig = ""

    '''
    ALLSTATEMENTS = []
    ALLRES = []
    D = {}
    '''

    for estimator in n_estimators:
        for depth in max_depths:
            for samplesplit in min_samples_splits:
                for sampleleaf in min_samples_leafs:
                    for c in criterion:

                        model = RandomForestClassifier(min_samples_split=samplesplit, criterion=c,
                                                       n_estimators=estimator,
                                                       max_depth=depth, min_samples_leaf=sampleleaf, random_state=0)
                        model.fit(train_X, train_y)
                        preds = model.predict(test_X)

                        acctrain = round(model.score(train_X, train_y) * 100, 2)
                        acctest = round(model.score(test_X, test_y) * 100, 2)

                        precisionscore = precision_score(test_y, preds, average='weighted')
                        recallscore = recall_score(test_y, preds, average='weighted')
                        f1score = f1_score(test_y, preds, average='weighted')
                        rocaucsc = roc_auc_score(test_y, model.predict(test_X))

                        res = [acctrain, acctest, round(precisionscore, 2), round(recallscore, 2), round(f1score, 2), round(rocaucsc,2)]
                        statement = "NEstimator = " + str(estimator) + " Depth = " + str(
                            depth) + " Samples Split = " + str(
                            samplesplit) + " Sample Leaves = " + str(sampleleaf) + " Criterion = " + str(c)

                        config = [estimator, depth, samplesplit, sampleleaf, c]

                        '''
                        print(count)
                        print(statement)
                        print(res)
                        count += 1
                        print("\n")
                        '''

                        if rocaucsc > maxauc:
                            maxauc = rocaucsc
                            bestconfig = config
                            best = res

                        if rocaucsc < minauc:
                            minauc = rocaucsc
                            worstconfig = config
                            worst = res

                        if estimator == 10 and depth == None and samplesplit == 2 and sampleleaf == 1 and c == 'gini':
                            defaultconfig = config
                            defaultres = res

                        '''
                        ALLSTATEMENTS.append(statement)
                        ALLRES.append(res)

                        D[statement] = res
                        '''
    file.write("\n")
    file.write(str(f))
    file.write("\n")

    file.write("\nBest = ")
    for item in bestconfig:
        file.write("%s, " % item)
    file.write("\n")
    for item in best:
        file.write("%s ," % item)
    file.write("\n")

    file.write("\nWorst = ")
    for item in worstconfig:
        file.write("%s ," % item)
    file.write("\n")
    for item in worst:
        file.write("%s ," % item)
    file.write("\n")

    file.write("\nDefault = ")
    for item in defaultconfig:
        file.write("%s ," % item)
    file.write("\n")
    for item in defaultres:
        file.write("%s ," % item)
    file.write("\n")

    file.write("\n")
    file.write("\n")

print (time.time()-start_time, "seconds")
