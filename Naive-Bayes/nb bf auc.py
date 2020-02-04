import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import mean_absolute_error,precision_score,recall_score,f1_score,roc_curve
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from sklearn.metrics import roc_curve, roc_auc_score
import math

import time
start_time = time.time()

file=open("bf_auc2.txt","a+")
path="C:\\Users\\Aman\\Documents\\Defect Prediction\\Datasets"
filelist=os.listdir(path)

alpha = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
binarize = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
fit_prior = [True, False]

file.write("Alpha, Binarize, Fit Prior")
file.write("\n")
file.write("Accuracy train, Accuracy test, Precision, Recall, F1")
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

    for a in alpha:
        for b in binarize:
            for fp in fit_prior:

                model = BernoulliNB(alpha=a, binarize=b, fit_prior=fp)
                model.fit(train_X, train_y)
                preds = model.predict(test_X)

                acctrain = round(model.score(train_X, train_y) * 100, 2)
                acctest = round(model.score(test_X, test_y) * 100, 2)

                precisionscore = precision_score(test_y, preds, average='weighted')
                recallscore = recall_score(test_y, preds, average='weighted')
                f1score = f1_score(test_y, preds, average='weighted')
                rocaucsc = roc_auc_score(test_y, model.predict(test_X))

                res = [acctrain, acctest, round(precisionscore, 2), round(recallscore, 2), round(f1score, 2), round(rocaucsc,2)]
                statement = "Alpha = " + str(a) + " Binarize = " + str(b) + " Fit Prior = " + str(fp)

                config = [a, b, fp]

                print(count)
                print(statement)
                print(res)
                count += 1
                print("\n")

                if rocaucsc > maxauc:
                    maxauc = rocaucsc
                    bestconfig = config
                    best = res

                if rocaucsc < minauc:
                    minauc = rocaucsc
                    worstconfig = config
                    worst = res

                if a == 1 and b == 0 and fp == True:
                    defaultconfig = config
                    defaultres = res

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