import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_absolute_error, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, roc_auc_score
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

file = open("brute_knn_new.txt", "a+")
path = "C:\\Users\\Aman\\Documents\\Defect Prediction\\Datasets"
filelist = os.listdir(path)

algos = ["brute", "kd_tree", "ball_tree"]
K = [i for i in range(1, 11)]  # k=1 to 10
weights = ["uniform", "distance"]

file.write("K value, algorithm, weights")
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

    best_config_auc = []
    best_res_auc = []
    worst_config_auc = []
    worst_res_auc = []

    default_config = []
    default_res = []

    min_auc = 2
    max_auc = -1

    count = 1
    for k in K:
        for a in algos:
            for w in weights:

                model = KNeighborsClassifier(n_neighbors=k, weights=w, algorithm=a)
                model.fit(train_X, train_y)
                preds = model.predict(test_X)
                acctrain = round(model.score(train_X, train_y) * 100, 2)
                acctest = round(model.score(test_X, test_y) * 100, 2)
                precisionscore = precision_score(test_y, preds, average='weighted')
                recallscore = recall_score(test_y, preds, average='weighted')
                f1score = f1_score(test_y, preds, average='weighted')

                gmean = math.sqrt(precisionscore * recallscore)
                #false_positive_rate, true_positive_rate, thresholds = roc_curve(test_y, model.predict(test_X))
                rocaucsc = roc_auc_score(test_y, model.predict(test_X))

                res = [acctrain, acctest, round(precisionscore, 2), round(recallscore, 2), round(f1score, 2),
                       round(rocaucsc, 2), round(gmean, 2)]
                config = [k, a, w]

                if k == 5 and a == "brute" and w == "uniform":
                    default_config = config
                    default_res = res

                if rocaucsc > max_auc:
                    max_auc = rocaucsc
                    best_config_auc = config
                    best_res_auc = res

                if rocaucsc < min_auc:
                    min_auc = rocaucsc
                    worst_config_auc = config
                    worst_res_auc = res

                print(count)
                count += 1
                print(best_config_auc)
                print(best_res_auc)
                print(worst_config_auc)
                print(worst_res_auc)
                print("\n")

    file.write("\n")
    file.write(str(f))
    file.write("\n")

    file.write("\nBest = ")
    for item in best_config_auc:
        file.write("%s, " % item)
    file.write("\n")
    for item in best_res_auc:
        file.write("%s ," % item)
    file.write("\n")

    file.write("\nWorst = ")
    for item in worst_config_auc:
        file.write("%s ," % item)
    file.write("\n")
    for item in worst_res_auc:
        file.write("%s ," % item)
    file.write("\n")

    file.write("\nDefault = ")
    for item in default_config:
        file.write("%s ," % item)
    file.write("\n")
    for item in default_res:
        file.write("%s ," % item)
    file.write("\n")

    file.write("\n")
    file.write("\n")


