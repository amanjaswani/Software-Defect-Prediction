# Software-Defect-Prediction
Hyperparameter Tuning using Brute Force and Differential Evolution



## Background
Defect prediction is a practice that focuses on identifying where defects will appear in a software system and which modules require extensive testing. Many studies conducted on software defect prediction in the past decade have applied Machine Learning (ML) based approaches, having performed comparisons of different ML algorithms on a collection of datasets. These ML algorithms usually have certain ‘adaptable parameters’, called ‘hyperparameters’, whose tuning may provide better results for the prediction problem at hand than the results obtained by using the algorithm with default parameter values. A common assumption made by many studies in literature is that these values won’t significantly impact the outcome, and hence they have set these hyperparameters to their default values in their experiments.



## Aims
To investigate whether optimizing hyperparameters of a model affects the performance of the ML algorithm, and also find the better approach to parameter tuning out of the following – 
i.	Brute Force (BF)
ii.	Differential Evolution (DE)



## Method
For each of the 12 datasets (obtained from the tera-PROMISE repository), five ML algorithms were implemented in which the hyperparameters were first manipulated by brute force, and later by differential evolution. The Worst Case Performance results (the combination of hyperparameters that give lowest AUC – Area Under the Curve score) for both these approaches were then compared, grouped on the basis of the ML algorithms used.



## Results
Results show that the performance of a ML algorithm is significantly impacted after tuning. DE almost always outperforms BF approach results in terms of the metric chosen for evaluation (AUC in this case). Thus, Parameter Tuning via DE approach does indeed pay off in the end.



## Conclusions
The performance of a ML algorithm can be significantly improved by employing Parameter Tuning and the DE approach for optimizing parameters is faster and also gives better results than the BF approach.



## Novelty/Contribution 
The main contribution of this study is to explore different tuning approaches to find out the better approach by analyzing the worst case performance of the approaches used.

