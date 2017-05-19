import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

from matplotlib import pyplot as plt

#init PRNG for reproducibility
np.random.seed(123456)

#Load gene expression data
gene_data=pd.read_csv("breast_subtyping.txt", sep='\t')

#Load clinical data
clin_data = pd.read_csv("patient_classes.txt", sep='\t', comment='#')

#Since the dataset is heavily unbalanced, we restrict our example 
#to the two most common classes
to_remove = clin_data[(clin_data.x != "LumA") & (clin_data.x != "LumB")].index


#Create feature matrix where each row corresponds to a subject
#and each column corresponds to a gene
X = gene_data.T.drop(to_remove).sort_index()
X = X.dropna(axis=1)
print "Input Dataset size is {}".format(X.shape)
#As labels we want learn to predict, we use the status of relapse of each subject
le = LabelEncoder()
y = le.fit_transform(clin_data.drop(to_remove).x.sort_index())
print "Output label size is {}".format(y.shape)


#Next, we split the data into training and test set keeping the balance of labels
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, stratify=y)


#Most of the default values for the parameters of a Random Forest
#classifier work reasonably well.
#We need to estimate the number of trees to grow.
#As a rule of thumb, the more trees we grow,
#the better generalization performances we get until a point where 
#the performances stop to improve. Using more trees than necessary
#results in a waste of resources.

#Other parameters should be estimated also, however, here for the sake
#of exposition, we will keep them fixed to reasonable values that 
#are known to work well in practice.

#We will grow an increasing number of unconstrained trees, like in the 
#previuos use case, expecting that partially overfit trees will 
#produce more independent predictions. Also the size of the random 
#subset of features to be selected is fixed to sqrt(p), where p is 
#the total number of features, as suggested in breiman2001.

#We use odd numbers of trees because we want to avoid the case
#of a tie since we are working on the prediction of a binary outcome
n_trees = [21, 41, 61, 81, 101, 201, 301, 401, 501, 601, 701, 
		   801, 901, 1001, 1201, 1401, 1601, 1801, 2001, 2401, 
		   2801, 3401, 4001, 5001, 6001, 8001, 10001]
oob_scores = []
for t in n_trees:
    print "Fitting a Random Forest of {} trees".format(t)
    rf = RandomForestClassifier(
        n_estimators=t,
        criterion="entropy", 
        min_samples_leaf=1, 
        min_samples_split=2, 
        max_features="sqrt", 
        min_impurity_split=0,
        oob_score=True,
        n_jobs=1, 
    )
    
    oob_score = rf.fit(X_train, y_train).oob_score_
    print "\tOOB accuracy {:.02f}%".format(oob_score*100)
    oob_scores.append(oob_score)


plt.plot(n_trees, oob_scores, "b")
plt.ylim(.5, 1)
plt.legend(["OOB Accuracy Estimate"])
plt.savefig("oob.png", dpi=300)

#To choose the optimal number of estimators,
#we inspect the OOB plot to look for the point
#where the OOB estimate stabilizes

#Here the estimates begin to stabilize after
#6001 trees are trained, therefore we fix the
#number of trees to 6001 and compare the OOB
#performances with the test set accuracy predictions

best_rf = RandomForestClassifier(
    n_estimators=6001,
    criterion="entropy", 
    min_samples_leaf=1, 
    min_samples_split=2, 
    max_features="sqrt", 
    min_impurity_split=0,
    oob_score=True,
    n_jobs=1, 
)
best_rf.fit(X_train, y_train)
print "Test set accuracy of the best model {:.02f}%".format(best_rf.score(X_test, y_test)*100)

#Finally, we rank the most relevant features with respect
#to improvements to the impurity criterion across the trees
#of the ensemble
top = 10
importances = best_rf.feature_importances_
idx = importances.argsort()[::-1]
pos = np.arange(top)[::-1] + .5
plt.barh(pos, importances[idx[:top]], align="center")
plt.yticks(pos, X.columns[idx[:top]])
plt.savefig("relevant.png", dpi=300)

