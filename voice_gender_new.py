import os
import time
import warnings
warnings.filterwarnings('ignore')

import random as rdm
import numpy as np
import matplotlib.pyplot as plt #像matlab一样的绘图
import pandas as pd #pandas，数据分析

import mglearn
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import KFold #k折验证
from sklearn.preprocessing import StandardScaler

#简单决策树
from sklearn.tree import DecisionTreeClassifier 

#集成方法
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import GradientBoostingClassifier 

#支持向量机，严格数学方法
from sklearn.svm import SVC 

#多层神经网络
from sklearn.neural_network import MLPClassifier #（更多：CNN卷积神经网络，RNN递归神经网络等）


#自己添加的方法
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier


def plot_feature_importances_mydata(model):
    plt.figure(figsize=(100, 20))
    n_features = X_train[0].shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), list(mydata)[0:-1])
    plt.xlabel("Variable importance")
    plt.ylabel("Independent Variable")


mydata = pd.read_csv("D:\\Python Files\\voice9_rand.csv") 

chanum = 9  #可以只取一部分特征

print(mydata.head(5))  #读前五行

#预处理
mydata.loc[:,'label'][mydata['label']=="male"] = 0
mydata.loc[:,'label'][mydata['label']=="female"] = 1
X_train = []
X_test = []
Y_train = []
Y_test = []

#归一化
scaler = StandardScaler()  
scaler.fit(mydata.iloc[:,0:chanum]) 

#K折交叉验证
#这里好像没有随机划分, 即“把男女打乱”, 故男女打乱为手动操作
k_splits = 5
kf = KFold(n_splits = k_splits)

for train, test in kf.split(mydata):
    trainData = mydata.iloc[train,:]
    testData = mydata.iloc[test,:]

    X_train.append(scaler.transform(trainData.iloc[:,0:chanum]))  #取所有行，前20列，删掉了label
    X_test.append(scaler.transform(testData.iloc[:,0:chanum]))  #取所有行，前20列，删掉了label
    Y_train.append(list(trainData['label'].values))
    Y_test.append(list(testData['label'].values))


#tree
temptrainacc = 0
temptestacc = 0
for i in range(k_splits):
    tree = DecisionTreeClassifier(random_state = 0)
    tree.fit(X_train[i], Y_train[i])
    temptrainacc += tree.score(X_train[i], Y_train[i])
    temptestacc += tree.score(X_test[i], Y_test[i])
    #print(temptestacc)
temptrainacc /= k_splits
temptestacc /= k_splits
print("Decision Tree")
print("    Accuracy on training set: {:.3f}".format(temptrainacc))
print("    Accuracy on test set: {:.3f}".format(temptestacc))

#forest
temptrainacc = 0
temptestacc = 0
for i in range(k_splits):
    forest = RandomForestClassifier(n_estimators = 100, random_state = 0)
    forest.fit(X_train[i], Y_train[i])
    temptrainacc += forest.score(X_train[i], Y_train[i])
    temptestacc += forest.score(X_test[i], Y_test[i])
    #print(temptestacc)
temptrainacc /= k_splits
temptestacc /= k_splits
print("Random Forests")
print("    Accuracy on training set: {:.3f}".format(temptrainacc))
print("    Accuracy on test set: {:.3f}".format(temptestacc))

#gbrt
temptrainacc = 0
temptestacc = 0
for i in range(k_splits):
    gbrt = GradientBoostingClassifier(random_state = 0)
    gbrt.fit(X_train[i], Y_train[i])
    temptrainacc += gbrt.score(X_train[i], Y_train[i])
    temptestacc += gbrt.score(X_test[i], Y_test[i])
    #print(temptestacc)
temptrainacc /= k_splits
temptestacc /= k_splits
print("Gradient Boosting")
print("    Accuracy on training set: {:.3f}".format(temptrainacc))
print("    Accuracy on test set: {:.3f}".format(temptestacc))

#svm
temptrainacc = 0
temptestacc = 0
for i in range(k_splits):
    svm = SVC()
    svm.fit(X_train[i], Y_train[i])
    temptrainacc += svm.score(X_train[i], Y_train[i])
    temptestacc += svm.score(X_test[i], Y_test[i])
    #print(temptestacc)
temptrainacc /= k_splits
temptestacc /= k_splits
print("SVM")
print("    Accuracy on training set: {:.3f}".format(temptrainacc))
print("    Accuracy on test set: {:.3f}".format(temptestacc))

#mlp
temptrainacc = 0
temptestacc = 0
for i in range(k_splits):
    mlp = MLPClassifier(random_state = 0)
    mlp.fit(X_train[i], Y_train[i])
    temptrainacc += mlp.score(X_train[i], Y_train[i])
    temptestacc += mlp.score(X_test[i], Y_test[i])
    #print(temptestacc)
temptrainacc /= k_splits
temptestacc /= k_splits
print("MLP")
print("    Accuracy on training set: {:.3f}".format(temptrainacc))
print("    Accuracy on test set: {:.3f}".format(temptestacc))

#bag
temptrainacc = 0
temptestacc = 0
for i in range(k_splits):
    bag = BaggingClassifier(random_state = 0)
    bag.fit(X_train[i], Y_train[i])
    temptrainacc += bag.score(X_train[i], Y_train[i])
    temptestacc += bag.score(X_test[i], Y_test[i])
    #print(temptestacc)
temptrainacc /= k_splits
temptestacc /= k_splits
print("Bagging")
print("    Accuracy on training set: {:.3f}".format(temptrainacc))
print("    Accuracy on test set: {:.3f}".format(temptestacc))

#stack
temptrainacc = 0
temptestacc = 0
for i in range(k_splits):
    stack = StackingClassifier(estimators=[('dt_clf',DecisionTreeClassifier(random_state=0)), ('svm_clf',SVC()), ('gb_clf',GradientBoostingClassifier(random_state = 0))])
    stack.fit(X_train[i], Y_train[i])
    temptrainacc += stack.score(X_train[i], Y_train[i])
    temptestacc += stack.score(X_test[i], Y_test[i])
    #print(temptestacc)
temptrainacc /= k_splits
temptestacc /= k_splits
print("Stacking")
print("    Accuracy on training set: {:.3f}".format(temptrainacc))
print("    Accuracy on test set: {:.3f}".format(temptestacc))

#ada
temptrainacc = 0
temptestacc = 0
for i in range(k_splits):
    ada = AdaBoostClassifier(base_estimator = GradientBoostingClassifier(random_state=0), random_state=0)
    ada.fit(X_train[i], Y_train[i])
    temptrainacc += ada.score(X_train[i], Y_train[i])
    temptestacc += ada.score(X_test[i], Y_test[i])
    #print(temptestacc)
temptrainacc /= k_splits
temptestacc /= k_splits
print("Adaptive Boosting")
print("    Accuracy on training set: {:.3f}".format(temptrainacc))
print("    Accuracy on test set: {:.3f}".format(temptestacc))

#vote
temptrainacc = 0
temptestacc = 0
for i in range(k_splits):
    vote = VotingClassifier(estimators=[('rf_clf',RandomForestClassifier(random_state=0)), \
        ('svm_clf',SVC()), \
        ('bg_clf',BaggingClassifier(random_state = 0)), \
        ('mlp_clf',MLPClassifier(random_state = 0)), \
        ('ada_clf',AdaBoostClassifier(base_estimator = GradientBoostingClassifier(random_state=0), random_state=0))], voting='hard')
    vote.fit(X_train[i], Y_train[i])
    temptrainacc += vote.score(X_train[i], Y_train[i])
    temptestacc += vote.score(X_test[i], Y_test[i])
    #print(temptestacc)
temptrainacc /= k_splits
temptestacc /= k_splits
print("Voting")
print("    Accuracy on training set: {:.3f}".format(temptrainacc))
print("    Accuracy on test set: {:.3f}".format(temptestacc))


# Plot the histograms   #先画20个特征的分布统计图

male = mydata.loc[mydata['label']==0]
female = mydata.loc[mydata['label']==1]
fig, axes = plt.subplots(10, 2, figsize=(50,100))  #1
ax = axes.ravel()  #把axes拆成1darray, ax是2darray

for i in range(chanum):
    ax[i].hist(male.iloc[:,i], bins=30, color=mglearn.cm3(0), alpha=.5)
    ax[i].hist(female.iloc[:, i], bins=30, color=mglearn.cm3(2), alpha=.5)
    ax[i].set_title(list(male)[i])
    ax[i].set_yticks(())

ax[0].set_xlabel("Feature magnitude")
ax[0].set_ylabel("Frequency")
ax[0].legend(["male", "female"], loc="best")

fig.tight_layout() #1


# Plot the variable importance of each classifier, and the heatmap on first layer weights for neural network

plot_feature_importances_mydata(tree) #2
plot_feature_importances_mydata(forest) #3
plot_feature_importances_mydata(gbrt) #4
plot_feature_importances_mydata(ada) #5

#6
plt.figure(figsize=(100, 60))
plt.imshow(mlp.coefs_[0], interpolation='none', cmap='viridis')
plt.yticks(range(chanum), list(mydata)[0:-1],fontsize = 10)
plt.xlabel("Columns in weight matriloc", fontsize = 50)
plt.ylabel("Input feature", fontsize = 50)
plt.colorbar().set_label('Importance',size=50)
    
plt.show()

exit()