import os
import time
import warnings
warnings.filterwarnings('ignore')

import random as rdm
import numpy as np
import matplotlib.pyplot as plt #像matlab一样的绘图
import pandas as pd #pandas，数据分析

import mglearn
#sci-kit learn
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






#Read the voice dataset #comma-seperated values  #这就是数据集
mydata = pd.read_csv("D:\\Python Files\\voice.csv") 

chanum = 20  #可以只取一部分特征







#猜测：每一段都视作稳定, 都有基波+谐波

# meanfreq, mean freq
# sd, standard deviation of freq
# median, median freq
# Q25, of freq
# Q75, of freq
# IQR, interquartile(Q75-Q25) of freq
# skew, 偏度(左偏，右偏), u^3/sigma^3
# kurt, 峰度, u^4/sigma^4
# sp.ent, 谱熵
# sfm, 谱平坦度
# mode, (众数) of freq
# centroid, “质心”频率（参考频谱）
# meanfun, fundamental freq(基频)
# minfun, 
# maxfun, 
# meandom, dominant freq(主频率?)
# mindom, 
# maxdom, 
# dfrange, df range
# modindx, 调制指数modulation index, 没搞懂先忽略掉（影响太小）

# 分一些“几乎无影响”~“显著影响”, 1,2,3,4

# 1: minfun, maxfun, median, Q75, skew, kurt, meandom, mindom, maxdom, dfrange, modindx,
# 2: meanfreq, mode,
# 3: sd, sp.ent, sfm, centroid,
# 4: meanfun, IQR, Q25,


#Preview voice dataset
print(mydata.head())  #读前五行


#Prepare data for modeling
male = mydata.loc[mydata['label']=='male']
female = mydata.loc[mydata['label']=='female']
mydata.loc[:,'label'][mydata['label']=="male"] = 0
mydata.loc[:,'label'][mydata['label']=="female"] = 1



mydata_train, mydata_test = train_test_split(mydata, random_state=0, test_size=.2)  #拆分训练集和验证集,20%验证集
#random state保证每次输入一样的种子，获得一样的训练集/验证集，保证输出稳定
scaler = StandardScaler()  #归一化的scaler,u=0,d^2=1
scaler.fit(mydata_train.iloc[:,0:chanum])  #把这个dataframe归一化
X_train = scaler.transform(mydata_train.iloc[:,0:chanum])
X_test = scaler.transform(mydata_test.iloc[:,0:chanum])
Y_train = list(mydata_train['label'].values)
Y_test = list(mydata_test['label'].values)


# ##################################
#
# mydata：pandas.core.frame.DataFrame
# male,female,train,test 也都是 pandas.core.frame.DataFrame
# x最里面是20个特征的归一化值(2534*20),(634*20)，y每一个0是男1是女(2534*1),(634*1)
# x是2darray, y是list
# 下面是一大堆模型
#
# ##################################
    

#Train decision tree model 1
tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, Y_train)
print("Decision Tree")
print("    Accuracy on training set: {:.3f}".format(tree.score(X_train, Y_train)))
print("    Accuracy on test set: {:.3f}".format(tree.score(X_test, Y_test)))

#Train random forest model 2
forest = RandomForestClassifier(n_estimators = 100, random_state = 0)
forest.fit(X_train, Y_train)
print("Random Forests")
print("    Accuracy on training set: {:.3f}".format(forest.score(X_train, Y_train)))
print("    Accuracy on test set: {:.3f}".format(forest.score(X_test, Y_test)))

#Train gradient boosting model 3
gbrt = GradientBoostingClassifier(random_state=0)
gbrt.fit(X_train, Y_train)
print("Gradient Boosting")
print("    Accuracy on training set: {:.3f}".format(gbrt.score(X_train, Y_train)))
print("    Accuracy on test set: {:.3f}".format(gbrt.score(X_test, Y_test)))

#Train support vector machine model 4
svm = SVC()
svm.fit(X_train, Y_train)
print("Support Vector Machine")
print("    Accuracy on training set: {:.3f}".format(svm.score(X_train, Y_train)))
print("    Accuracy on test set: {:.3f}".format(svm.score(X_test, Y_test)))

#Train neural network model 5
mlp = MLPClassifier(random_state=0)
mlp.fit(X_train, Y_train)
print("MLP")
print("    Accuracy on training set: {:.3f}".format(mlp.score(X_train, Y_train)))
print("    Accuracy on test set: {:.3f}".format(mlp.score(X_test, Y_test)))

#Train bagging model 6
bag = BaggingClassifier(random_state=0)
bag.fit(X_train, Y_train)
print("Bagging")
print("    Accuracy on training set: {:.3f}".format(bag.score(X_train, Y_train)))
print("    Accuracy on test set: {:.3f}".format(bag.score(X_test, Y_test)))

#Train stacking model 7
stack = StackingClassifier(estimators=[('dt_clf',DecisionTreeClassifier(random_state=0)), ('svm_clf',SVC()), ('gb_clf',GradientBoostingClassifier())])
stack.fit(X_train, Y_train)
print("Stacking")
print("    Accuracy on training set: {:.3f}".format(stack.score(X_train, Y_train)))
print("    Accuracy on test set: {:.3f}".format(stack.score(X_test, Y_test)))

#Train adaptive boosting model 8
ada = AdaBoostClassifier(base_estimator = GradientBoostingClassifier(random_state=0), random_state=0)
ada.fit(X_train, Y_train)
print("Adaptive Boosting")
print("    Accuracy on training set: {:.3f}".format(ada.score(X_train, Y_train)))
print("    Accuracy on test set: {:.3f}".format(ada.score(X_test, Y_test)))

#Train voting model 9
vote = VotingClassifier(estimators=[('rf_clf',RandomForestClassifier(random_state=0)), \
        ('svm_clf',SVC()), \
        ('bg_clf',BaggingClassifier(random_state = 0)), \
        ('mlp_clf',MLPClassifier(random_state = 0)), \
        ('ada_clf',AdaBoostClassifier(base_estimator = GradientBoostingClassifier(random_state=0), random_state=0))], voting='hard')
vote.fit(X_train, Y_train)
print("Voting")
print("    Accuracy on training set: {:.3f}".format(vote.score(X_train, Y_train)))
print("    Accuracy on test set: {:.3f}".format(vote.score(X_test, Y_test)))



##################################
#                                #
# ↑↑↑↑↑↑↑↑↑↑↑↑ 处理 ↑↑↑↑↑↑↑↑↑↑↑↑ #
#                                #
# ↓↓↓↓↓↓↓↓↓↓↓↓ 画图 ↓↓↓↓↓↓↓↓↓↓↓↓ #
#                                #                                  
##################################



def plot_feature_importances_mydata(model):
    plt.figure(figsize=(100, 20))
    n_features = X_train.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), list(mydata)[0:-1])
    plt.xlabel("Variable importance")
    plt.ylabel("Independent Variable")
    
# Plot the variable importance of each classifier # Plot the heatmap on first layer weights for neural network
# Plot the histograms   #画20个特征的分布图
fig, axes = plt.subplots(10, 2, figsize=(50,100))  #1
ax = axes.ravel()  #把axes拆成1darray, ax是2darray

for i in range(chanum):
    ax[i].hist(male.iloc[:,i], bins=30, color=mglearn.cm3(0), alpha=.5)
    ax[i].hist(female.iloc[:, i], bins=30, color=mglearn.cm3(2), alpha=.5)
    ax[i].set_title(list(male)[i])
    ax[i].set_yticks(())

# ax[0].set_xlabel("Feature magnitude")
# ax[0].set_ylabel("Frequency")
# ax[0].legend(["male", "female"], loc="best")

# fig.tight_layout() #1

# plot_feature_importances_mydata(tree) #2
# plot_feature_importances_mydata(forest) #3
# plot_feature_importances_mydata(gbrt) #4
plot_feature_importances_mydata(ada) #5

# plot_feature_importances_mydata(bag) #6
# plot_feature_importances_mydata(vote) #7
# plot_feature_importances_mydata(stack) #8


# plt.figure(figsize=(100, 60))
# plt.imshow(mlp.coefs_[0], interpolation='none', cmap='viridis')
# plt.yticks(range(chanum), list(mydata)[0:-1],fontsize = 10)
# plt.xlabel("Columns in weight matriloc", fontsize = 50)
# plt.ylabel("Input feature", fontsize = 50)
# plt.colorbar().set_label('Importance',size=50)
    
plt.show()

exit()