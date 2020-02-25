import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

from sklearn import svm
from sklearn.svm import SVC
from sklearn.feature_selection import RFE



from sklearn.metrics import confusion_matrix, f1_score

from sklearn.model_selection import train_test_split

print(LogisticRegression())


dfOriginal = pd.read_csv('parkinsons.csv') # 195 rows x 24 columns
df = dfOriginal.drop(['name'], axis=1)

#looking for empty columns
print(df[df==0].count(axis=0))
print("\n")

# Compute the correlation matrix
corr = df.corr()
#sns.heatmap(corr)
# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=np.bool))
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


#print(df.corr())

df_data = np.asarray(df.drop('status',1))
df_target = np.asarray(df['status'])


print("mean of status: {}".format(df_target.mean()))

#print(df_target)


def print_num_of_positive_and_negative(data):
    print("sum: {}".format(len(data)))
    positive = 0
    for i in data:
        if i == 1 :
            positive+=1
    print("positive : {}\nnegative : {}".format(positive,len(data)-positive))

x_train, x_test, y_train, y_test = train_test_split(df_data, df_target, test_size=0.25,random_state=0)#, shuffle = False, stratify=None) #

print("in y train :")
print_num_of_positive_and_negative(y_train)
print("in y test :")
print_num_of_positive_and_negative(y_test)



# normalize x_train and x_test matrix
means = np.mean(x_train, axis=0)
stds = np.std(x_train, axis=0)
x_train = (x_train - means) / stds
x_test = (x_test - means) / stds




def plot_confusion_matrix(logisticReg_predict_x, y,score):
    plt.figure()
    sns.heatmap(confusion_matrix(y, logisticReg_predict_x), annot=True, linewidths=0.5, cmap="Reds_r", square=True)
    plt.ylabel("Real value")
    plt.xlabel("Predicted Value")
    plt.title("confusion matrix\nscore : {}".format(score))
    plt.show()


################  LOGISTIC REGRESSION ################################

logisticRegr = LogisticRegression(C=1.0)
logisticRegr.fit(x_train,y_train)
y_pred = logisticRegr.predict(x_test)
score = f1_score(y_test, y_pred, sample_weight = None)

plot_confusion_matrix(y_pred,y_test,score)


logisticRegr = LogisticRegression(C=1.0, class_weight={0: 0.3})
logisticRegr.fit(x_train,y_train)
y_pred = logisticRegr.predict(x_test)
score = f1_score(y_test, y_pred, sample_weight = None)

plot_confusion_matrix(y_pred,y_test,score)

logisticRegr = LogisticRegression(C=1.0, class_weight={0: 0.3},tol=0.01)
logisticRegr.fit(x_train,y_train)
y_pred = logisticRegr.predict(x_test)
score = f1_score(y_test, y_pred, sample_weight = None)

plot_confusion_matrix(y_pred,y_test,score)
################  SVM ################################


clf = svm.SVC(C=1.0)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
score = f1_score(y_test, y_pred, sample_weight = None)

plot_confusion_matrix(y_pred,y_test,score)


clf = svm.SVC(C=1.0, class_weight={1: 0.9})
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
score = f1_score(y_test, y_pred, sample_weight = None)

plot_confusion_matrix(y_pred,y_test,score)



features = [name for name in df.drop('status',1).columns] # names of all features
# Create the RFE object and rank each pixel
svc = SVC(kernel="linear", C=1)
rfe = RFE(estimator=svc, n_features_to_select=5, step=1)
rfe.fit(x_train, y_train)

print(features)
print(len(features))
ranking = rfe.ranking_#.reshape(features)
print(len(ranking))
print(ranking)
new_features=[]
for i,j in zip(features,ranking):
    if j ==1 :
        new_features.append(i)
    print(i,j)

print(new_features)



##fdsafdsa



