import warnings
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import pandas as pd
from sklearn import tree, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix, recall_score, accuracy_score, precision_score ,mean_squared_error , r2_score
from sklearn.model_selection import train_test_split ,  cross_val_score, KFold
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.naive_bayes import GaussianNB
warnings.filterwarnings("ignore")

df = pd.read_csv("DATA.csv")
""" sns.heatmap(df.corr(), annot=True)
plt.show() """

def scores(actual, predicted):
    # # karışıklık matrisi
    c_matrix = confusion_matrix(actual, predicted)
    print(c_matrix)
    score = f1_score(actual, predicted)
    recall = recall_score(actual, predicted)
    accuracy = accuracy_score(actual, predicted)
    precision = precision_score(actual, predicted)
    print('Kesinlik：', precision * 100)
    print('Doğruluk:', accuracy * 100)
    print('Recall: ', recall * 100)
    print("F1: ", score * 100) 


#----------------------------------------------------------------------------------------------------------------------------------
# Verilerdeki True, False ve "MonkeyPox" etiketinin "Pozitif", "Negatif" değerlerini 1, 0 olarak değiştirdik
df = df.replace(["Positive", "Negative", True, False], [1, 0, 1, 0])
#print(df['MonkeyPox'].value_counts()); 
# x özellik verilerini depolar, y etiketleri depolar
df_X = df.drop(["MonkeyPox", "Patient_ID"], axis=1)
df_Y = df["MonkeyPox"]
# Sistemik Hastalığı Yok Ateş, Şişmiş Lenf Düğümleri olarak ayırdık
df_X_final = pd.get_dummies(df_X, drop_first=True)
""" plt.figure(figsize=(15,12))
sns.heatmap(df_X_final.corr(), cmap=plt.cm.CMRmap_r, annot = True)
plt.show() """
print(df_X_final.head())


#-------------------------------------------------------------------------------------------------------------------------------------------------
#Lojistik Regresyon
x_train, x_test, y_train, y_test = train_test_split(df_X_final, df_Y, test_size=0.3, random_state=True)
lr_model = LogisticRegression()
lr_model.fit(x_train, y_train)
y_pred = lr_model.predict(x_test)
print('Lojistik Regresyon: ')
scores(y_test, y_pred)

#-------------------------------------------------------------------------------------------------------------------------------------------------

# karar ağacı
dtc = DecisionTreeClassifier(max_depth=100, min_samples_leaf=100)
dtc.fit(x_train, y_train)
y_pred_dtc = dtc.predict(x_test)
print('Karar Ağacı: ')
scores(y_test, y_pred_dtc)

feature_names = list(df_X_final.columns.values)
plt.figure(figsize=(18, 12))
_ = tree.plot_tree(dtc, filled=True, feature_names=feature_names)  # Dönen değer önemli olmadığı için direk alt çizgi ile alınır.

#plt.show()

#---------------------------------------------------------------------------------------------------------------------------------------------------
#Random Forest
rfc = RandomForestClassifier(random_state=1,n_estimators=100, max_depth=8, min_samples_leaf=5, max_features=3,  min_samples_split=100, bootstrap=True)
history = rfc.fit(x_train, y_train, )
y_pred_rfc = rfc.predict(x_test)
print('Rastgele Orman: ')
scores(y_test, y_pred_rfc)

#-----------------------------------------------------------------------------------------------------------------------------------------------------
###XGBOOSTT
print("XGBOOST: ")

xgbr = xgb.XGBClassifier()
xgbr.fit(x_train, y_train)
y_pred_xg = xgbr.predict(x_test)
scores(y_test,y_pred_xg) 



""" score = xgbr.score(x_train, y_train)   
print("Training score: ", score) 
 
# - cross validataion 
scoress = cross_val_score(xgbr, x_train, y_train, cv=5)
print("Mean cross-validation score: %.2f" % scoress.mean())

kfold = KFold(n_splits=10, shuffle=True)
kf_cv_scores = cross_val_score(xgbr, x_train, y_train, cv=kfold )
print("K-fold CV average score: %.2f" % kf_cv_scores.mean()) 
 
ypred = xgbr.predict(x_test)

mse = mean_squared_error(y_test, ypred)
print("MSE: %.2f" % mse)
print("RMSE: %.2f" % (mse**(1/2.0)))  """

""" x_ax = range(len(y_test))
plt.scatter(x_ax, y_test, s=5, color="blue", label="original")
plt.plot(x_ax, y_pred_xg, lw=0.8, color="red", label="predicted")
plt.legend()
plt.show()   """
#---------------------------------------------------------------------------------------------------------------------------------------------

#catboost
print("catboost::")
model_cb = CatBoostClassifier(iterations=100, learning_rate=0.1)

model_cb.fit(x_train, y_train)
y_pred_cat = model_cb.predict(x_test)
scores(y_test,y_pred_cat)

#-----------------------------------------------------------------------------------------------------------------------------
#adaboost
print("Adaboost:: ")
model_ab = AdaBoostClassifier(random_state=1, n_estimators=100)

model_ab.fit(x_train, y_train)
y_pred_ada=model_ab.predict(x_test)
scores(y_test,y_pred_ada)


svc = SVC(kernel = "rbf", C = 100, gamma = 10)
svc.fit(x_train, y_train)
y_pred_svc = svc.predict(x_test)
print("Support Vector Classifier：")
scores(y_test, y_pred_svc)

bayes_modle=GaussianNB()
bayes_modle.fit(x_train,y_train)
y_pred_bayes=bayes_modle.predict(x_test)
print("Gaussian：")
scores(y_test, y_pred_bayes)