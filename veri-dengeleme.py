
import numpy as np
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import tree, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix, recall_score, accuracy_score, precision_score ,mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.utils import resample

def scores(actual, predicted):
    # # karışıklık matrisi
    # c_matrix = confusion_matrix(actual, predicted)
    # print(c_matrix)
    score = f1_score(actual, predicted)
    recall = recall_score(actual, predicted)
    accuracy = accuracy_score(actual, predicted)
    precision = precision_score(actual, predicted)
    print('Kesinlik：', precision * 100)
    print('Doğruluk:', accuracy * 100)
    print('Recall: ', recall * 100)
    print("F1: ", score * 100) 

# veri kümesi oluşturun
data= pd.read_csv("DATA.csv")
data = data.replace(["Positive", "Negative", True, False], [1, 0, 1, 0])
data = data.drop([ "Patient_ID"], axis=1)
# veriyi hedef değişkenine göre ayırın
minority = data[data['MonkeyPox'] == 1]
majority = data[data['MonkeyPox'] == 0]

# küçük veri kümesini büyük veri kümesiyle eşit hale getirin
minority_upsampled = resample(minority,
                               replace=True,
                               n_samples=len(majority),
                               random_state=123)

# verileri tekrar birleştirin
upsampled = pd.concat([majority, minority_upsampled])

# verileri tekrar rastgele olarak ayırın
data = upsampled.sample(frac=1, random_state=123).reset_index(drop=True)

print(data['MonkeyPox'].value_counts(normalize=True))
sns.countplot(x='Systemic Illness', data=data, hue='MonkeyPox',palette='Set1')
plt.title("Sistemik Hastalığın Maymun Çiçeğine Yol Açma Grafiği")

plt.rcParams["font.sans-serif"] = ['SimHei']
plt.rcParams["axes.unicode_minus"] = False
palette = sns.color_palette('pastel')
fig, ax = plt.subplots(figsize=(8, 4))
sns.countplot(x='MonkeyPox', data=data, palette=palette, ax=ax)
ax.set_title("Monkeypox'un Dağılımı", fontsize=15)



#Çeşitli hastalıklar ile maymun çiçeği enfeksiyonu
fig, ax = plt.subplots(2, 4, figsize=(20, 8))
ax = ax.flatten()
for idx, feature in enumerate(data.columns.drop(['MonkeyPox', 'Systemic Illness'])):
    print(idx)
    print(feature)
    sns.countplot(x=feature, hue='MonkeyPox', data=data , ax=ax[idx])
    ax[idx].set_title(feature, fontsize=20)
    ax[idx].set(ylabel=None, xlabel=None)
    ax[idx].tick_params(axis='both', labelsize=12)



plt.show()

# dengelemenin sonucunu görüntüleyin
print('Hedef değişken sıklığı (dengeleme öncesi):')
print(data['MonkeyPox'].value_counts()) 

df = data.replace(["Positive", "Negative", True, False], [1, 0, 1, 0])
# x özellik verilerini depolar, y etiketleri depolar
df_X = df.drop(["MonkeyPox"], axis=1)
df_Y = data["MonkeyPox"]
df_X_final = pd.get_dummies(df_X, drop_first=True)

x_train, x_test, y_train, y_test = train_test_split(df_X_final, df_Y, test_size=0.2, random_state=True)
lr_model = LogisticRegression()
lr_model.fit(x_train, y_train)
y_pred = lr_model.predict(x_test)
print('Lojistik Regresyon: ')
scores(y_test, y_pred)



# karar ağacı
dtc = DecisionTreeClassifier(max_depth=100, min_samples_leaf=100)
dtc.fit(x_train, y_train)
y_pred_dtc = dtc.predict(x_test)
print('Karar Ağacı: ')
scores(y_test, y_pred_dtc)

feature_names = list(df_X_final.columns.values)
plt.figure(figsize=(18, 12))
_ = tree.plot_tree(dtc, filled=True, feature_names=feature_names)  # Dönen değer önemli olmadığı için direk alt çizgi ile alınır.


#Random Forest
rfc = RandomForestClassifier(random_state=1,n_estimators=100, max_depth=8, min_samples_leaf=5, max_features=3,  min_samples_split=100, bootstrap=True)
history = rfc.fit(x_train, y_train, )
y_pred_rfc = rfc.predict(x_test)
print('Rastgele Orman: ')
scores(y_test, y_pred_rfc)


print("XGBOOST: ")

xgbr = xgb.XGBClassifier()
xgbr.fit(x_train, y_train)
y_pred_xg = xgbr.predict(x_test)
scores(y_test,y_pred_xg) 


#catboost
print("catboost::")
model_cb = CatBoostClassifier(iterations=100, learning_rate=0.1)

model_cb.fit(x_train, y_train)
y_pred_cat = model_cb.predict(x_test)
scores(y_test,y_pred_cat)

#adaboost
print("Adaboost:: ")
model_ab = AdaBoostClassifier(random_state=1, n_estimators=100)

model_ab.fit(x_train, y_train)
y_pred_ada=model_ab.predict(x_test)
scores(y_test,y_pred_ada)