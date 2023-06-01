import warnings
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import tree, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix, recall_score, accuracy_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier 

warnings.filterwarnings("ignore")


df = pd.read_csv("DATA.csv")

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

#print(df.head())
print(df.describe().T)
#print(df.info())
#MonkeyPox un dağılımı
print(df['MonkeyPox'].value_counts(normalize=True))

plt.rcParams["font.sans-serif"] = ['SimHei']
plt.rcParams["axes.unicode_minus"] = False
palette = sns.color_palette('pastel')
fig, ax = plt.subplots(figsize=(8, 4))
sns.countplot(x='MonkeyPox', data=df, palette=palette, ax=ax)
ax.set_title("Monkeypox'un Dağılımı", fontsize=15)
plt.show()

#Çeşitli hastalıklar ile maymun çiçeği enfeksiyonu
fig, ax = plt.subplots(2, 4, figsize=(20, 8))
ax = ax.flatten()
for idx, feature in enumerate(df.columns.drop(['MonkeyPox', 'Systemic Illness', "Patient_ID"])):
    print(idx)
    print(feature)
    sns.countplot(x=feature, hue='MonkeyPox', data=df , ax=ax[idx])
    ax[idx].set_title(feature, fontsize=20)
    ax[idx].set(ylabel=None, xlabel=None)
    ax[idx].tick_params(axis='both', labelsize=12)

plt.show()

