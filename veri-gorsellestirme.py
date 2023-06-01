import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy as py
from sklearn.model_selection import train_test_split
from math import log
import seaborn as sns
import time
import operator
import csv
#import treePlotter
#from IPython.display import display
import warnings
warnings.filterwarnings("ignore")  # 过滤警告信息

# Patient_ID -> 病人ID
# Systemic Illness  -> 全身疾病 (清洗成0、1、2.....)
# Rectal Pain -> 直肠疼痛（TRUE/FALSE）
# Sore Throat -> 喉咙痛（TRUE/FALSE）
# Rectal Pain -> 直肠疼痛（TRUE/FALSE）
# Penile Oedema -> 阴茎水肿（TRUE/FALSE）
# Oral Lesions -> 口腔病变（TRUE/FALSE）
# Solitary Lesion -> 孤立性病变（TRUE/FALSE）
# Swollen Tonsils -> 扁桃体肿胀（TRUE/FALSE）
# HIV Infection -> HIV感染（TRUE/FALSE）
# Sexually Transmitted Infection => 性传播感染（TRUE/FALSE）
# MonkeyPox => 猴痘（Positive/Negative）
#
# TRUE=1，FALSE=0
# Positive=1,Negative=0

DATA = pd.read_csv(r"DATA.csv")

print(DATA.head())
print(DATA.info())
print(DATA.describe().T)
print(DATA["Systemic Illness"].min())

### Maymun çiçeği hastalarının oranı

DATA['MonkeyPox'].value_counts(normalize=True)

plt.rcParams["font.sans-serif"] = ['SimHei']
plt.rcParams["axes.unicode_minus"] = False
fig, ax = plt.subplots(figsize=(8, 4))
sns.countplot(x='MonkeyPox', data=DATA, ax=ax,palette='Set1')
ax.set_title("Maymun Çiçeği Dağılımı", fontsize=15)

plt.show()

### veri ön işleme

def createDataSet(filepath):
    #veri ön işleme
    data = pd.read_csv(filepath)
    #  Patient_ID sütununu kaldırıyoruz
    data = data.drop(columns=['Patient_ID'], axis=1)
    #  özellik etiketi
    labels = data.columns.values.tolist()
    # True/false değerlerini 0 ve 1 e çevriyirouz
    for item in labels:
        data[item].replace({False: 0, True: 1}, inplace=True)
    # Sistemik Hastalık sayısal dönüştürme
    data['Systemic Illness'].replace({'None': 0, 'Fever': 1, 'Swollen Lymph Nodes': 2, 'Muscle Aches and Pain': 3}, inplace=True)
    data['MonkeyPox'].replace({'Positive': 1, 'Negative': 0}, inplace=True)
    return data, labels

dataset,columnLabels = createDataSet('DATA.csv')
print(dataset.head())


#veri görüntüleme

sns.countplot(x='Systemic Illness', data=DATA, hue='MonkeyPox',palette='Set1')
plt.title("Sistemik Hastalığın Maymun Çiçeğine Yol Açma Grafiği")
plt.show()

sns.countplot(x='Sore Throat', data=DATA, hue='MonkeyPox',palette='Set1')
plt.title("Boğaz Ağrısının Maymun Çiçeğine Yol Açma Grafiği")
plt.show()

sns.countplot(x='Penile Oedema', data=DATA, hue='MonkeyPox',palette='Set1')
plt.title("Penis Ödeminin Maymun Çiçeğine Yol Açma Grafiği")
plt.show()

sns.countplot(x='Oral Lesions', data=DATA, hue='MonkeyPox',palette='Set1')
plt.title("Ağız Lezyonlarının  Maymun Çiçeğine Yol Açma Grafiği")
plt.show()

sns.countplot(x='Solitary Lesion', data=DATA, hue='MonkeyPox',palette='Set1')
plt.title("Soliter Lezyonların Maymun Çiçeğine Yol Açma Grafiği")
plt.show()

sns.countplot(x='Swollen Tonsils', data=DATA, hue='MonkeyPox',palette='Set1')
plt.title("Şişmiş Bademciklerin Maymun Çiçeğine Yol Açma Grafiği")
plt.show()

sns.countplot(x='HIV Infection', data=DATA, hue='MonkeyPox',palette='Set1')
plt.title("HIV hastalığının Maymun Çiçeğine Yol Açma Grafiği")
plt.show()

sns.countplot(x='Sexually Transmitted Infection', data=DATA, hue='MonkeyPox',palette='Set1')
plt.title("Cinsel Yolla Geçen Hastalıkların Maymun Çiçeğine Yol Açma Grafiği")
plt.show()