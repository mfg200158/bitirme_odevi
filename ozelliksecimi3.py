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
from xgboost import XGBClassifier
from sklearn.feature_selection import RFE, RFECV
from sklearn.naive_bayes import GaussianNB
from catboost import CatBoostClassifier
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
x_train, x_test, y_train, y_test = train_test_split(df_X_final, df_Y, test_size=0.3, random_state=True)

 # Create the model
rf_model = RandomForestClassifier(n_estimators=100)

# Create the RFE object
rf_model = RFE(rf_model,n_features_to_select= 6 , step=1)

# Apply feature selection
rf_model.fit(x_train, y_train)

y_pred = rf_model.predict(x_test)
print('Random Forest: ')
scores(y_test, y_pred)

# Get the selected feature indices and rankings
selected_features = rf_model.get_support(indices=True)

feature_rankings = rf_model.ranking_
print("Seçilen özelliklerin rank skorları: ")
for feature, rank in zip(selected_features, feature_rankings):
    print(df.columns[feature], "Rank:", rank) 


#####################################################

# Create the model
xg_model = XGBClassifier()

# Create the RFE object
xg_model = RFE(xg_model,n_features_to_select= 6 , step=1)

# Apply feature selection
xg_model.fit(x_train, y_train)

y_pred = xg_model.predict(x_test)
print('XGBoost: ')
scores(y_test, y_pred)

# Get the selected feature indices and rankings
selected_features = xg_model.get_support(indices=True)

feature_rankings = xg_model.ranking_
print("Seçilen özelliklerin rank skorları: ")
for feature, rank in zip(selected_features, feature_rankings):
    print(df.columns[feature], "Derece:", rank)


#####################################################

# Create the model
dt_model = DecisionTreeClassifier()

# Create the RFE object
dt_model = RFE(dt_model,n_features_to_select= 6 , step=1)

# Apply feature selection
dt_model.fit(x_train, y_train)

y_pred = xg_model.predict(x_test)
print('Karar Ağacı: ')
scores(y_test, y_pred)

# Get the selected feature indices and rankings
selected_features = dt_model.get_support(indices=True)

feature_rankings = dt_model.ranking_
print("Seçilen özelliklerin rank skorları: ")
for feature, rank in zip(selected_features, feature_rankings):
    print(df.columns[feature], "Derece:", rank)


#####################################################

# Create the model
ab_model = AdaBoostClassifier()

# Create the RFE object
ab_model = RFE(ab_model,n_features_to_select= 6 , step=1)

# Apply feature selection
ab_model.fit(x_train, y_train)

y_pred = ab_model.predict(x_test)
print('AdaBoost: ')
scores(y_test, y_pred)

# Get the selected feature indices and rankings
selected_features = ab_model.get_support(indices=True)

feature_rankings = ab_model.ranking_
print("Seçilen özelliklerin rank skorları: ")
for feature, rank in zip(selected_features, feature_rankings):
    print(df.columns[feature], "Derece:", rank)

#####################################################
#####################################################

# Create the model
cb_model = CatBoostClassifier(iterations=10, learning_rate=0.3)


# Create the RFE object
cb_model = RFE(cb_model,n_features_to_select= 6 , step=1)

# Apply feature selection
cb_model.fit(x_train, y_train)

y_pred = cb_model.predict(x_test)
print('CatBoost: ')
scores(y_test, y_pred)

# Get the selected feature indices and rankings
selected_features = cb_model.get_support(indices=True)

feature_rankings = cb_model.ranking_
print("Seçilen özelliklerin rank skorları: ")
for feature, rank in zip(selected_features, feature_rankings):
    print(df.columns[feature], "Derece:", rank) 






