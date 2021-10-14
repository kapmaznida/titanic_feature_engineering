#############################################
# ÖDEV: İş Problemi
# Bir makine öğrenmesi pipeline’ı için gerekli olan veri ön işleme ve değişken mühendisliği scriptinin hazırlanması gerekmektedir.
# Veri seti bu script’ten geçirildiğinde modellemeye hazır hale gelmesi beklenmektedir.
#############################################

# Görev 1:
# Çalışma dizininde helpers adında bir dizin açıp, içerisine data_prep.py adında bir script ekleyiniz.
# Feature Engineering bölümünde kendimize ait tüm fonksiyonları, bu script içerisine toplayınız.
# Burada olması gereken fonksiyonlar:
# outlier_thresholds, replace_with_thresholds, check_outlier, grab_outliers, remove_outlier,
# missing_values_table, missing_vs_target, label_encoder, one_hot_encoder, rare_analyser, rare_encoder

# Görev 2:
# titanic_data_prep adında bir fonksiyon yazınız.
# Bu fonksiyon için gerekli olan veri ön işleme ya da eda fonksiyonlarını helpers içerisinde yer alan eda.py ve
# data_prep.py dosyaları içerisinden alınız.

# Görev 3:
# Veri ön işleme yaptığınız veri setini pickle ile diske kaydediniz.


## DEĞİŞKENLER
# Survived – Hayatta Kalma 0 Öldü, 1 Hayatta Kaldı
# Pclass – Bilet Sınıfı
# 1 = 1. sınıf, 2 = 2.sınıf, 3 = 3.sınıf
# Sex – Cinsiyet
# Age – Yaş
# Sibsp – Titanicte’ki kardeş / eş sayısı
# Parch – Titanicte’ki ebeveyn / çocuk sayısı
# Embarked: – Yolcunun gemiye biniş yaptığı liman (C = Cherbourg, Q = Queenstown, S = Southampton
# Fare – Bilet ücreti
# Cabin: Kabin numarası

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from helpers.eda import *
from helpers.data_prep import *

from sklearn.metrics import classification_report, roc_auc_score, plot_confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

def load():
    df = pd.read_csv("weeks/week_06/titanic.csv")
    return df

data = load()
check_df(data)

data["Age"].hist(bins = 5)
plt.show()

# categoric, numeric, categoric but cardinal değerlerimizi analiz edelim.

cat_cols, num_cols, cat_but_car_cols = grab_col_names(data, cat_th=10, car_th=20)
# Observations: 891
# Variables: 12
# cat_cols: 6
# num_cols: 3
# cat_but_car: 3
# num_but_cat: 4

for col in cat_cols:
    target_summary_with_cat(data, "Survived", col)
# Dikkatimi çeken kategorik sütunlar:
#         TARGET_MEAN
# Sex
# female        0.742
# male          0.189

#         TARGET_MEAN
# Pclass
# 1             0.630
# 2             0.473
# 3             0.242

for col in num_cols:
    target_summary_with_num(data, "Survived", col)

# PassengerId Id olmasından mütevellit saöma sapan geldi anlamsız çöp.
#            Fare
# Survived
# 0        22.118
# 1        48.395
# Daha çok para ödeyenler yaşamış çıkarımını yapabilecek gibiyiz.

#             Age
# Survived
# 0        30.626
# 1        28.344
# Yaş konusunda can alıcı bir fark bulamadık.

for col in cat_cols:
    cat_summary(data, col, plot=True)

for col in num_cols:
    num_summary(data, col, plot=False)


def titanic_df_pred(df):
    dataframe = df.copy()
    # Feature Engineering
    dataframe.columns = [col.upper() for col in dataframe.columns]
    dataframe["NEW_CABIN_BOOL"] = dataframe["CABIN"].notnull().astype('int')
    dataframe["NEW_TITLE"] = dataframe.NAME.str.extract(' ([A-Za-z]+)\.', expand=False)
    dataframe["FAMILY_SIZE"] = dataframe["SIBSP"] + dataframe["PARCH"] + 1
    dataframe["NEW_IS_ALONE"] = ["YES" if i > 1 else "NO" for i in dataframe["FAMILY_SIZE"]]
    cat_cols, num_cols, cat_but_car_cols = grab_col_names(dataframe)
    num_cols = [col for col in num_cols if "PASSENGERID" not in col]

    # Missing Values
    missing_cols = [col for col in dataframe.columns if (dataframe[col].isnull().any()) & (col != "Cabin")]
    for i in missing_cols:
        if i == "AGE":
            dataframe[i].fillna(dataframe.groupby("PCLASS")[i].transform("median"),inplace=True)
        elif dataframe[i].dtype == "O":
            dataframe[i].fillna(dataframe[i].mode()[0], inplace=True)
        else:
            dataframe[i].fillna(dataframe[i].media(), inplace=True)

    # Outliers
    for col in num_cols:
        replace_with_thresholds(dataframe, col)

    deleted_cols = ["CABIN", "SIBSP", "PARCH", "TICKET", "NAME"]
    dataframe = dataframe.drop(deleted_cols, axis=1)

    dataframe["NEW_AGE_CAT"] = pd.cut(dataframe["AGE"], bins=[0, 25, 40, 55, dataframe["AGE"].max()+1],
                                      labels=[1, 2, 3, 4]).astype('int')

    dataframe.loc[(dataframe["SEX"] == "male") & (dataframe["AGE"] <= 25), "NEW_SEX_CAT"] = "youngmale"
    dataframe.loc[(dataframe["SEX"] == "male") & (
        dataframe["AGE"] > 25) & (dataframe["AGE"] < 55), "NEW_SEX_CAT"] = "maturemale"
    dataframe.loc[(dataframe['SEX'] == 'male') & (dataframe['AGE'] > 55), 'NEW_SEX_CAT'] = 'seniormale'
    dataframe.loc[(dataframe['SEX'] == 'female') & (dataframe['AGE'] <= 20), 'NEW_SEX_CAT'] = 'youngfemale'
    dataframe.loc[(dataframe['SEX'] == 'female') & (
            (dataframe['AGE'] > 20) & (dataframe['AGE']) < 55), 'NEW_SEX_CAT'] = 'maturefemale'
    dataframe.loc[(dataframe['SEX'] == 'female') & (dataframe['AGE'] > 55), 'NEW_SEX_CAT'] = 'seniorfemale'

    cat_cols, num_cols, cat_but_car_cols = grab_col_names(dataframe)

    # Label Encoding
    binary_cols = [col for col in dataframe.columns if dataframe[col].dtype not in [int, float]
                   and dataframe[col].nunique() == 2]

    for col in binary_cols:
        dataframe = label_encoder(dataframe, col)

    # Rare Encoding
    dataframe = rare_encoder(dataframe, 0.01, cat_cols)

    # One-Hot Encoding
    ohe_cols = [col for col in dataframe.columns if 10 >= dataframe[col].nunique() > 2]
    dataframe = one_hot_encoder(dataframe, 0.01,  cat_cols)

    # Standart Scaler
    scaler = StandardScaler()
    dataframe[num_cols] = scaler.fit_transform(dataframe[num_cols])

    return dataframe

data.head()
data.describe()

data_prep = titanic_df_pred(data)

data_prep.head()
data_prep.describe()
check_df(data_prep)
data_prep.isnull().sum().sum()


# Model
y = data_prep["SURVIVED"]
X = data_prep.drop(["PASSENGERID", "SURVIVED"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

print(classification_report(y_test, y_pred))
roc_auc_score(y_test, y_pred)

plot_confusion_matrix(rf_model, X_test, y_pred)
plt.show()

# Model Tuning

rf_params = {"n_estimator": [1000, 1200],
             "max_depth": [10, 12],
             "min_samples_split": [2, 3, 5]}

rf = RandomForestClassifier(random_state=46)

rf_cv = GridSearchCV(rf, rf_params, cv=10, n_jobs=1, verbose=2).fit(X_train, y_train)
rf_cv.best_params_

# Final Model
rf_tuned = rf.set_params (**rf_cv.best_params_, random_state=46).fit(X_train, y_train)
y_final_pred = rf_tuned.predict(X_test)

print (classification_report(y_test, y_final_pred))
roc_auc_score(y_test, y_final_pred)

plot_confusion_matrix(rf_tuned, X_test, y_test)
plt.show ()

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="VALUE", Y="Feature", data= feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])


    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importance.png')

plot_importance(rf_tuned, X_train)

import pandas as pd
df = data = pd.read_csv("weeks/week_06/titanic.csv")

df_final = titanic_df_pred(df)

df_final.head()
df_final.shape


import pickle
pickle.dump(df_final, open("Hws/week6/titanic_last_df.pkl", "wb"))



pd.read_pickle("ws/week6/titanic_last_df.pkl")





























