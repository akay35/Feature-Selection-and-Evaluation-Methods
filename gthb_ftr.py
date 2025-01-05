import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve
import shap
from sklearn.datasets import load_iris
from sklearn.feature_selection import RFE
from statsmodels.stats.outliers_influence import variance_inflation_factor
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

warnings.simplefilter(action="ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.3f' % x)  # virgülden sonra 3 vasamak göster
warnings.simplefilter(action='ignore', category=Warning)

# Telco müşteri churn verileri, üçüncü çeyrekte Kaliforniya'daki 7043 müşteriye ev telefonu ve İnternet hizmetleri sağlayan
# hayali bir telekom şirketi hakkında bilgi içerir. Hangi müşterilerin hizmetlerinden ayrıldığını, kaldığını veya hizmete kaydolduğunu içermektedir.

# 21 Değişken 7043 Gözlem

# CustomerId : Müşteri İd’si
# Gender : Cinsiyet
# SeniorCitizen : Müşterinin yaşlı olup olmadığı (1, 0)
# Partner : Müşterinin bir ortağı olup olmadığı (Evet, Hayır) ? Evli olup olmama
# Dependents : Müşterinin bakmakla yükümlü olduğu kişiler olup olmadığı (Evet, Hayır) (Çocuk, anne, baba, büyükanne)
# tenure : Müşterinin şirkette kaldığı ay sayısı
# PhoneService : Müşterinin telefon hizmeti olup olmadığı (Evet, Hayır)
# MultipleLines : Müşterinin birden fazla hattı olup olmadığı (Evet, Hayır, Telefon hizmeti yok)
# InternetService : Müşterinin internet servis sağlayıcısı (DSL, Fiber optik, Hayır)
# OnlineSecurity : Müşterinin çevrimiçi güvenliğinin olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# OnlineBackup : Müşterinin online yedeğinin olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# DeviceProtection : Müşterinin cihaz korumasına sahip olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# TechSupport : Müşterinin teknik destek alıp almadığı (Evet, Hayır, İnternet hizmeti yok)
# StreamingTV : Müşterinin TV yayını olup olmadığı (Evet, Hayır, İnternet hizmeti yok) Müşterinin, bir üçüncü taraf sağlayıcıdan televizyon programları yayınlamak için İnternet hizmetini kullanıp kullanmadığını gösterir
# StreamingMovies : Müşterinin film akışı olup olmadığı (Evet, Hayır, İnternet hizmeti yok) Müşterinin bir üçüncü taraf sağlayıcıdan film akışı yapmak için İnternet hizmetini kullanıp kullanmadığını gösterir
# Contract : Müşterinin sözleşme süresi (Aydan aya, Bir yıl, İki yıl)
# PaperlessBilling : Müşterinin kağıtsız faturası olup olmadığı (Evet, Hayır)
# PaymentMethod : Müşterinin ödeme yöntemi (Elektronik çek, Posta çeki, Banka havalesi (otomatik), Kredi kartı (otomatik))
# MonthlyCharges : Müşteriden aylık olarak tahsil edilen tutar
# TotalCharges : Müşteriden tahsil edilen toplam tutar
# Churn : Müşterinin kullanıp kullanmadığı (Evet veya Hayır) - Geçen ay veya çeyreklik içerisinde ayrılan müşteriler

df = pd.read_csv("Telco-Customer-Churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
df["Churn"] = df["Churn"].apply(lambda x: 1 if x == "Yes" else 0)
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
def grab_col_names(dataframe, cat_th=3, car_th=15):

    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

################################################################################################################
######################## FEATURE ENGINEERING

df["MultipleLines"].unique()   #array(['No phone service', 'No', 'Yes'], dtype=object)
df["MultipleLines_New"] = df["MultipleLines"].apply(
    lambda x: "no" if x == "No phone service" else ("yes" if x == "Yes" else "no"))

df["TenureCategory"] = pd.qcut(df["tenure"], 6, labels=[1, 2, 3, 4, 5, 6])
df['CLTV'] = df['tenure'] * df['MonthlyCharges']
df['PaymentConsistency'] = df['TotalCharges'] / df['tenure'].replace(0, 1)
df['ServiceIntensity'] = (
    (df['PhoneService'] == 'Yes').astype(int) +
    (df['MultipleLines'] == 'Yes').astype(int) +
    (df['InternetService'] != 'No').astype(int) +
    (df['OnlineSecurity'] == 'Yes').astype(int) +
    (df['OnlineBackup'] == 'Yes').astype(int) +
    (df['DeviceProtection'] == 'Yes').astype(int) +
    (df['TechSupport'] == 'Yes').astype(int) +
    (df['StreamingTV'] == 'Yes').astype(int) +
    (df['StreamingMovies'] == 'Yes').astype(int)
)
##########################################
df['DependencyRatio'] = (
    (df['Dependents'] == 'Yes').astype(int) /
    ((df['Partner'] == 'Yes').astype(int).replace(0, 1)))
##########################################
def create_gender_category(row):
    if row['gender'] == 'Male' and row['SeniorCitizen'] == 1:
        return 'MALE_SENIOR'
    elif row['gender'] == 'Male' and row['SeniorCitizen'] == 0:
        return 'MALE_NORMAL'
    elif row['gender'] == 'Female' and row['SeniorCitizen'] == 1:
        return 'FEMALE_SENIOR'
    else:
        return 'FEMALE_NORMAL'

# Yeni kolonu ekleyelim
df['Gender_Senior'] = df.apply(create_gender_category, axis=1)
##########################################
def determine_security_backup(row):
    if row["OnlineSecurity"] == "Yes" and row["OnlineBackup"] == "Yes":
        return "Both"
    elif row["OnlineSecurity"] == "Yes" and row["OnlineBackup"] == "No":
        return "Security"
    elif row["OnlineSecurity"] == "No" and row["OnlineBackup"] == "Yes":
        return "Backup"
    else:
        return "Nothing"

# apply fonksiyonu ile her satıra uyguluyoruz
df["Security_Backup"] = df.apply(determine_security_backup, axis=1)
##########################################
odeme_tipi = ["Month-to-month", "One year", "Two year"]
df["Contract"] = pd.Categorical(df["Contract"], categories=odeme_tipi, ordered=True)
df["Contract_Encoded"] = df["Contract"].cat.codes ## Sıralı veriyi kullanarak label encoding
##---------------------------------------------------------------------------------------------------
df["InternetService"].unique()  #Out[99]: array(['DSL', 'Fiber optic', 'No'], dtype=object)
internet_tipi = ["No", "DSL", "Fiber optic"]
df["InternetService"] = pd.Categorical(df["InternetService"], categories=internet_tipi, ordered=True)
df["Int_Service_Enc"] = df["InternetService"].cat.codes

################################################################################################################
######################## DROP

df.drop(["customerID", "gender", "SeniorCitizen", "MonthlyCharges", "TotalCharges", "PhoneService",
         "Contract", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
         "TechSupport", "StreamingTV", "StreamingMovies", "Dependents", "tenure"], inplace = True, axis=1)

################################################################################################################
######################## ONE HOT ENCODING - LABEL ENCODING

dff = df.copy()
cat_cols, num_cols, cat_but_car = grab_col_names(dff)
cat_cols = [col for col in cat_cols if col not in ["Churn"]]

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe
dff = one_hot_encoder(dff, cat_cols, drop_first=True)

cat_cols, num_cols, cat_but_car = grab_col_names(dff)
for col in cat_cols:
    dff[col] = dff[col].replace({True: 1, False: 0})

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
################################################################################################################
######################## VIF ( Variance Inflation Factor )
a = df[num_cols]
vif_data = pd.DataFrame()
vif_data['Feature'] = a.columns
vif_data['VIF'] = [variance_inflation_factor(a.values, i) for i in range(a.shape[1])]

print(vif_data)
# print(vif_data)
#               Feature   VIF
# 0      TenureCategory 5.096
# 1                CLTV 7.815
# 2  PaymentConsistency 1.297
# 3    ServiceIntensity 3.522
# 4    Contract_Encoded 2.207
# 5     Int_Service_Enc 2.595

################################################################################################################
######################## RFE ( Recursive Feature Elimination )
b = dff['Churn']
A = dff.drop(["Churn"], axis=1)
X_trainA, X_test, y_trainA, y_test = train_test_split(A, b, test_size=0.2, random_state=17)
modelA = LGBMClassifier()
rfe = RFE(modelA, n_features_to_select=12)  # Burada 12 özellik seçilecektir
X_train_rfe = rfe.fit_transform(X_trainA, y_trainA)
selected_features = A.columns[rfe.support_]
ranking = rfe.ranking_
# Seçilen Özellikler: Index(['TenureCategory', 'CLTV', 'PaymentConsistency', 'ServiceIntensity', 'Contract_Encoded',
#                            'Int_Service_Enc', 'Partner_Yes', 'PaperlessBilling_Yes', 'PaymentMethod_Electronic check',
#                            'Gender_Senior_MALE_NORMAL', 'Security_Backup_Nothing', 'DependencyRatio_1.0'], dtype='object')
columns_to_drop = [col for col in dff.columns if col not in selected_features + ['Churn']]
dff_filtered = dff.drop(columns=columns_to_drop)
ranking_df = pd.DataFrame({'Feature': A.columns, 'Ranking': ranking}).sort_values(by='Ranking', ascending=True)
#                                   Feature  Ranking
# 0                          TenureCategory        1
# 1                                    CLTV        1
# 2                      PaymentConsistency        1
# 3                        ServiceIntensity        1
# 4                        Contract_Encoded        1
# 5                         Int_Service_Enc        1
# 6                             Partner_Yes        1
# 18                Security_Backup_Nothing        1
# 8                       MultipleLines_Yes        1
# 9                    PaperlessBilling_Yes        1
# 15              Gender_Senior_MALE_NORMAL        1
# 11         PaymentMethod_Electronic check        1
# 20                    DependencyRatio_1.0        2
# 14            Gender_Senior_FEMALE_SENIOR        3
# 10  PaymentMethod_Credit card (automatic)        4
# 17                   Security_Backup_Both        5
# 16              Gender_Senior_MALE_SENIOR        6
# 12             PaymentMethod_Mailed check        7
# 7          MultipleLines_No phone service        8
# 19               Security_Backup_Security        9
# 13                  MultipleLines_New_yes       10

################################################################################################################
######################## MODEL BUILDING

y = dff["Churn"]
X = dff.drop(["Churn"], axis=1)

################################################################################################################
######################## LightGBM

lgbm_model = LGBMClassifier()
LGBMClassifier().get_params()

lgbm_final = lgbm_model.set_params(
    colsample_bytree=0.5,
    learning_rate=0.01,
    n_estimators=300,
    class_weight='balanced',
    num_leaves= 35).fit(X, y)

cv_results = cross_validate(lgbm_final, X, y, cv=10, scoring=["accuracy", "f1", "recall", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()
cv_results['test_recall'].mean()

# cv_results['test_accuracy'].mean()
# Out[581]: 0.7410174887169567
# cv_results['test_f1'].mean()
# Out[582]: 0.6181219298864251
# cv_results['test_roc_auc'].mean()
# Out[583]: 0.8321368552018391
# cv_results['test_recall'].mean()
# Out[584]: 0.7891926858720029

################################################
# Feature Importance
################################################

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(lgbm_final, X)
