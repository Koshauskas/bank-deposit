#Project_classification

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from  sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing  import LabelEncoder
from sklearn import linear_model 
from sklearn import tree 
from sklearn import ensemble 
from sklearn import metrics 
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split 
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('bank_fin.csv', sep = ';')
pd.set_option('display.max_columns', 500) # show all columns in terminal
9

print(df.info()) #there are columns with missing data

cols_null_percent = df.isnull().mean() * 100 #percent of missing data in each column

cols_with_null = cols_null_percent[cols_null_percent > 0].sort_values(ascending=False) #sorted information of missing data 

print(cols_with_null)

print(df['job'].value_counts())
print(df['education'].value_counts())


df['balance'] = df['balance'].apply(lambda x: (str(x).replace('na', '0').replace(' ','').replace(',','.')[:-1])) #put values in column 'balance' in right form
df['balance'] = df['balance'].astype('float')

print(df['balance'].mean())

print(df[(df['job'] == 'management') & (df['education'] == 'secondary')]['balance'].mean())

def outliers_iqr(data, feature): #check feature if it has outliers
    x = data[feature]
    quartile_1, quartile_3 = x.quantile(0.25), x.quantile(0.75),
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    print('lower: ', lower_bound)
    print('upper: ', upper_bound)
    outliers = data[(x < lower_bound) | (x > upper_bound)]
    cleaned = data[(x >= lower_bound) & (x <= upper_bound)]
    return outliers, cleaned

outliers, df = outliers_iqr(df, 'balance')

print(df.info())

print(df['deposit'].value_counts())
#sns.countplot(df, x='deposit') check
df['deposit'] = df['deposit'].apply(lambda x: 1 if x == 'yes' else 0) #rewrite column 'deposit' to binary form
print(df.describe())
types = df.dtypes
cat_features = list(types[(types == 'object')].index)

n = len(cat_features)
#fig, axes = plt.subplots(n, 2, figsize=(15, 40))
#for i, feature in enumerate(cat_features):
#    count_data = (df[feature].value_counts(normalize=True)
#                  .sort_values(ascending=False)
#                  .rename('percentage')
#                  .reset_index())
#    count_barplot = sns.barplot(data=count_data, x='index', y='percentage', ax=axes[i][0])
#    count_barplot.xaxis.set_tick_params(rotation=60)
#    mean_barplot = sns.barplot(data=df, x=feature, y='income', ax=axes[i][1])
#    mean_barplot.xaxis.set_tick_params(rotation=60)
#plt.tight_layout()
print(df.describe(include='object'))

print(df['poutcome'].value_counts())

months = list(set(df['month'].values))
#print(months)
for i in months: #in what months clients prefer open or not open deposit
    print(i)
    print(df[df['month'] == i]['deposit'].value_counts(normalize=True))

def conditions(x): #make age categories
    if x < 30:
        return '<30'
    elif (x <= 40 and x >= 30):
        return '30-40'
    elif (x <= 50 and x > 40):
        return '40-50'
    elif (x <= 60 and x > 50):
        return '50-60'
    else:
        return '60+'

df['ages'] = df['age'].apply(conditions)

ages_intervals = list(set(df['ages'].values))
for i in ages_intervals: #in what ages clients prefer open or not open deposit
    print(i)
    print(df[df['ages'] == i]['deposit'].value_counts(normalize=True))

maritals = list(set(df['marital'].values))
for i in maritals: #clients with which marital status prefer open or not open deposit
    print(i)
    print(df[df['marital'] == i]['deposit'].value_counts(normalize=True))



open_dep = df[df['deposit'] == 1]
not_open_dep = df[df['deposit'] == 0]
pivot_open = open_dep.pivot_table(values="job", index='marital', columns='education', aggfunc='count')
print(pivot_open) # secondary +married most open deposit; unknown +divorced - less
pivot_not_open = not_open_dep.pivot_table(values="job", index='marital', columns='education', aggfunc='count')
print(pivot_not_open)


#le = preprocessing.LabelEncoder()
#cols = list(df.select_dtypes(include='object').columns)
#for i in cols:
#    le.fit(df[i])
#    le.transform(df[i])

#le.fit(df['education'])
#le.transform(df['education'])
#print(list(le.classes_))
#print(df['deposit'].std())

cols = ['default', 'housing', 'loan']
for col in cols:
    df[col] = df[col].apply(lambda x: 1 if x == 'yes' else 0)

nom_cols = ['job', 'marital', 'contact', 'month', 'poutcome', 'education', 'ages']

dummies_df = pd.get_dummies(df, columns = nom_cols)
dummies_df = dummies_df.drop(['default', 'housing', 'loan'], axis=1)
print(dummies_df.head())
print(dummies_df.corr()) # correlation 0.45 with duration
plt.figure(figsize = (10, 5))
sns.heatmap(dummies_df.corr(), annot = True)

X = dummies_df.drop(['deposit'], axis=1)
y = dummies_df['deposit']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state = 42, test_size = 0.33) #split data to test and train datas

selector = SelectKBest(f_classif, k=15) #features selection
selector.fit(X_train, y_train)
print(selector.get_feature_names_out())

scaler = preprocessing.MinMaxScaler() #scaling each feature to a given range - [0, 1]
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


log_reg = linear_model.LogisticRegression(
    solver='sag', 
    random_state=1, 
    max_iter=1000 
) #build logistic regression model
log_reg.fit(X_train_scaled, y_train)
y_train_pred = log_reg.predict(X_train_scaled)
print(metrics.classification_report(y_train, y_train_pred))
y_test_pred = log_reg.predict(X_test_scaled)
print(metrics.classification_report(y_test, y_test_pred))

dt = tree.DecisionTreeClassifier(
    criterion='entropy',
    random_state=42
) #build decision tree model

dt.fit(X_train, y_train)

y_train_pred = dt.predict(X_train)
print('Train: {:.2f}'.format(metrics.f1_score(y_train, y_train_pred)))
y_test_pred = dt.predict(X_test)
print('Test: {:.2f}'.format(metrics.f1_score(y_test, y_test_pred)))

depths = list(range(3,8)) #build several decision tree models with different max_depths
for depth in depths:
    print('Max depth is ' + str(depth))
    dt = tree.DecisionTreeClassifier(
    criterion='entropy',
    max_depth=depth,
    random_state=42)
    dt.fit(X_train, y_train)
    y_train_pred = dt.predict(X_train)
    print('Train: {:.2f}'.format(metrics.f1_score(y_train, y_train_pred)))
    y_test_pred = dt.predict(X_test)
    print('Test: {:.2f}'.format(metrics.f1_score(y_test, y_test_pred)))

param_grid = [{'min_samples_split': [2, 5, 7, 10], 'max_depth':[3,5,7]}]
grid_search = GridSearchCV(
    estimator=tree.DecisionTreeClassifier(
        criterion='entropy',
        random_state=42 
    ), 
    param_grid=param_grid, 
    cv=5, 
    n_jobs = -1
)
grid_search.fit(X_train, y_train) 

print("Best parameter values: {}".format(grid_search.best_params_))
print("accuracy on test data: {:.2f}".format(grid_search.score(X_test, y_test)))
y_test_pred = grid_search.predict(X_test)
print('f1_score on test data: {:.2f}'.format(metrics.f1_score(y_test, y_test_pred)))

rf = ensemble.RandomForestClassifier(n_estimators = 100,
criterion = 'gini',
min_samples_leaf = 5,
max_depth = 10,
random_state = 42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_train)
y_pred_test_rf = rf.predict(X_test)
print("Metrics for train data RF: \n", metrics.classification_report(y_train, y_pred_rf))
print("Metrics for test data RF: \n", metrics.classification_report(y_test, y_pred_test_rf))

gbc = ensemble.GradientBoostingClassifier(learning_rate = 0.05,
n_estimators = 300,
min_samples_leaf = 5,
max_depth = 5,
random_state = 42)
gbc.fit(X_train, y_train)
y_pred_gbc = gbc.predict(X_train)
y_pred_test_gbc = gbc.predict(X_test)
print("Metrics for train data GBC: \n", metrics.classification_report(y_train, y_pred_gbc))
print("Metrics for test data GBC: \n", metrics.classification_report(y_test, y_pred_test_gbc))

print('Random Forest model gives a better forecast, gradient boosting leads to overfitting')

