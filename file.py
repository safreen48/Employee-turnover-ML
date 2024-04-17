
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

df = pd.read_csv("employee.csv")
# print(df) 
# print(df.info())
# print(df.shape)
# print(df.isna().sum())

df = df.rename(columns={'sales': 'department'})
# print(df.info())

# print(df.describe())

print(df["left"].value_counts())

mean = df['satisfaction_level'].mean()
median = df['satisfaction_level'].median()
# print(mean)
# print(median)

mean = df['last_evaluation'].mean()
median = df['last_evaluation'].median()
# rint(mean)
# print(median)

mean = df['number_project'].mean()
median = df['number_project'].median()
# print(mean)
# print(median)

mean = df['average_montly_hours'].mean()
median = df['average_montly_hours'].median()
# print(mean)
# print(median)

mean = df['time_spend_company'].mean()
median = df['time_spend_company'].median()
# print(mean)
# print(median)

mean = df['Work_accident'].mean()
median = df['Work_accident'].median()
# print(mean)
# print(median)

mean = df['left'].mean()
median = df['left'].median()
# print(mean)
# print(median)

mean = df['promotion_last_5years'].mean()
median = df['promotion_last_5years'].median()
# print(mean)
# print(median)

# print(df['sales'].value_counts())
# print(df['salary'].value_counts())

# pd.crosstab(df.department, df.left).plot(kind='bar')
# plt.title('Turnover Frequency for Department')
# plt.xlabel('Department')
# plt.ylabel('Frequency of Turnover')
# plt.savefig('department_bar_chart')
# plt.show()

#print(pd.crosstab(df.department, df.left))


y = df.loc[:, ["left"]].values
x = df.loc[:, ["satisfaction_level", "last_evaluation",
               "number_project", "average_montly_hours", "time_spend_company",
               "Work_accident", "promotion_last_5years", "department", "salary"]].values
# print(y)
# print(x[0])

ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [7, 8])], remainder='passthrough')
x = ct.fit_transform(x)
# print(x.shape)
print(x[0])

sc = StandardScaler()
x = sc.fit_transform(x)

print(x)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0)
# print(x_test[0])

'''regressor = RandomForestRegressor(n_estimators=40, random_state=1)
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)
print(y_pred)

# print('Logistic regression accuracy: {:.3f}'.format(
#    accuracy_score(y_test, regressor.predict(x_test))))

a = accuracy_score(y_pred, y_test, normalize=False)
#b = confusion_matrix(y_test, y_pred)'''

'''svc = SVC()          
svc.fit(x_train, y_train)

print(accuracy_score(y_test, svc.predict(x_test)))'''

logreg = LogisticRegression()
logreg.fit(x_train, y_train)

print(accuracy_score(y_test, logreg.predict(x_test)))
