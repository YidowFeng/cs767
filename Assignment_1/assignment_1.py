import pandas as pd
import sys
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import accuracy_score
from scipy import stats

x_all = fetch_california_housing(as_frame=True) # Attributes
df = pd.DataFrame(data=x_all.data,
                  columns=x_all.feature_names)
y_all = x_all.target # Target value - median house value
print(x_all.DESCR)

df.hist(figsize=(10,10))
plt.show()

# print(y_all.shape)
# x_all= x_all.transpose()
X_train, X_test, y_train, y_test = train_test_split(df, y_all, test_size=0.3, random_state=42)
huber = HuberRegressor(alpha=0,epsilon=sys.maxsize).fit(X_train, y_train)
pred = huber.predict(X_test)
print(pred)
acc = huber.score(X_test, y_test)
print(stats.describe(pred))
print("accuracy is:", acc)
print("Huber linear function coefficients:", huber.coef_)

huber = HuberRegressor(alpha=0,epsilon=1).fit(X_train, y_train)
pred = huber.predict(X_test)
print(pred)
print(stats.describe(pred))
acc = huber.score(X_test, y_test)
print("accuracy is:", acc)
print("Huber linear function coefficients:", huber.coef_)



