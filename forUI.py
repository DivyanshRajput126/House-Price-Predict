import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostRegressor
from sklearn.pipeline import Pipeline

# data = pd.read_csv(r"C:\Users\Divyansh\OneDrive\Desktop\.vscode\PYTHON\Infolabz\house_price_prediction_dataset.csv")
data = pd.read_csv('./house_price_prediction_dataset.csv')

# data.head()

# data.info()

# data.isna().sum()

# data.describe()

data.shape

X = data.iloc[:,:-1]
y = data.iloc[:,-1:]

# X.head()

# y.head()

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=32)

X_train.shape,X_test.shape,y_train.shape,y_test.shape

pipe = Pipeline([
    ('Scaler',StandardScaler()),
    ('Model',AdaBoostRegressor())
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

y_pred

y_test

print(pipe.score(X_train,y_train)*100)

print(pipe.score(X_test,y_test)*100)
