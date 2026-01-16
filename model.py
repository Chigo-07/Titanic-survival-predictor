import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv("titanic.csv")
data = data[["Pclass", "Sex", "Age", "Fare", "Survived"]]
data["Sex"] = data["Sex"].map({"male": 0, "female": 1})
data["Age"].fillna(data["Age"].mean(), inplace=True)

X = data.drop("Survived", axis=1)
y = data["Survived"]

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)

with open("model.h5", "wb") as f:
    pickle.dump((model, scaler), f)

print("Titanic model saved!")
