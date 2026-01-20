import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

# 1. Load the Titanic dataset
# Ensure 'titanic.csv' is in a 'data' folder
try:
    df = pd.read_csv('../data/titanic.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: titanic.csv not found. Check the path.")

# 2. Data Preprocessing
# Selected Features: Pclass, Sex, Age, SibSp, Fare
selected_features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Fare']
target = 'Survived'

# Create a working dataframe
data = df[selected_features + [target]].copy()

# a. Handling missing values
data['Age'] = data['Age'].fillna(data['Age'].median())
data['Fare'] = data['Fare'].fillna(data['Fare'].median())

# b. Encoding categorical variables
# We use LabelEncoder for 'Sex' and save it to ensure the App uses the exact same mapping
le_sex = LabelEncoder()
data['Sex'] = le_sex.fit_transform(data['Sex'])

# c. Split Data (Validation Step - addressing feedback)
X = data[selected_features]
y = data[target]

# Split: 80% Training, 20% Validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Implement Machine Learning Algorithm (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)

# 4. Train the model
model.fit(X_train, y_train)

# 5. Evaluate the model
# Using the Validation set to generate the report
y_pred = model.predict(X_val)

print("--- Model Evaluation ---")
print(f"Accuracy Score: {accuracy_score(y_val, y_pred):.4f}")
print("Classification Report:")
print(classification_report(y_val, y_pred))

# 6. Save the model and encoder
joblib.dump(model, 'titanic_survival_model.pkl')
joblib.dump(le_sex, 'sex_encoder.pkl') 

print("Model and Encoder saved successfully!")

# 7. Verification: Reloading
print("\n--- Reload Verification ---")
try:
    loaded_model = joblib.load('titanic_survival_model.pkl')
    test_prediction = loaded_model.predict([X_val.iloc[0]])
    print("Model reloaded and prediction test passed.")
except Exception as e:
    print(f"Reload failed: {e}")