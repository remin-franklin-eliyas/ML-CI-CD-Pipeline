import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
import pickle
import os

data = load_breast_cancer()
x = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

os.makedirs('app/artifacts', exist_ok=True)
with open('app/artifacts/model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved successfully.")