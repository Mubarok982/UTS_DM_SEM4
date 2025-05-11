# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import pickle

# Baca data
df = pd.read_csv("custome.csv")

# Hapus kolom yang tidak diperlukan
df = df.drop(columns=["CustomerID", "Name"])

# Encode target
df["Churn_Status"] = df["Churn_Status"].astype(int)

# Encode fitur kategorikal
label_encoders = {}
for col in ["Gender", "Membership_Level"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Pisah fitur dan target
X = df.drop("Churn_Status", axis=1)
y = df["Churn_Status"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Latih model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Simpan model dan metadata
with open("churn_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("features.pkl", "wb") as f:
    pickle.dump(list(X.columns), f)

with open("label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

