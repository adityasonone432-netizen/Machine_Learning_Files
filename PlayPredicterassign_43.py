import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# ====================================================================
# Step 1 :  Get data - Load from CSV file
# ====================================================================
df = pd.read_csv("MarvellousInfosystems_PlayPredictor.csv")
Border = "=" * 50
print(Border)
print("Marvellous infosystem -  play predicter")
print(Border)
print(" STEP 1 : Data loaded from CSV file--------")
print(Border)
print(df.to_string(index=True))
print(f"\n Total Records : {len(df)}")

# =====================================================================
# STEP 2 : Clean, Prepare & Manipulate Data
# =====================================================================

# Label encoder use to create object
le_weather = LabelEncoder()  # weather column encode
le_temp = LabelEncoder()  # temp column encode
le_play = LabelEncoder()  # Play column encode

df["Weather_enc"] = le_weather.fit_transform(df["Weather"])
df["Temperature_enc"] = le_temp.fit_transform(df["Temperature"])
df["Play_enc"] = le_play.fit_transform(df["Play"])

X = df[["Weather_enc", "Temperature_enc"]].values
y = df["Play_enc"].values

print(Border)
print("\n STEP 2 : Data encoded with LabelEncoder")
print(f"  Weather  classes : {list(le_weather.classes_)}")
print(f"  Temp     classes : {list(le_temp.classes_)}")
print(f"  Play     classes : {list(le_play.classes_)}")

print(Border)
# ==================================================================
# STEP 3 : Train Data  (K = 3, whole dataset)
# ===================================================================

K = 3
model = KNeighborsClassifier(n_neighbors=K)
model.fit(X, y)
print(f"\n STEP 3 : Model trained with KNN (K = {K}) on full dataset")
print(Border)

# ===================================================================
# STEP 4 : Test Data
# ==================================================================
print("\nSTEP 4 : Test Predictions")
print("-" * 40)

test_cases = [
    ("Sunny", "Mild"),
    ("Overcast", "Hot"),
    ("Rainy", "Cool"),
    ("Sunny", "Cool"),
]

for w, t in test_cases:
    w_enc = le_weather.transform([w])[0]
    t_enc = le_temp.transform([t])[0]
    y_pred = model.predict([[w_enc, t_enc]])
    result = le_play.inverse_transform(y_pred)[0]
    print(f"  Weather: {w:<10}  Temp: {t:<5}  ->  Play: {result}")


# =================================================================
# STEP 5 : Calculate Accuracy (CheckAccuracy)
# =================================================================
def CheckAccuracy(X, y, k):
    mid = len(X) // 2
    X_train, X_test = X[:mid], X[mid:]  # first four train and remaining test
    y_train, y_test = y[:mid], y[mid:]  # same as labels
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred) * 100


print("\nSTEP 5 : Accuracy by changing value of K")
print("-" * 40)
print(f"  {'K':<5}  {'Accuracy':>10}")
print(f"  {'-'*5}  {'-'*10}")
for k in range(1, len(X) // 2 + 1):
    acc = CheckAccuracy(X, y, k)
    print(f"  K = {k:<3}  {acc:>8.1f} %")

print("\n" + "=" * 50)
print("  Assignment completed successfully!")
print("=" * 50)
