import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


# ============================================================================
# STEP 1 : Get data (load) from csv files
# ============================================================================
Border = "=" * 40

print(Border)
print("Step 1 : Get data (load) from csv files ")
print(Border)

# Load csv files
df = pd.read_csv("WinePredictor.csv")

print(f"Dataset loaded successfully!")
print(f"Total Records : {df.shape[0]}")
print(f"Total Features: {df.shape[1]}")
print(f"\nFirst 5 rows :")
print(df.head())

# STEP 2 : Clean, Prepare and Manipulate Data
# Features (X) or Target (y)
# ============================================================

print(Border)
print(" STEP 2 : Clean, Prepare and Manipulate Data")
print(Border)

# Null values check karo
print("Null values check :")
print(df.isnull().sum())
df = df.sample(frac=1, random_state=42).reset_index(
    drop=True
)  # frac = 100% data randomly   # reset.index(drop=true)=index 0,1,2,3,..clean
print("\nData shuffled successfully")

# X = 13 features (Alcohol, Malic acid, Ash ... Proline)
# y = Class (1, 2, 3)
# iloc is used to select row and columns using index numbers
X = df.iloc[:, 1:].values  #  :,All columns except Class   1: = column index 1 to all
y = df.iloc[:, 0].values  # first column = Class

scalar = StandardScaler()  # transform = us range se scale karo
X = scalar.fit_transform(X)  # fit=learn data mean from range


print(f"\nFeatures (X) shape : {X.shape}")
print(f"Target   (y) shape : {y.shape}")
print(f"Classes available  : {np.unique(y)}")
print("Scalling Done  : Yes (StandardScalar)")

# ============================================================
# STEP 3 : Train Data
#  Divide Dataset into half - model train
# ============================================================

print(Border)
print(" STEP 3 : Train Data")
print(Border)


mid = len(X) // 2

X_train = X[:mid]
y_train = y[:mid]

print(f"Total data    : {len(X)} records")
print(f"Training data : {len(X_train)} records (first half)")

# KNN Model create (default k=3)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

print("Model trained successfully")

# ============================================================
# STEP 4 : Test Data
# Remaining half data use to - predict
# ============================================================

print(Border)
print(" STEP 4 : Test Data")
print(Border)

X_test = X[mid:]  # Second half - testing
y_test = y[mid:]  # Second half  actual labels

print(f"Testing data  : {len(X_test)} records (second half)")


y_pred = model.predict(X_test)

print("Prediction done successfully")

# Predicted vs Expected table
print(f"\n  {'Sr':<5} {'Expected':>10} {'Predicted':>12} {'Match':>12}")
print(f"  {'-'*5} {'-'*10} {'-'*12} {'-'*12}")

for i in range(len(y_test)):
    match = "Correct" if y_test[i] == y_pred[i] else "Incorrect"
    print(f"  {i+1:<5} {y_test[i]:>10} {y_pred[i]:>12} {match:>12}")

# ============================================================
# STEP 5 : Calculate Accuracy
# Each K value create accuracy calculate
# ============================================================

print(Border)
print(" STEP 5 : Accuracy by changing value of K")
print(Border)

print(f"  {'K':<5} {'Accuracy':>10}")
print(f"  {'-'*5} {'-'*10}")

best_k = 1
best_acc = 0

# K = 1 se len(X)//2 tak har value try karo
for k in range(1, len(X) // 2 + 1):

    # each K create new model
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    pred = knn.predict(X_test)

    acc = accuracy_score(y_test, pred) * 100

    print(f"  {k:<5} {acc:>9.1f} %")

    if acc > best_acc:
        best_acc = acc
        best_k = k
print(f"\n Best k = {best_k}")
print(f" Best Acc = {best_acc:.1f} %")
