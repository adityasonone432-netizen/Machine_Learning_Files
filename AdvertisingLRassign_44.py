import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# ============================================================================
# STEP 1 : Get data (load) from csv files
# ============================================================================
Border = "=" * 40

print(Border)
print("Step 1 : Get data (load) from csv files ")
print("load data succesufully")
print(Border)
df = pd.read_csv("MarvellousAdvertising .csv")
print(f"\n Total Records :{len(df)}")
print(df.head())
print(df.tail())

# ===========================================================================
# STEP 2 : Clean, Prepare and Manipulate Data
# ===========================================================================

print(Border)
print("STEP 2 : Clean, Prepare and Manipulate Data ")
print(Border)
X = df[["TV", "radio", "newspaper"]].values
y = df["sales"].values

print("Features X shape :", X.shape)
print("Target y shape :", y.shape)
# missing  values check
print("\n Null/ missing values check ")
print(df.isnull().sum())

# ===========================================================================
# STEP 3 : Train the data (Divide dataset into half)
# ===========================================================================

print(Border)
print("STEP 3 : Train the data ")
print(Border)

mid = len(X) // 2

X_train = X[:mid]  # first half - training
y_train = y[:mid]  # first half ke labels

print(f"Total data     : {len(X)} records")
print(f"Training data  : {len(X_train)} records (first half) ")

model = LinearRegression()
model.fit(X_train, y_train)
print("\n Model train successfully")
print(
    f"Model Coefficients :{model.coef_}"
)  # m1,m2,m3 = coefficients(TV,radio newspaper)   # y =m1x1 + m2x2 + m3x2 + C
print(f"Model Intercept : {model.intercept_:.4f}")  # C intercept (Starting point)

# ===========================================================================
# STEP 4 : Test the data
# ===========================================================================

print(Border)
print(" STEP 4 : Test the data ")
print(Border)

X_test = X[mid:]  # Second half - testing
y_test = y[mid:]  # Second half ke actual labels

print(f"Testing data   : {len(X_test)} records (second half)")
y_pred = model.predict(X_test)
print("Prediction Done Successfully")

# ===========================================================================
# STEP 5 : Display Predicted vs Expected Values
# Predicted values or actual (expected) values
# ===========================================================================

print(Border)
print(" STEP 5 : Display Predicted vs Expected Values ")
print(Border)
print(f"{'Sr ':<5} {'Expected':>10} {'Predicted' :>12}")
print(f"{"-"*5} {'_'*10} {"-"*12}")

for i in range(len(y_test)):
    print(f"  {i+1:<5} {y_test[i]:>10.2f} {y_pred[i]:>12.2f}")

MSE = mean_squared_error(y_test, y_pred)
RMSE = np.sqrt(MSE)
print(f"\nMean Squared Error  (MSE)  : {MSE:.4f}")
print(f"Root Mean Sq Error  (RMSE) : {RMSE:.4f}")

print("\n" + "=" * 50)
print("Assignment completed successfully")
print("=" * 50)
