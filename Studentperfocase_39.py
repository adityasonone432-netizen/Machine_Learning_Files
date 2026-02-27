import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)
Border="-"*40

#############################################################################################
# Step 1 Load the dataset 
#############################################################################################

print(Border)
print("Step 1: Load the dataset")
print(Border)

df=pd.read_csv("student_performance_ml.csv")
print("Dataset load successfully")
print("initial records :")
print(df.head())

#################################################################################################
# Step 2:Data anaiysis (EDA)
##################################################################################################
print(Border)
print("Step 2:Data anaiysis (EDA) ")
print(Border)

print("shape of dataset :",df.shape)
print("Column names :",list(df.columns))

print("Missing values per column :")
print(df.isnull().sum())

print("Statastical summary :")
print(df.describe())


#################################################################################################
# Step 3:Decide X and y Features / target
##################################################################################################
print(Border)
print("Step 3: Decide X and y Features / target ")
print(Border)

features_cols=[
    "StudyHours",
    "Attendance",
    "PreviousScore",
    "AssignmentsCompleted",
    "SleepHours"
]

X=df[features_cols]   # Independent variable 
y=df["FinalResult"]   # Target variable (pass/fail)

print("X shape :",X.shape)
print("y shape :",y.shape)

#################################################################################################
# Step 4:  Data visualization 
##################################################################################################
print(Border)
print("Step 4: Data visualization ")
print(Border)


# Scatter plot to show relation between study hours & score
plt.figure(figsize=(8,5))

plt.scatter(df["StudyHours"], df["PreviousScore"],c=df["StudyHours"],cmap="viridis",label="Student Performance")



plt.xlabel("Study Hours")
plt.ylabel("Previous Score")
plt.title("Study Hours vs Previous Score")
plt.colorbar(label='Intensity of study')

plt.legend() 
plt.grid(True)
plt.show() 


#########################################################
# Step 5 : Split dataset into Training & Testing
#########################################################

print(Border)
print("Step 5 : Split dataset into Training & Testing ")
print(Border)

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    y,
    test_size=0.3,      # 30% testing
    random_state=42     # shuffle data  results same 
)

print("Data splitting completed")

print("X_train : ", X_train.shape)
print("X_test  : ", X_test.shape)
print("Y_train : ", Y_train.shape)
print("Y_test  : ", Y_test.shape)


#########################################################
# Step 6 : Build  Decision tree model
#########################################################

print(Border)
print("Step 6 : Build Decision Tree model")
print(Border)


model = DecisionTreeClassifier(
    criterion="gini",   # measure quality of split
    max_depth=3,        # control tree depth 
    random_state=42
)

print("Model created successfully :", model)


#########################################################
# Step 7 : Train the model
#########################################################

print(Border)
print("Step 7 : Train the model")
print(Border)

# Train model using training data
model.fit(X_train, Y_train)

print("Model training completed")


#########################################################
# Step 8 : Prediction on test data 
#########################################################

print(Border)
print("Step 8 : Prediction on test data")
print(Border)

# Predict test results
Y_pred = model.predict(X_test)

print("Actual values :")
print(Y_test.values)

print("Predicted values :")
print(Y_pred)


#########################################################
# Step 9 : Evaluate model performance
#########################################################

print(Border)
print("Step 9 : Model performance")
print(Border)

# Calculate accuracy
accuracy = accuracy_score(Y_test, Y_pred)
print("Accuracy of model : ", accuracy * 100, "%")

# Confusion matrix
cm = confusion_matrix(Y_test, Y_pred)
print("Confusion matrix :")
print(cm)

# Classification report (precision, recall, f1)
print("Classification Report :")
print(classification_report(Y_test, Y_pred))


#########################################################
# Step 10 : Plot confusion matrix
#########################################################

print(Border)
print("Step 10 : Plot confusion matrix")
print(Border)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])
disp.plot()

plt.title("Confusion Matrix - Student Performance")
plt.show()


#########################################################
# Step 11 : Predict for new student
#########################################################

print(Border)
print("Step 11 : New student prediction")
print(Border)

# New student data
new_student =pd.DataFrame( 
    [[6, 85, 66, 7, 7]],
    columns=features_cols

)

result = model.predict(new_student)

if result[0] == 1:
    print("Student Will Pass")
else:
    print("Student Will Fail") 






