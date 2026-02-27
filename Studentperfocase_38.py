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
# Step 1 Load and Display basic info 
#############################################################################################

print(Border)
print("Step 1: Load and Display basic info ")
print(Border)

df=pd.read_csv("student_performance_ml.csv")
print("Dataset load successfully")
print("initial records :")

# Q1 basic info 
print(df.head())
print(df.tail())
print("\nTotal rows and columns:",df.shape)
print("\nList of column names :",list(df.columns))
print("\nData types of each column :\n",df.dtypes)

#################################################################################################
# Step 2:Data anaiysis (EDA)
##################################################################################################
print(Border)
print("Step 2:Data anaiysis (EDA)" )
print(Border)

#  Q2 Student counting 
print("Total numbers of students :",len(df))
print("Student Pass (1):",(df["FinalResult"]==1).sum())
print("Student Fail (0):",(df["FinalResult"]==0).sum())
print("Missing values per column :")

# Q3 Statastical calculation mean, max, min 
print(df.isnull().sum())
print("Average StudyHours:", df['StudyHours'].mean())
print("Average Attendance:", df['Attendance'].mean())
print("Maximum PreviousScore:", df['PreviousScore'].max())
print("Minimum SleepHours:", df['SleepHours'].min())


# Q4 Balance anaiysis 
counts = df['FinalResult'].value_counts()
percentage = df['FinalResult'].value_counts(normalize=True) * 100

print("Value Counts:\n", counts)
# print("\nPercentage of Pass and Fail:\n", percentage )
#  or 
print("Percentage of Pass and Fail:")
for index,value in percentage.items():
    print(f"{index}:{value}%")


# Justification
if abs(percentage[0] - percentage[1]) < 30:
    print("\nObservation: Dataset is Balanced ")
else:
    print("\nObservation: Dataset is Unbalanced ")

# Q5 General Obeservation 
print("1. Higher StudyHours generally lead to better results as they improve core knowledge.")
print("2. Higher Attendance ensures the student doesn't miss key concepts, increasing pass chance.")
print("3. PreviousScore is a strong indicator of future academic performance.")
print("4. Balanced SleepHours help in better memory retention during exams.")
    
 



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

# Q6 plot histogram of study hours 
plt.figure(figsize=(8,5))
sns.histplot(df['StudyHours'], kde=True, color='blue')
plt.title(" Distribution of Study Hours")
plt.show() 

# Q8 Create Boxplot for Attendance (Checking outliers)
plt.figure(figsize=(8,5))
sns.boxplot(x=df['Attendance'], color='orange')
plt.title(" Attendance Boxplot (Checking Outliers)")
plt.show()

# Q9 AssignmentsCompleted vs FinalResult
plt.figure(figsize=(8,5))
sns.barplot(x='FinalResult', y='AssignmentsCompleted', data=df)
plt.title(" Relationship - Assignments vs Result")
plt.show()

# Q10 SleepHours vs FinalResult
plt.figure(figsize=(8,5))
sns.violinplot(x='FinalResult', y='SleepHours', data=df)
plt.title(" SleepHours vs FinalResult")
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






