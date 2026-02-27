import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

Border = "-"*50

# --- PREPARATION ---
df = pd.read_csv("student_performance_ml.csv")
X = df.drop('FinalResult', axis=1)
y = df['FinalResult']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Training a base model for initial steps
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

#############################################################################################
# Q1: Feature Importance
#############################################################################################
print(Border)
print("Q1: Feature Importance Analysis")
print(Border)

# Use: To identify which features (StudyHours, Attendance, etc.) affect the result the most.
importances = model.feature_importances_
for feature, importance in zip(X.columns, importances):
    print(f"Feature: {feature:20} | Score: {importance:.4f}")

# Finding Max and Min contribution
print("\nMost contributing feature:", X.columns[importances.argmax()])
print("Least contributing feature:", X.columns[importances.argmin()])


#############################################################################################
# Q2: Remove SleepHours and Compare Accuracy
#############################################################################################
print(Border)
print("Q2: Removing 'SleepHours' Feature")
print(Border)

# Use: To check if 'SleepHours' is a redundant feature or if it significantly affects performance.
X_new = X.drop('SleepHours', axis=1)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_new, y, test_size=0.3, random_state=42)

model2 = DecisionTreeClassifier(max_depth=3, random_state=42)
model2.fit(X_train2, y_train2)
new_acc = accuracy_score(y_test2, model2.predict(X_test2))

print(f"Original Accuracy (All features): {accuracy_score(y_test, model.predict(X_test))*100:.2f}%")
print(f"New Accuracy (Without SleepHours): {new_acc*100:.2f}%")


#############################################################################################
# Q3: Train using ONLY StudyHours and Attendance
#############################################################################################
print(Border)
print("Q3: Training with only StudyHours & Attendance")
print(Border)

# Use: To see if the model can still perform well using only the two most common behavioral factors.
X_minimal = df[['StudyHours', 'Attendance']]
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_minimal, y, test_size=0.3, random_state=42)

model_min = DecisionTreeClassifier(max_depth=3, random_state=42)
model_min.fit(X_train_m, y_train_m)
min_acc = accuracy_score(y_test_m, model_min.predict(X_test_m))
print(f"Minimal Feature Accuracy: {min_acc*100:.2f}%")


###############################################################################################
# Q4: Prediction for 5 New Students
###############################################################################################
# Use: To check if our model can predict for unknown data.
new_data = {
    'StudyHours': [8, 2, 5, 9, 4],
    'Attendance': [90, 45, 70, 95, 55],
    'PreviousScore': [85, 30, 65, 92, 40],
    'AssignmentsCompleted': [10, 2, 8, 10, 5],
    'SleepHours': [8, 5, 7, 9, 6]
}
df_new = pd.DataFrame(new_data)
predictions = model.predict(df_new) #
df_new['Predicted_Result'] = predictions
print("\nQ4: New Student Predictions:\n", df_new)

###############################################################################################
# Q5: Manual Accuracy Calculation
###############################################################################################

# Use: To verify the accuracy logic manually.
y_pred = model.predict(X_test)
correct = (y_test == y_pred).sum() #
total = len(y_test)
manual_acc = (correct / total) * 100
print(f"\nQ5: Manual Accuracy: {manual_acc:.2f}%")

################################################################################################
# Q6: Identify Misclassified Students
################################################################################################

# Use: To see where the model made wrong predictions.
misclassified = X_test[y_test != y_pred].copy() #
misclassified['Actual'] = y_test[y_test != y_pred]
misclassified['Predicted'] = y_pred[y_test != y_pred]
print("\nQ6: Misclassified Students:\n", misclassified)


#############################################################################################
# Q7: Random State Comparison 
#############################################################################################
print(Border)
print("Q7: Effect of Random State on Accuracy")
print(Border)

# Use: To observe how different data shuffling (random_state) changes the testing accuracy.
states = [0, 10, 42]   
for s in states:
    xt, xv, yt, yv = train_test_split(X, y, test_size=0.3, random_state=s)
    m = DecisionTreeClassifier(max_depth=3, random_state=42)
    m.fit(xt, yt)
    print(f"Accuracy with random_state={s}: {accuracy_score(yv, m.predict(xv))*100:.2f}%")


#############################################################################################
# Q8: Decision Tree Visualization 
#############################################################################################
# Use: To visualize the decision-making logic of the model. 
# The 'Root Node' is the feature that splits the data most effectively (Highest Information Gain).
plt.figure(figsize=(12,8))
plot_tree(model, feature_names=list(X.columns), class_names=['Fail', 'Pass'], filled=True)
plt.title("Decision Tree Visualization")
plt.show()
print("Q8: Graph created successfully")


#############################################################################################
# Q9: Create PerformanceIndex Column 
#############################################################################################
print(Border)
print("Q9: Adding PerformanceIndex Feature")
print(Border)

# Use: Feature Engineering to combine related features and check if accuracy improves.
df['PerformanceIndex'] = (df['StudyHours'] * 2) + df['Attendance']
print("New Column Added Successfully.")
print(df[['StudyHours', 'Attendance', 'PerformanceIndex']].head())


#############################################################################################
# Q10: Max Depth Analysis & Overfitting
#############################################################################################
print(Border)
print("Q10: Training Accuracy vs Testing Accuracy (max_depth=None)")
print(Border)

# Use: To identify Overfitting. If training is 100% but testing is low, the model has memorized the data.
overfit_model = DecisionTreeClassifier(max_depth=None, random_state=42)
overfit_model.fit(X_train, y_train)

train_acc = accuracy_score(y_train, overfit_model.predict(X_train))
test_acc = accuracy_score(y_test, overfit_model.predict(X_test))

print(f"Training Accuracy: {train_acc*100:.2f}%")
print(f"Testing Accuracy: {test_acc*100:.2f}%")

if train_acc == 1.0 and test_acc < train_acc:
    print("Observation: The model is OVERFITTING because it memorized the training data.")