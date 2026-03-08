import math


# ----------------------------
# DATASET for  1 & 2
# ----------------------------
dataset_1 = [
    {"point": "A", "x": 1, "y": 2, "label": "Red"},
    {"point": "B", "x": 2, "y": 3, "label": "Red"},
    {"point": "C", "x": 3, "y": 1, "label": "Blue"},
    {"point": "D", "x": 6, "y": 5, "label": "Blue"},
]

# ----------------------------
# DATASET for 3
# ----------------------------
dataset_3 = [
    {"study_hours": 2, "attendance": 60, "result": "Fail"},
    {"study_hours": 5, "attendance": 80, "result": "Pass"},
    {"study_hours": 6, "attendance": 85, "result": "Pass"},
    {"study_hours": 1, "attendance": 50, "result": "Fail"},
]


# ============================================================
# FUNCTION 1: Euclidean Distance
# Formula: sqrt((x2-x1)^2 + (y2-y1)^2)
# Use:  To calculate  distance between  Points
# ============================================================
def euclidean_distance(a1, b1, a2, b2):
    return round(math.sqrt((a2 - a1) ** 2 + (b2 - b1) ** 2), 2)


# ============================================================
# FUNCTION 2: KNN Predict
# Use: Dataset predict new point to labels name
# Steps: Distance -> Sort -> K select -> Majority vote
# ============================================================
def knn_predict(new_x, new_y, k):
    distances = []

    # Step 1: Euclidean distance calculate each point
    for data in dataset_1:
        dist = euclidean_distance(new_x, new_y, data["x"], data["y"])
        distances.append(
            {"point": data["point"], "distance": dist, "label": data["label"]}
        )

    # Step 2: Distance sort (small first)
    distances.sort(key=lambda d: d["distance"])

    # Step 3: select K nearest neighbors
    k_nearest = distances[:k]

    # Step 4: Majority voting - which label is more majority
    votes = {}
    for neighbor in k_nearest:
        label = neighbor["label"]
        votes[label] = votes.get(label, 0) + 1

    # more majority label predicted class label
    predicted_class = max(votes, key=votes.get)
    return k_nearest, predicted_class


# ============================================================
# FUNCTION 3: KNN for Pass/Fail
# Use: Study Hours & Attendance ne Pass/Fail predict
# ============================================================
def knn_predict_pass_fail(new_sh, new_att, k=3):
    distances = []

    for data in dataset_3:
        dist = euclidean_distance(
            new_sh, new_att, data["study_hours"], data["attendance"]
        )
        distances.append({"distance": dist, "result": data["result"]})

    distances.sort(key=lambda d: d["distance"])
    k_nearest = distances[:k]

    votes = {}
    for neighbor in k_nearest:
        label = neighbor["result"]
        votes[label] = votes.get(label, 0) + 1

    return max(votes, key=votes.get)


# ============================================================
# Step 1 - KNN with K=3, show nearest neighbors
# ============================================================
def Marvellous1(new_x, new_y):
    print("\n" + "=" * 45)
    print(" Step 1 - KNN Classifier (K=3)")
    print("=" * 45)

    k_nearest, predicted = knn_predict(new_x, new_y, k=3)

    print("\nNearest Neighbors:")
    for n in k_nearest:
        print(f"  {n['point']} - Distance: {n['distance']}")

    print(f"\nPredicted Class: {predicted}")
    print("=" * 45)


# ============================================================
# Step 2 - Show how prediction changes with K
# ============================================================
def Marvellous2(new_x, new_y):
    print("\n" + "=" * 45)
    print(" Step 2 - Effect of K on Prediction")
    print("=" * 45)

    print("\nPrediction Results:")
    print("-" * 30)

    for k in [1, 3, 5]:
        # K=5 but dataset present  only 4 points , when use min
        actual_k = min(k, len(dataset_1))
        _, predicted = knn_predict(new_x, new_y, k=actual_k)
        print(f"  K = {k}  -->  {predicted}")

    print("\nExplanation (Why result changes when K increases):")
    print("  K=1 : Only 1 closest point considered-> Overfitting possible")
    print("  K=3 : 3 nearest points vote  -> Balanced result")
    print("  K=5 : More points consider -> Blue becomes dominant ")
    print("=" * 45)


# ============================================================
# Step 3 - Student Pass/Fail Prediction
# ============================================================
def Marvellous3():
    print("\n" + "=" * 45)
    print(" Step 3 - Student Pass/Fail KNN")
    print("=" * 45)

    study_hours = float(input("Enter Study Hours: "))
    attendance = float(input("Enter Attendance: "))

    result = knn_predict_pass_fail(study_hours, attendance, k=3)

    print(f"\nPredicted Result: {result}")
    print("=" * 45)


def main():
    print("\n" + "*" * 45)
    print(" KNN complete successfully ")
    print("*" * 45)

    new_x = float(input("Enter X coordinate: "))
    new_y = float(input("Enter Y coordinate: "))

    Marvellous1(new_x, new_y)
    Marvellous2(new_x, new_y)
    Marvellous3()

    print("\n------ All Assignments Completed------\n")


if __name__ == "__main__":
    main()
