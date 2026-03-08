import matplotlib.pyplot as plt


X = [1, 2, 3, 4, 5]  # X = independent variable (input values)
Y = [3, 4, 2, 4, 5]  # Y = dependent variable (output values)

#  Experience vs Salary
exp_data = [1, 2, 3, 4, 5]  # Years of experience
salary_data = [20000, 25000, 30000, 35000, 40000]  # Salary in rupees


# ============================================================
# HELPER FUNCTION 1: find_average(values)
# Purpose : Calculate mean of a list
# Formula : X̄ = (x1 + x2 + ... + xn) / n
# ============================================================
def find_average(values):
    total = sum(values)  # Add all values:  x1 + x2 + ... + xn
    n = len(values)  # Count of values: n
    return total / n  # Return X̄ = total / n


# ============================================================
# HELPER FUNCTION 2: get_slope(x_vals, y_vals, mx, my)
# Purpose : Calculate slope of the best fit line
# Formula :      Σ (xi - X̄)(yi - Ȳ)
#           m =  ──────────────────
#                  Σ (xi - X̄)²
# ============================================================
def get_slope(x_vals, y_vals, mx, my):
    top = 0  # Numerator:   Σ (xi - X̄)(yi - Ȳ)
    bottom = 0  # Denominator: Σ (xi - X̄)²

    for i in range(len(x_vals)):  # Loop through each data point
        dx = x_vals[i] - mx  # (xi - X̄) : x distance from mean
        dy = y_vals[i] - my  # (yi - Ȳ) : y distance from mean
        top += dx * dy  # Add (xi - X̄)(yi - Ȳ) to numerator
        bottom += dx**2  # Add (xi - X̄)² to denominator

    return round(top / bottom, 2)  # m = numerator / denominator


# ============================================================
# HELPER FUNCTION 3: get_intercept(mx, my, m)
# Purpose : Calculate where the line crosses Y axis
# Formula : c = Ȳ - m × X̄
# ============================================================
def get_intercept(mx, my, m):
    return round(my - (m * mx), 2)  # c = Ȳ - m × X̄


# ============================================================
# HELPER FUNCTION 4: get_prediction(x, m, c)
# Purpose : Predict Y for a given X using the line equation
# Formula : Ŷ = m × X + c
# ============================================================
def get_prediction(x, m, c):
    return round((m * x) + c, 2)  # Ŷ = m × X + c


# ============================================================
# HELPER FUNCTION 5: get_mse(actual, predicted)
# Purpose : Measure average prediction error
# Formula :       Σ (yi - ŷi)²
#           MSE = ────────────
#                      n
# ============================================================
def get_mse(actual, predicted):
    n = len(actual)  # Total number of values: n
    total = 0  # Will store: Σ (yi - ŷi)²

    for i in range(n):
        error = actual[i] - predicted[i]  # (yi - ŷi) : actual minus predicted
        total += error**2  # Add (yi - ŷi)² to total

    return round(total / n, 4)  # MSE = Σ(yi - ŷi)² / n


# ============================================================
# HELPER FUNCTION 6: get_r2(actual, predicted)
# Purpose : Check how well the line fits the data (0 to 1)
# Formula :         Σ (yi - ŷi)²      SS_res
#           R² = 1 - ──────────── = 1 - ──────
#                     Σ (yi - Ȳ)²      SS_tot
# ============================================================
def get_r2(actual, predicted):
    mean_y = find_average(actual)  # Ȳ = mean of actual Y values
    ss_res = 0  # Residual SS: Σ (yi - ŷi)²
    ss_tot = 0  # Total SS:    Σ (yi - Ȳ)²

    for i in range(len(actual)):
        ss_res += (actual[i] - predicted[i]) ** 2  # (yi - ŷi)² added to SS_res
        ss_tot += (actual[i] - mean_y) ** 2  # (yi - Ȳ)²  added to SS_tot

    return round(1 - (ss_res / ss_tot), 4)  # R² = 1 - SS_res / SS_tot


# ============================================================
# QUESTION 1 : Simple Linear Regression - Manual Calculation
# ============================================================
def solve_q1():
    print("\n" + "=" * 52)
    print("  Q1 - Simple Linear Regression (Manual)")
    print("=" * 52)

    mx = find_average(X)  # X̄ = mean of X
    my = find_average(Y)  # Ȳ = mean of Y
    m = get_slope(X, Y, mx, my)  # m = slope
    c = get_intercept(mx, my, m)  # c = intercept

    print(f"  Mean of X        : {mx}")
    print(f"  Mean of Y        : {my}")
    print(f"  Slope      (m)   : {m}")
    print(f"  Intercept  (c)   : {c}")
    print(f"  Regression Line  : Y = {m}X + {c}")
    print(f"  Predicted Y (X=6): {get_prediction(6, m, c)}")  # Ŷ = m×6 + c
    print("=" * 52)

    return m, c  # Return slope and intercept for Q2


# ============================================================
# QUESTION 2 : Model Performance - MSE and R² Calculation
# ============================================================
def solve_q2(m, c):
    print("\n" + "=" * 52)
    print("  Q2 - Model Performance (MSE & R² Score)")
    print("=" * 52)

    # Predict Y for all X values using Ŷ = m × Xi + c
    y_pred = [get_prediction(xi, m, c) for xi in X]

    # Print intermediate calculations table
    print(
        f"\n  {'X':>3} | {'Actual Y':>8} | {'Pred Y':>7} | {'Error':>6} | {'Error²':>8}"
    )
    print("  " + "-" * 44)
    for i in range(len(X)):
        err = round(Y[i] - y_pred[i], 2)  # Error = Actual - Predicted
        err_sq = round(err**2, 4)  # Error² = error squared
        print(f"  {X[i]:>3} | {Y[i]:>8} | {y_pred[i]:>7} | {err:>6} | {err_sq:>8}")

    mse = get_mse(Y, y_pred)  # MSE = Σ(yi - ŷi)² / n
    r2 = get_r2(Y, y_pred)  # R²  = 1 - SS_res / SS_tot

    print(f"\n  MSE (Mean Squared Error) : {mse}  → Formula: Σ(yi-ŷi)² / n")
    print(f"  R² Score                 : {r2}  → Formula: 1 - SS_res / SS_tot")
    print(f"  R² closer to 1 = better model fit")
    print("=" * 52)


# ============================================================
# QUESTION 3 : Salary Prediction + Regression Graph
# ============================================================
def solve_q3():
    print("\n" + "=" * 50)
    print("  Q3 - Salary Prediction + Graph")
    print("=" * 50)

    mx = find_average(exp_data)  # X̄ = mean of experience
    my = find_average(salary_data)  # Ȳ = mean of salary
    m = get_slope(exp_data, salary_data, mx, my)  # m = slope
    c = get_intercept(mx, my, m)  # c = intercept

    predicted_salary = get_prediction(6, m, c)  # Ŷ = m × 6 + c

    print(f"  Slope (m)     : {m}")
    print(f"  Intercept (c) : {c}")
    print(f"  Equation      : Salary = {m} × Experience + {c}")
    print(f"  Predicted Salary for 6 Years Experience : ₹{predicted_salary}")

    # Generate regression line Y values for all X in dataset
    reg_line = [get_prediction(x, m, c) for x in exp_data]  # Ŷi = m × xi + c

    # Plot graph
    plt.figure(figsize=(8, 5))  # Set figure size
    plt.scatter(
        exp_data,
        salary_data,
        color="blue",  # Blue dots = actual data
        label="Actual Data",
        zorder=5,
    )
    plt.plot(
        exp_data,
        reg_line,
        color="red",  # Red line = regression line
        label="Regression Line",
    )
    plt.scatter(
        [6],
        [predicted_salary],
        color="green",  # Green star = X=6 prediction
        marker="*",
        s=250,
        label=f"X=6 → ₹{predicted_salary}",
    )
    plt.title("Linear Regression - Experience vs Salary")  # Graph title
    plt.xlabel("Years of Experience")  # X axis label
    plt.ylabel("Salary (₹)")  # Y axis label
    plt.legend()  # Show legend box
    plt.grid(True)  # Show background grid
    plt.tight_layout()  # Auto adjust spacing
    plt.savefig("salary_graph.png")  # Save graph as PNG file
    plt.show()  # Display the graph
    print("  Graph saved as 'salary_graph.png'")
    print("=" * 50)


# ============================================================
# QUESTION 4 : Why is KNN called a lazy learner?
# ============================================================
def solve_q4():
    print(
        "\n  Q4 → KNN stores the entire dataset and does NO training. All calculation happens only at prediction time — that is why it is called a lazy learner."
    )


# ============================================================
# QUESTION 5 : What happens if K is too small?
# ============================================================
def solve_q5():
    print(
        "\n  Q5 → K too small (e.g. K=1) means only 1 neighbor decides — model overfits to noise and gives wrong results on new data."
    )


# ============================================================
# QUESTION 6 : What happens if K is too large?
# ============================================================
def solve_q6():
    print(
        "\n  Q6 → K too large means too many distant neighbors vote — model underfits, becomes too simple, and loses accuracy."
    )


# ============================================================
# QUESTION 7 : Why does linear regression minimize squared error?
# ============================================================
def solve_q7():
    print(
        "\n  Q7 → Squaring removes negatives so errors don't cancel, penalizes large errors more, and makes the formula mathematically easy to minimize. Formula: MSE = (1/n) Σ(Y - Yp)² "
    )


# ============================================================
# QUESTION 8 : Difference between MSE and R²?
# ============================================================
def solve_q8():
    print(
        "\n  Q8 → MSE measures average error size (lower = better) using (1/n) Σ(Y - Yp)². R² measures how well line fits data (1 = perfect) using 1 - SS_res/SS_tot. MSE has units, R² is unitless."
    )


# ============================================================
# QUESTION 9 : Why R² cannot be greater than 1?
# ============================================================
def solve_q9():
    print(
        "\n  Q9 → R² = 1 - (SS_res / SS_tot). Since SS_res >= 0 always, the fraction >= 0, so R² <= 1 always. It can be negative if the model is worse than a flat horizontal line."
    )


# ============================================================
# QUESTION 10 : Can KNN be used for regression?
# ============================================================
def solve_q10():
    print(
        "\n  Q10 → Yes. For KNN Regression, instead of majority voting, we take the AVERAGE of K nearest neighbors' Y values as the predicted output."
    )


# ============================================================
# MAIN FUNCTION - Runs all questions in order
# ============================================================
def main():
    print("\n" + "*" * 50)
    print("   LINEAR REGRESSION - Complete Assignment")
    print("*" * 50)

    m, c = solve_q1()  # Q1: Mean, Slope, Intercept, Equation
    solve_q2(m, c)  # Q2: MSE, R² with intermediate table
    solve_q3()  # Q3: Salary prediction + matplotlib graph
    solve_q4()  # Q4: Lazy learner explanation
    solve_q5()  # Q5: K too small
    solve_q6()  # Q6: K too large
    solve_q7()  # Q7: Why squared error
    solve_q8()  # Q8: MSE vs R²
    solve_q9()  # Q9: R² max value
    solve_q10()  # Q10: KNN for regression

    print("\n" + "*" * 50)
    print("  Completed Successfully ")
    print("*" * 50)


if __name__ == "__main__":
    main()
