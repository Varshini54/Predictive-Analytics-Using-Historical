# ==========================
# Predictive Analytics Using Historical Data
# Linear Regression Project
# ==========================

# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# --------------------------
# Step 1: Load Dataset
# --------------------------
df = pd.read_excel("/content/Historical_Sales.xlsx")

# Display first 5 rows
print("Dataset Preview:")
print(df.head())

# --------------------------
# Step 2: Convert Date
# --------------------------
df["Date"] = pd.to_datetime(df["Date"])

# Convert dates into numbers
df["Day_Number"] = (df["Date"] - df["Date"].min()).dt.days

# --------------------------
# Step 3: Select Features
# --------------------------
X = df[["Day_Number"]]
y = df["Sales"]

# --------------------------
# Step 4: Train Model
# --------------------------
model = LinearRegression()
model.fit(X, y)

# --------------------------
# Step 5: Predict Sales
# --------------------------
df["Predicted_Sales"] = model.predict(X)

# --------------------------
# Step 6: Model Evaluation
# --------------------------
mse = mean_squared_error(y, df["Predicted_Sales"])
r2 = r2_score(y, df["Predicted_Sales"])

print("\nModel Evaluation")
print("----------------")
print("Mean Squared Error :", round(mse,2))
print("R2 Score :", round(r2,4))

# --------------------------
# Step 7: Save Prediction File
# --------------------------
df.to_excel("Historical_Sales_Predictions.xlsx", index=False)

# --------------------------
# Step 8: Sales Trend Graph
# --------------------------
plt.figure(figsize=(10,5))
plt.plot(df["Date"], df["Sales"], marker='o')
plt.title("Historical Sales Trend")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.grid(True)

plt.savefig("sales_trend.png")
plt.show()

# --------------------------
# Step 9: Actual vs Predicted
# --------------------------
plt.figure(figsize=(10,5))

plt.plot(df["Date"], df["Sales"], label="Actual Sales", linewidth=2)

plt.plot(df["Date"],
         df["Predicted_Sales"],
         linestyle="--",
         label="Predicted Sales",
         linewidth=2)

plt.title("Actual vs Predicted Sales")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)

plt.savefig("actual_vs_predicted.png")
plt.show()

# --------------------------
# Step 10: Predict Future 10 Days
# --------------------------
future_days = pd.DataFrame({
    "Day_Number": range(df["Day_Number"].max()+1,
                        df["Day_Number"].max()+11)
})

future_predictions = model.predict(future_days)

future_dates = pd.date_range(
    start=df["Date"].max()+pd.Timedelta(days=1),
    periods=10
)

future_df = pd.DataFrame({
    "Future_Date": future_dates,
    "Predicted_Sales": future_predictions
})

print("\nFuture Sales Prediction")
print("-----------------------")
print(future_df)

future_df.to_excel("Future_Sales_Prediction.xlsx", index=False)

print("\nProject Completed Successfully!")
print("Generated Files:")
print("1. Historical_Sales_Predictions.xlsx")
print("2. Future_Sales_Prediction.xlsx")
print("3. sales_trend.png")
print("4. actual_vs_predicted.png")
