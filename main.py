import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_csv(
    "House Price Prediction Dataset - House Price Prediction Dataset.csv"
)

print("House Price Prediction System")
print("")

print("\nDataset loaded successfully!")
print("Number of records:", len(df))

print("\nFirst 5 records:")
print(df.head())

X = df[["Area", "Bedrooms", "Bathrooms", "Floors", "YearBuilt"]]

y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

model = LinearRegression()

model.fit(X_train, y_train)

print("\nLinear Regression model trained successfully!")

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("\nModel Evaluation")
print("")
print("R² Score:", r2)
print("Mean Squared Error:", mse)


results = pd.DataFrame({
    "Actual Price": y_test.values[:10],
    "Predicted Price": y_pred[:10]
})

print("\nActual vs Predicted Prices:")
print(results)

print("\nHouse Price Prediction")
print("")

area = float(input("Enter Area: "))
bedrooms = float(input("Enter Number of Bedrooms: "))
bathrooms = float(input("Enter Number of Bathrooms: "))
floors = float(input("Enter Number of Floors: "))
year_built = float(input("Enter Year Built: "))

new_house = pd.DataFrame({
    "Area": [area],
    "Bedrooms": [bedrooms],
    "Bathrooms": [bathrooms],
    "Floors": [floors],
    "YearBuilt": [year_built]
})

predicted_price = model.predict(new_house)[0]

print("\nPredicted House Price: ₹", round(predicted_price, 2))