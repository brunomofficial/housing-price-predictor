import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

#Load the data
df = pd.read_csv("Housing data.csv")

#Clean and prepare
df = df[pd.to_numeric(df["price"], errors='coerce').notna()]
print(f"Total rows: {len(df)}")
print(f"Price range: ${df['price'].min():,.0f} to ${df['price'].max():,.0f}")
print(f"Sqft_living range: {df['sqft_living'].min():,.0f} to {df['sqft_living'].max():,.0f}")

#remove outliers
df = df[(df['price'] > 10000) & (df['price'] < 5000000)]  # Remove extreme outliers
df = df[(df['sqft_living'] > 300) & (df['sqft_living'] < 10000)]

print(f"\nAfter filtering: {len(df)} rows")

#train the model
X = df[['sqft_living']]
y = df[['price']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

#results
coef_value = model.coef_.item()
intercept_value = model.intercept_.item()

print(f"Price per sq ft of living : ${coef_value:.2f}")
print(f"Base price: ${intercept_value:,.2f}")
print(f"R² score: {r2_score(y_test, model.predict(X_test)):.4f}")

#make predictions
print("\nPredictions based on sq ft living size:")
for sqft in [1000, 1500, 2000, 2500, 3000]:
    #convert input to DataFrame with feature name to avoid warning
    pred_input = pd.DataFrame([[sqft]], columns = ['sqft_living'])
    pred_value = model.predict(pred_input).item()
    print(f"{sqft} sq ft : ${pred_value:,.2f}")

#quick scatter plot with sampling (10% of data)
plt.figure(figsize=(10, 6))
sample_df = df.sample(frac=0.1, random_state=42)  # Take 10% sample
plt.scatter(sample_df['sqft_living'], sample_df['price'], alpha=0.3)
plt.xlabel('Square Feet Living')
plt.ylabel('Price')
plt.title('Price vs Square Footage (10% sample)')
plt.show()



