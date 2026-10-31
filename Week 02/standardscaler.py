import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.DataFrame({
    "Age": [22, 35, 45, 28, 40],
    "Salary": [30000, 80000, 150000, 45000, 100000]
})

print("Original Data:")
print(df)

scaler = StandardScaler()

scaled_data = scaler.fit_transform(df)

scaled_df = pd.DataFrame(
    scaled_data,
    columns=df.columns
)

print("\nScaled Data:")
print(scaled_df)

print("\nMeans:")
print(scaled_df.mean())

print("\nStandard Deviations:")
print(scaled_df.std(ddof=0))