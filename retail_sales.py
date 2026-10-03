import pandas as pd
import numpy as np
from scipy.stats import zscore

# Retail sales data
data = {
    "Day": range(1, 11),
    "Revenue": [45, 52, np.nan, 48, 390, 55, np.nan, 50, 47, 53]
}

df = pd.DataFrame(data)

# Report missing values
print("Missing values in Revenue:", df["Revenue"].isnull().sum())

# Fill missing values with median
df["Revenue"] = df["Revenue"].fillna(df["Revenue"].median())

print("\nAfter median fill:")
print(df)

# Min-Max normalization
prices = df["Revenue"]
df["Revenue_MinMax"] = (
    prices - prices.min()
) / (prices.max() - prices.min())

# Z-score standardization
df["Revenue_Zscore"] = zscore(df["Revenue"])

# IQR calculation
Q1 = df["Revenue"].quantile(0.25)
Q3 = df["Revenue"].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Outlier Boolean column
df["Outlier"] = (
    (df["Revenue"] < lower_bound)
    | (df["Revenue"] > upper_bound)
)

# Print IQR outliers
outliers = df[df["Outlier"]]

print("\nIQR bounds:", lower_bound, "-", upper_bound)

print("\nIQR outliers detected on day(s):")
print(outliers["Day"].tolist())

# Final table
print("\nFinal table:")
print(
    df[
        [
            "Day",
            "Revenue",
            "Revenue_MinMax",
            "Revenue_Zscore",
            "Outlier"
        ]
    ]
)
