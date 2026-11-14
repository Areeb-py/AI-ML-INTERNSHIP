import pandas as pd

df = pd.DataFrame({
    "Gender": ["Male", "Female", "Male"],
    "City": ["Lahore", "Karachi", "Islamabad"],
    "Age": [25, 30, 22]
})

encoded_df = pd.get_dummies(
    df,
    columns=["Gender", "City"],
    dtype=int
)

print(encoded_df)
