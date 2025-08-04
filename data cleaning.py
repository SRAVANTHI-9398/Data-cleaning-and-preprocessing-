

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler

# --- Step 1: Import the dataset and explore basic info ---
# Let's assume the dataset is a CSV file named 'titanic.csv'
# You'll need to have this file in the same directory as your script
# or provide the full path to the file.

try:
    df = pd.read_csv('titanic.csv')
    print("Dataset imported successfully.")
except FileNotFoundError:
    print("Error: 'titanic.csv' not found. Please download the dataset and place it in the correct directory.")
    # You can download the dataset from the link provided in the image or other sources.
    # For now, let's stop the execution if the file is not found.
    exit()

# Display the first few rows of the dataset
print("\n--- First 5 rows of the dataset ---")
print(df.head())

# Get basic information about the dataset (data types, non-null counts)
print("\n--- Dataset Info ---")
df.info()

# Get a summary of numerical columns
print("\n--- Descriptive Statistics ---")
print(df.describe())

# Check for null values in each column
print("\n--- Null Values per Column ---")
print(df.isnull().sum())

# --- Step 2: Handle missing values ---
# The Titanic dataset has missing values in 'Age', 'Cabin', and 'Embarked'.

# Handle missing 'Age' values by imputing with the median.
# Using the median is often better than the mean if the data has outliers.
median_age = df['Age'].median()
df['Age'].fillna(median_age, inplace=True)

# Handle missing 'Embarked' values by imputing with the mode (most frequent value).
mode_embarked = df['Embarked'].mode()[0]
df['Embarked'].fillna(mode_embarked, inplace=True)

# 'Cabin' has a large number of missing values. A common approach is to drop it
# if it's not useful for the analysis, or to create a new category for 'Missing'.
# For this example, let's create a new 'Cabin_Known' feature.
df['Cabin_Known'] = df['Cabin'].isnull().apply(lambda x: 0 if x else 1)
df.drop('Cabin', axis=1, inplace=True) # Now we can drop the original 'Cabin' column

print("\n--- Null Values after Handling Missing Values ---")
print(df.isnull().sum())

# --- Step 3: Convert categorical features into numerical using encoding ---
# 'Sex' and 'Embarked' are categorical features. We'll use LabelEncoder for 'Sex'
# and one-hot encoding for 'Embarked' (to avoid assuming an order).

# For 'Sex':
le = LabelEncoder()
df['Sex_encoded'] = le.fit_transform(df['Sex'])
df.drop('Sex', axis=1, inplace=True)

# For 'Embarked':
# pandas get_dummies is a simple way to perform one-hot encoding.
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True) # drop_first=True to avoid multicollinearity

# Check the new dataframe
print("\n--- DataFrame after Encoding ---")
print(df.head())

# --- Step 4: Normalize/Standardize the numerical features ---
# Standardizing features like 'Age' and 'Fare' can be important for algorithms
# that are sensitive to the scale of the input features (e.g., SVM, K-Means).

numerical_features = ['Age', 'Fare']
scaler = StandardScaler()

# Fit and transform the numerical features
df[numerical_features] = scaler.fit_transform(df[numerical_features])

print("\n--- DataFrame after Standardization ---")
print(df.head())

# --- Step 5: Visualize outliers using boxplots and remove them ---
# Let's visualize the outliers for 'Age' and 'Fare' before and after handling.
# Note: Since we standardized the data, the values will be centered around 0.

# Before (or using the original data)
# To keep this code runnable, let's use the standardized data for visualization.
# Outliers will be visible as points beyond the whiskers.

# Boxplot for 'Age'
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sns.boxplot(y=df['Age'])
plt.title('Boxplot of Age (Standardized)')

# Boxplot for 'Fare'
plt.subplot(1, 2, 2)
sns.boxplot(y=df['Fare'])
plt.title('Boxplot of Fare (Standardized)')
plt.tight_layout()
plt.show()

# --- Removing Outliers (Example using IQR) ---
# We can identify and remove outliers, though this should be done carefully
# as outliers can sometimes be valid data points.
# Let's demonstrate for 'Fare'. We'll find the IQR and define bounds.

Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Create a new DataFrame without the outliers
df_no_outliers = df[(df['Fare'] >= lower_bound) & (df['Fare'] <= upper_bound)]

print(f"\nOriginal number of rows: {len(df)}")
print(f"Number of rows after removing fare outliers: {len(df_no_outliers)}")

# You can now use `df_no_outliers` for your further machine learning tasks.