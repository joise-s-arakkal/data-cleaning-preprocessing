import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder


# 1. Import the dataset and explore basic info (nulls, data types)
# Load dataset
df = pd.read_csv("Titanic-Dataset.csv")

# Explore the dataset
print(df.head())
print(df.info())   
print(df.describe())
print(df.isnull().sum())


# 2. Handle missing values using mean/median/imputation.
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df.drop(columns=['Cabin'], inplace=True)


# 3 Convert categorical features into numerical using encoding.
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])  # male=1, female=0
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)


# 4 Normalize/standardize the numerical features.
scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])


# 5. Visualize outliers using boxplots and remove them.

plt.figure(figsize=(10, 5))
sns.boxplot(data=df[['Age', 'Fare']])
plt.title("Boxplot of Age and Fare after Standardization")
plt.show()

# 5. Visualize outliers using boxplots and remove them. (Remove outliers using IQR)
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    return df[~((df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR)))]

df = remove_outliers(df, 'Age')
df = remove_outliers(df, 'Fare')

# Save the DataFrame to a CSV file
df.to_csv('cleaned_data.csv', index=False)