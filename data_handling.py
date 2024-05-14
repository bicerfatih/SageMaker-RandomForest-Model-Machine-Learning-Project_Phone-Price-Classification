# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset from a CSV file
df = pd.read_csv("data/mobilephone_price.csv")

# Display the first few rows of the dataframe
print(df.head())

# Print the shape of the dataframe (number of rows and columns)
print(df.shape)

# Display the distribution of the target variable 'price_range' as a percentage
df['price_range'].value_counts(normalize=True)

# Print the names of the columns in the dataframe
print(df.columns)
# Calculate and print the percentage of missing values in each column
print(df.isnull().mean() * 100)

# Get a list of all column names
features = list(df.columns)
# Extract the label (target) column name and remove it from the features list
label = features.pop(-1)
print(label)

# Split the dataframe into feature matrix 'x' and target vector 'y'
x = df[features]
y = df[label]
# Display the first few rows of the feature matrix 'x'
print(x.head())
# Display the first few rows of the target vector 'y'
print(y.head())
# Display the distribution of the target variable
print(y.value_counts())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=0)

# Print the shapes of the training and testing sets
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# Create new dataframes for the training and testing sets with the target variable included
trainX = pd.DataFrame(X_train)
trainX[label] = y_train
testX = pd.DataFrame(X_test)
testX[label] = y_test

# Print the shapes of the new training and testing dataframes
print(trainX.shape)
print(testX.shape)
# Display the first few rows of the training dataframe
print(trainX.head())

# Check for missing values in the training and testing dataframes
trainX.isnull().sum()
testX.isnull().sum()

# Save the training and testing dataframes to CSV files
trainX.to_csv("data/train_v1.csv", index=False)
testX.to_csv("data/test_v1.csv", index=False)
