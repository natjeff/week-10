# Exercise 1:

import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Load the data
url = "https://raw.githubusercontent.com/leontoddjohnson/datasets/refs/heads/main/data/coffee_analysis.csv"
data = pd.read_csv(url)

# Prepare the model
X = data[['100g_USD']]   # predictor
y = data['rating']       # response

# Train the model
model = LinearRegression()
model.fit(X, y)

# Save the trained model
with open("model_1.pickle", "wb") as f:
    pickle.dump(model, f)

# Print confirmation message
print("Model training complete. File 'model_1.pickle' created successfully.")




# Exercise 2:

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import pickle

# Load the data
url = "https://raw.githubusercontent.com/leontoddjohnson/datasets/refs/heads/main/data/coffee_analysis.csv"
data = pd.read_csv(url)

# Define the roast_category function
def roast_category(roast):
    """Map roast names to numeric categories."""
    mapping = {
        'Light': 0,
        'Medium-Light': 1,
        'Medium': 2,
        'Medium-Dark': 3,
        'Dark': 4
    }
    return mapping.get(roast, np.nan)

# Apply the function
data['roast_cat'] = data['roast'].apply(roast_category)

# Prepare the model
X = data[['100g_USD', 'roast_cat']]
y = data['rating']
X = X.dropna()
y = y.loc[X.index]

# Decision Tree 
dtr = DecisionTreeRegressor(random_state=42)
dtr.fit(X, y)

# Save the trained model
with open("model_2.pickle", "wb") as f:
    pickle.dump(dtr, f)
print("Model_2 training complete. File 'model_2.pickle' saved successfully.")

# Test the model
df_X = pd.DataFrame([
    [10.00, 1],
    [15.00, 3],
    [8.50, np.nan]
], columns=["100g_USD", "roast_cat"])

y_pred = dtr.predict(df_X.values)
print("Predictions for sample input:")
print(y_pred)