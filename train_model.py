import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle

# Load cleaned car data
df = pd.read_csv('Cleaned_Car_data.csv')

# Split features and target
X = df[['name', 'company', 'year', 'kms_driven', 'fuel_type']]
y = df['Price']

# Create column transformer (encode categorical columns)
ohe = OneHotEncoder(handle_unknown='ignore')
column_trans = ColumnTransformer([
    ('ohe', ohe, ['name', 'company', 'fuel_type'])
], remainder='passthrough')

# Create pipeline (encoding + model)
pipeline = Pipeline(steps=[
    ('transformer', column_trans),
    ('model', LinearRegression())
])

# Train the model
pipeline.fit(X, y)

# Save model to file
with open('LinearRegressionModel.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

print("✅ Model trained and saved as LinearRegressionModel.pkl")
