# CAR-PREDICTION
To Determine factors that will influence a Customer to purchase a Car
Project Goal:

Build a prediction model to determine whether a customer will buy a car based on their gender, age, and salary.
Data:

car_data.csv containing information about customers and their purchase decisions.
Key Steps:

Loading Libraries:

Import NumPy, Pandas, Matplotlib, Seaborn, and scikit-learn modules for data manipulation, visualization, and model building.
Data Loading and Exploration:

Load the CSV file into a Pandas DataFrame.
Check for missing values and data types.
Get descriptive statistics using describe().
Data Preprocessing:

Handle missing values (if any).
Encode categorical features (gender) using LabelEncoder.
Explore correlations using a heatmap from Seaborn.
Data Splitting:

Divide the data into training and testing sets (70/30 split).
Feature Scaling:

Apply MinMaxScaler to normalize feature values.
Decision Tree Model:

Create a decision tree classifier with the 'gini' criterion.
Train the model on the scaled training set.
Evaluate its performance on the testing set using accuracy, confusion matrix, and classification report.
Explore cost-complexity pruning to potentially reduce overfitting.
Random Forest Model:

Create a Random Forest classifier.
Train and evaluate it similarly to the decision tree.
Findings:

Both models achieve good accuracy in predicting purchase decisions.
Random Forest often outperforms decision trees due to its ensemble nature.
Future Work:

Experiment with different hyperparameters for model tuning.
Explore other feature engineering techniques.
Test additional model types for potential improvements.
