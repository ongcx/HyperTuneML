## Overview:
This Python script demonstrates how to perform hyperparameter tuning for a Random Forest Classifier using Grid Search Cross-Validation.

### Instructions:
1. Installation:
- Ensure you have the necessary libraries installed:
```
pip install numpy scikit-learn
```
2. Usage:
- Import the required libraries and define the tune_hyperparameters function.
- Load your dataset (e.g., Iris dataset) or use your own data.
- Call the tune_hyperparameters function with your features X and target y arrays.

3. Functionality:
- The function splits the data, defines a parameter grid for tuning, and initializes a Random Forest Classifier.
- Grid Search CV is used to find the best hyperparameters based on the provided parameter grid.
- The function prints the best hyperparameters found and evaluates the model's accuracy on the test set.

### Example:
```
python
# Import necessary libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# Define a function for hyperparameter tuning
def tune_hyperparameters(X, y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define the parameter grid to search through
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    
    # Initialize the model
    model = RandomForestClassifier()
    
    # Perform grid search to find the best hyperparameters
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    
    # Print the best hyperparameters found
    print("Best hyperparameters: ", grid_search.best_params_)
    
    # Evaluate the model on the test set
    accuracy = grid_search.score(X_test, y_test)
    print("Accuracy on test set: ", accuracy)

# Example usage
if __name__ == "__main__":
    # Load sample dataset (Iris dataset)
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # Call the hyperparameter tuning function
    tune_hyperparameters(X, y)
```

### Note:
- Customize the parameter grid and dataset according to your specific requirements.
- Experiment with different datasets and parameters to optimize model performance.
