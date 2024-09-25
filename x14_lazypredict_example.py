# pip install pandas scikit-learn lazypredict
# 
import pandas as pd
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyClassifier

# Load your dataset
data = pd.read_csv('your_dataset.csv')

# Split the data into features and target
X = data.drop('target_column', axis=1)
y = data['target_column']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize LazyClassifier
clf = LazyClassifier()

# Fit the models and get results
models = clf.fit(X_train, X_test, y_train, y_test)

# Display results
print(models)
