import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
import warnings
import joblib

warnings.filterwarnings('ignore')

# Load and merge the data
print("Loading and merging data...")
inf = pd.read_csv('PCOS_infertility.csv')
woinf = pd.read_excel('PCOS_data_without_infertility.xlsx', sheet_name='Full_new')
data = pd.merge(woinf, inf, on='Patient File No.', suffixes=('', '_wo'), how='left')

# Data preprocessing
print("Preprocessing data...")
data = data.drop(['Unnamed: 44', 'Sl. No_wo', 'PCOS (Y/N)_wo', '  I   beta-HCG(mIU/mL)_wo', 'II    beta-HCG(mIU/mL)_wo', 'AMH(ng/mL)_wo', 'Sl. No', 'Patient File No.'], axis=1)
data = data.rename(columns={"PCOS (Y/N)": "Target"})
data.columns = [col.strip() for col in data.columns]

# Handle missing values
numeric_columns = data.select_dtypes(include=[np.number]).columns
categorical_columns = data.select_dtypes(exclude=[np.number]).columns

numeric_imputer = SimpleImputer(strategy='median')
categorical_imputer = SimpleImputer(strategy='most_frequent')

data[numeric_columns] = numeric_imputer.fit_transform(data[numeric_columns])
data[categorical_columns] = categorical_imputer.fit_transform(data[categorical_columns])

# Encode categorical variables
data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

# Separate features and target
X = data.drop('Target', axis=1)
y = data['Target']

# Feature selection using f_classif (ANOVA F-value)
print("Performing feature selection...")
selector = SelectKBest(f_classif, k=15)
X_new = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()].tolist()

# Add endometrium as the 16th feature
selected_features.append('Endometrium (mm)')

print("Selected features:")
for feature in selected_features:
    print(feature)

# Visualizations
print("Creating visualizations...")
plt.figure(figsize=(12, 10))
sns.heatmap(data[selected_features + ['Target']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Selected Features')
plt.tight_layout()
plt.show()

for feature in selected_features:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Target', y=feature, data=data)
    plt.title(f'Distribution of {feature} by Target')
    plt.show()

# Prepare data for modeling
X = data[selected_features]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')
print("Scaler saved as 'scaler.pkl'")

# Define models and parameters for grid search
models = {
    'Logistic Regression': (LogisticRegression(), {'C': [0.1, 1, 10], 'penalty': ['l1', 'l2']}),
    'SVM': (SVC(probability=True), {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']}),
    'Random Forest': (RandomForestClassifier(), {'n_estimators': [100, 200], 'max_depth': [5, 10]}),
    'KNN': (KNeighborsClassifier(), {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}),
    'Decision Tree': (DecisionTreeClassifier(), {'max_depth': [5, 10, 15], 'min_samples_split': [2, 5, 10]}),
    'XGBoost': (XGBClassifier(), {'n_estimators': [100, 200], 'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.1]}),
    'CatBoost': (CatBoostClassifier(verbose=0), {'iterations': [100, 200], 'depth': [4, 6, 8], 'learning_rate': [0.01, 0.1]})
}

# Train and tune individual models
print("Training and tuning individual models...")
best_models = {}
for name, (model, params) in models.items():
    print(f"Training {name}...")
    grid_search = GridSearchCV(model, params, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)
    best_models[name] = grid_search.best_estimator_
    y_pred = grid_search.predict(X_test_scaled)
    print(f"{name} Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Best parameters: {grid_search.best_params_}")
    print(classification_report(y_test, y_pred))
    print("\n")

# Stacking Classifier
print("Training Stacking Classifier...")
base_models = list(best_models.values())[:-1]
final_model = list(best_models.values())[-1]
stacking_clf = StackingClassifier(estimators=[(f"model_{i}", model) for i, model in enumerate(base_models)], final_estimator=final_model, cv=5)
stacking_clf.fit(X_train_scaled, y_train)
stacking_pred = stacking_clf.predict(X_test_scaled)
print(f"Stacking Classifier Accuracy: {accuracy_score(y_test, stacking_pred):.4f}")
print(classification_report(y_test, stacking_pred))

# Blending
print("Training Blending Ensemble...")
class BlendingEnsemble:
    def __init__(self, models, meta_model):
        self.models = models
        self.meta_model = meta_model

    def fit(self, X, y):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5, random_state=42)
        self.base_predictions = np.column_stack([
            model.fit(X_train, y_train).predict_proba(X_val)[:, 1] for model in self.models
        ])
        self.meta_model.fit(self.base_predictions, y_val)

    def predict(self, X):
        base_predictions = np.column_stack([
            model.predict_proba(X)[:, 1] for model in self.models
        ])
        return self.meta_model.predict(base_predictions)

    def predict_proba(self, X):
        base_predictions = np.column_stack([
            model.predict_proba(X)[:, 1] for model in self.models
        ])
        return self.meta_model.predict_proba(base_predictions)

blending_ensemble = BlendingEnsemble(list(best_models.values())[:-1], list(best_models.values())[-1])
blending_ensemble.fit(X_train_scaled, y_train)
blending_pred = blending_ensemble.predict(X_test_scaled)
print(f"Blending Ensemble Accuracy: {accuracy_score(y_test, blending_pred):.4f}")
print(classification_report(y_test, blending_pred))

# Voting Classifier
print("Training Voting Classifier...")
voting_clf = VotingClassifier(
    estimators=[(name, model) for name, model in best_models.items()],
    voting='soft'
)
voting_clf.fit(X_train_scaled, y_train)
voting_pred = voting_clf.predict(X_test_scaled)
print(f"Voting Classifier Accuracy: {accuracy_score(y_test, voting_pred):.4f}")
print(classification_report(y_test, voting_pred))

# Visualize ensemble results
ensemble_methods = ['Stacking', 'Blending', 'Voting']
ensemble_accuracies = [
    accuracy_score(y_test, stacking_pred),
    accuracy_score(y_test, blending_pred),
    accuracy_score(y_test, voting_pred)
]

plt.figure(figsize=(10, 6))
sns.barplot(x=ensemble_methods, y=ensemble_accuracies)
plt.title('Comparison of Ensemble Methods')
plt.ylabel('Accuracy')
plt.show()

# Function to make predictions using the best ensemble method
def predict_pcos(input_data, ensemble_model):
    if input_data.ndim == 1:
        input_data = input_data.reshape(1, -1)
    scaled_input = scaler.transform(input_data)
    prediction = ensemble_model.predict(scaled_input)
    probability = ensemble_model.predict_proba(scaled_input)[:, 1]
    return prediction[0], probability[0]

# Choose the best ensemble method
best_ensemble = max([stacking_clf, blending_ensemble, voting_clf], 
                    key=lambda model: accuracy_score(y_test, model.predict(X_test_scaled)))

# Example prediction
sample_input = X.iloc[0].values
prediction, probability = predict_pcos(sample_input, best_ensemble)
print(f"\nSample Prediction:")
print(f"PCOS Prediction: {'Positive' if prediction == 1 else 'Negative'}")
print(f"Probability of PCOS: {probability:.2f}")

# Save the best model
joblib.dump(best_ensemble, 'best_pcos_model4.pkl')
print("Best model saved as 'best_pcos_model4.pkl'")