import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split

# Define the BlendingEnsemble class
class BlendingEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self, models, blending_weights=None):
        self.models = models
        self.blending_weights = blending_weights if blending_weights else [1.0 / len(models)] * len(models)

    def fit(self, X, y):
        # Split the data into training and blending sets
        X_train, X_blend, y_train, y_blend = train_test_split(X, y, test_size=0.5, random_state=42)
        
        # Train base models
        for model in self.models:
            model.fit(X_train, y_train)
        
        # Generate blending features
        blend_features = np.column_stack([model.predict_proba(X_blend)[:, 1] for model in self.models])
        
        # Fit blending weights
        self.blending_weights = self._optimize_weights(blend_features, y_blend)
        
        return self

    def _optimize_weights(self, blend_features, y_blend):
        from scipy.optimize import minimize
        
        def loss(weights):
            blended_predictions = np.dot(blend_features, weights)
            return -np.mean(y_blend * np.log(blended_predictions) + (1 - y_blend) * np.log(1 - blended_predictions))
        
        initial_weights = np.ones(len(self.models)) / len(self.models)
        bounds = [(0, 1)] * len(self.models)
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        
        result = minimize(loss, initial_weights, bounds=bounds, constraints=constraints)
        return result.x

    def predict(self, X):
        blended_predictions = self.predict_proba(X)[:, 1]
        return (blended_predictions >= 0.5).astype(int)

    def predict_proba(self, X):
        blend_features = np.column_stack([model.predict_proba(X)[:, 1] for model in self.models])
        blended_probabilities = np.dot(blend_features, self.blending_weights)
        return np.column_stack([1 - blended_probabilities, blended_probabilities])

app = Flask(__name__)

# Load the pre-trained model and scaler
model = joblib.load('best_pcos_model4.pkl')
scaler = joblib.load('scaler.pkl')  # Make sure you have saved the scaler during model training

# Define the feature names in the correct order
feature_names = [
    'How old are You?(in years)', 'What is Your Weight?(In Kg)', 'BMI', 'Irregularity/Regularity in Period Cycle(4/2)', 'Cycle length(days)',
    'Hip(inch)', 'Waist(inch)', 'Have you recently gained Weight?(Y/N)', 'Have you noticed unusual hair growth on your face or any other parts like lower abdomen or back(Y/N)',
    'Did You Observe dark patches around your neck,armpits or chest?(Y/N)', 'Have you noticed excessive Hair lossor bald patches recently(Y/N)', 'Are you getting pimples on your face,forehead,jaw,upper back or upper arms?(Y/N)', 'Do you eat fast food recently,atleast twice a week(Y/N)',
    'Follicle Number left ', 'Follicle Number Right', 'Endometrium (mm)'
]

@app.route('/')
def home():
    return render_template('index.html', features=feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collecting form values
        input_features = {}
        for feature in feature_names:
            value = request.form.get(feature)
            if value is None:
                raise ValueError(f"Missing value for feature: {feature}")
            if feature.endswith('(Y/N)'):
                input_features[feature] = 1 if value.lower() in ['yes', 'y', '1'] else 0
            else:
                input_features[feature] = float(value)

        # Convert input data to numpy array
        input_array = np.array([input_features[feature] for feature in feature_names]).reshape(1, -1)
        
        # Apply scaling to the input data
        input_array_scaled = scaler.transform(input_array)
        
        # Make prediction
        prediction = model.predict(input_array_scaled)
        probability = model.predict_proba(input_array_scaled)[0][1]
        
        output = "Positive" if prediction[0] == 1 else "Negative"
        
        return render_template('after.html', 
                               prediction_text=f"PCOS Prediction: {output}",
                               probability_text=f"Probability of PCOS: {probability:.2f}")
    
    except Exception as e:
        return f"An error occurred: {str(e)}", 400

@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        data = request.get_json(force=True)
        input_features = {}
        for feature in feature_names:
            if feature not in data:
                raise ValueError(f"Missing feature: {feature}")
            if feature.endswith('(Y/N)'):
                input_features[feature] = 1 if data[feature].lower() in ['yes', 'y', '1'] else 0
            else:
                input_features[feature] = float(data[feature])
        
        input_array = np.array([input_features[feature] for feature in feature_names]).reshape(1, -1)
        
        # Apply scaling to the input data
        input_array_scaled = scaler.transform(input_array)
        
        prediction = model.predict(input_array_scaled)
        probability = model.predict_proba(input_array_scaled)[0][1]
        
        output = 'Positive' if prediction[0] == 1 else 'Negative'
        
        return jsonify({
            'prediction': output,
            'probability': float(probability)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    
@app.route('/bmi_calculator', methods=['GET', 'POST'])
def bmi_calculator():
    bmi_value = None
    if request.method == 'POST':
        try:
            height = request.form.get('height', type=float)
            weight = request.form.get('weight', type=float)
            if height and weight:
                if height <= 0 or weight <= 0:
                    raise ValueError("Height and weight must be positive values")
                bmi_value = round(weight / ((height/100) ** 2), 2)  # Assuming height is in cm
            else:
                raise ValueError("Both height and weight are required")
        except Exception as e:
            return render_template('bmi_calculator.html', error=str(e))
    return render_template('bmi_calculator.html', bmi=bmi_value)

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == "__main__":
    app.run(debug=True)
