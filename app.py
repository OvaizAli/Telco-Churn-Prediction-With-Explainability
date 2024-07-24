import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import lime
import lime.lime_tabular
import plotly.graph_objects as go
import scipy.optimize as opt
import random
import re

# Load the trained stacking model and PCA transformer from the pickle file
with open('stacking_model.pkl', 'rb') as model_file:
    stacking_model = pickle.load(model_file)

# Load the PCA transformer used during training
with open('pca_transformer.pkl', 'rb') as pca_file:
    pca = pickle.load(pca_file)

def random_input():
    """Generate random input values for testing."""
    return {
        'SeniorCitizen': random.choice([0, 1]),
        'tenure': random.randint(0, 72),  # Assuming tenure ranges from 0 to 72 months
        'MonthlyCharges': round(random.uniform(18.25, 111.60), 2),
        'TotalCharges': round(random.uniform(0.0, 8240.85), 2),
        'TotalMonthlyCost': round(random.uniform(18.25, 111.60), 2),
        'ChargesPerMonth': round(random.uniform(18.25, 111.60), 2),
        'NumServices': random.randint(3, 6),  # Assuming 3 to 6 services
        'HasDependents': random.choice([0, 1]),
        'IsSeniorCitizen': random.choice([0, 1]),
        'UsesFiberOptic': random.choice([0, 1]),
        'HasMultipleLines': random.choice([0, 1]),
        'IsMonthToMonthContract': random.choice([0, 1]),
        'UsesElectronicPayment': random.choice([0, 1]),
        'MonthlyChargesRatio': round(random.uniform(0.012952, 0.275130), 2),
        'ServiceToTenureRatio': round(random.uniform(0.029412, 0.855769), 2),
        'StreamingServices': random.choice([0, 1]),
        'SupportServices': random.choice([0, 1]),
        'OnlineBackup': random.choice(['No', 'No internet service', 'Yes']),
        'TenureGroup': random.choice(['0-1 year', '1-2 years', '2-4 years', '4-5 years', '5+ years'])
    }

def user_input_features():
    with st.sidebar:
        st.header('Input Customer Data')
        
        # Button to generate random input
        if st.button('Generate Random Input'):
            data = random_input()
            st.write("Random input values generated:")
            st.write(data)
        else:
            data = {
                'SeniorCitizen': st.selectbox('SeniorCitizen', [0, 1]),
                'tenure': st.number_input('Tenure', min_value=0, max_value=72, value=1),
                'MonthlyCharges': st.number_input('MonthlyCharges', min_value=18.25, max_value=111.60, value=18.25),
                'TotalCharges': st.number_input('TotalCharges', min_value=0.0, max_value=8240.85, value=0.0),
                'TotalMonthlyCost': st.number_input('TotalMonthlyCost', min_value=18.25, max_value=111.60, value=18.25),
                'ChargesPerMonth': st.number_input('ChargesPerMonth', min_value=18.25, max_value=111.60, value=18.25),
                'NumServices': st.number_input('NumServices', min_value=3, max_value=6, value=3),
                'HasDependents': st.selectbox('HasDependents', [0, 1]),
                'IsSeniorCitizen': st.selectbox('IsSeniorCitizen', [0, 1]),
                'UsesFiberOptic': st.selectbox('UsesFiberOptic', [0, 1]),
                'HasMultipleLines': st.selectbox('HasMultipleLines', [0, 1]),
                'IsMonthToMonthContract': st.selectbox('IsMonthToMonthContract', [0, 1]),
                'UsesElectronicPayment': st.selectbox('UsesElectronicPayment', [0, 1]),
                'MonthlyChargesRatio': st.number_input('MonthlyChargesRatio', min_value=0.012952, max_value=0.275130, value=0.012952),
                'ServiceToTenureRatio': st.number_input('ServiceToTenureRatio', min_value=0.029412, max_value=0.855769, value=0.029412),
                'StreamingServices': st.selectbox('StreamingServices', [0, 1]),
                'SupportServices': st.selectbox('SupportServices', [0, 1]),
                'OnlineBackup': st.selectbox('OnlineBackup', ['No', 'No internet service', 'Yes']),
                'TenureGroup': st.selectbox('TenureGroup', ['0-1 year', '1-2 years', '2-4 years', '4-5 years', '5+ years'])
            }

        # Map categorical features to label encodings
        data['OnlineBackup'] = {'No': 0, 'No internet service': 1, 'Yes': 2}[data['OnlineBackup']]
        data['TenureGroup'] = {'0-1 year': 0, '1-2 years': 1, '2-4 years': 2, '4-5 years': 3, '5+ years': 4}[data['TenureGroup']]

        # Define feature names
        feature_names = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges', 'TotalMonthlyCost', 'ChargesPerMonth',
                         'NumServices', 'HasDependents', 'IsSeniorCitizen', 'UsesFiberOptic', 'HasMultipleLines',
                         'IsMonthToMonthContract', 'UsesElectronicPayment', 'MonthlyChargesRatio',
                         'ServiceToTenureRatio', 'StreamingServices', 'SupportServices', 'OnlineBackup', 'TenureGroup']
                         
        features = pd.DataFrame(data, index=[0], columns=feature_names)
        return features

def plot_counterfactual(counterfactual, feature_names, model_name):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=counterfactual,
        y=feature_names,
        orientation='h',
        marker=dict(color='rgba(50, 171, 96, 0.6)',
                    line=dict(color='rgba(50, 171, 96, 1.0)', width=1)),
        name='Counterfactual Contribution'
    ))

    fig.update_layout(
        xaxis=dict(title='Feature Value'),
        yaxis=dict(title='Features'),
        margin=dict(l=200, r=200, t=100, b=100),
        paper_bgcolor='rgb(248, 248, 255)',
        plot_bgcolor='rgb(248, 248, 255)',
        showlegend=False
    )

    return fig

def plot_lime_explanation(exp, pca, original_feature_names, X_test_instance, model_name):
    feature_importances = exp.as_list()
    pc_names = [x[0] for x in feature_importances]
    pc_values = [x[1] for x in feature_importances]

    # Initialize the original feature importances
    original_feature_importances = np.zeros(len(original_feature_names))

    # Map PCA components back to original features
    for j, pc in enumerate(pc_names):
        match = re.search(r'PC (\d+)', pc)
        if match:
            pc_index = int(match.group(1)) - 1  # PCA indices are 1-based
        else:
            continue

        if pc_index >= pca.components_.shape[0]:
            continue

        # Accumulate importances for each original feature
        for k in range(len(original_feature_names)):
            # Contribution of each original feature to this PC
            original_feature_importances[k] += pca.components_[pc_index, k] * pc_values[j]

    # Sort and plot
    sorted_importances = sorted(zip(original_feature_names, original_feature_importances), key=lambda x: x[1], reverse=True)
    original_feature_names = [x[0] for x in sorted_importances]
    original_feature_values = [x[1] for x in sorted_importances]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=original_feature_values,
        y=original_feature_names,
        orientation='h',
        marker=dict(color='rgba(50, 171, 96, 0.6)',
                    line=dict(color='rgba(50, 171, 96, 1.0)', width=1)),
        name='Feature Importance'
    ))

    fig.update_layout(
        title=f'LIME Explanation for {model_name}',
        xaxis=dict(title='Feature Importance'),
        yaxis=dict(title='Features'),
        margin=dict(l=200, r=200, t=100, b=100),
        paper_bgcolor='rgb(248, 248, 255)',
        plot_bgcolor='rgb(248, 248, 255)',
        showlegend=False
    )

    fig.show()

    # Print positive, non-zero feature importances
    print(f"Important Features for the Prediction using {model_name}:")
    for feature, importance in zip(original_feature_names, original_feature_values):
        if importance > 0:
            print(f"{feature}: {importance:.7f}")


def generate_counterfactual(model, X_instance, predict_proba_func, pca, original_feature_names, threshold=0.5):
    def objective_function(x):
        """Objective function to minimize."""
        prob = predict_proba_func(x.reshape(1, -1))[0, 1]
        return -prob

    def constraint_function(x):
        """Constraint function to ensure the prediction is different from the original."""
        prob = predict_proba_func(x.reshape(1, -1))[0, 1]
        return prob - threshold

    # Initial guess in PCA space
    x0 = X_instance.copy()

    # Define bounds for PCA components (ensure these are valid for PCA space)
    bounds = [(-10, 10) for _ in range(x0.shape[0])]  # Adjust bounds if needed based on PCA component range

    # Define constraints
    constraints = [{'type': 'ineq', 'fun': constraint_function}]

    # Optimize to find counterfactual in PCA space
    result = opt.minimize(objective_function, x0, bounds=bounds, constraints=constraints)

    # Transform back to original feature space
    if pca:
        counterfactual_pca = result.x
        counterfactual = pca.inverse_transform(counterfactual_pca.reshape(1, -1))[0]
    else:
        counterfactual = result.x

    return counterfactual

def plot_lime_explanation(exp, pca, original_feature_names, X_test_instance, model_name):
    feature_importances = exp.as_list()
    pc_names = [x[0] for x in feature_importances]
    pc_values = [x[1] for x in feature_importances]

    # Initialize the original feature importances
    original_feature_importances = np.zeros(len(original_feature_names))

    # Map PCA components back to original features
    for j, pc in enumerate(pc_names):
        match = re.search(r'PC (\d+)', pc)
        if match:
            pc_index = int(match.group(1)) - 1  # PCA indices are 1-based
        else:
            continue

        if pc_index >= pca.components_.shape[0]:
            continue

        # Accumulate importances for each original feature
        for k in range(len(original_feature_names)):
            # Contribution of each original feature to this PC
            original_feature_importances[k] += pca.components_[pc_index, k] * pc_values[j]

    # Sort and plot
    sorted_importances = sorted(zip(original_feature_names, original_feature_importances), key=lambda x: x[1], reverse=True)
    original_feature_names = [x[0] for x in sorted_importances]
    original_feature_values = [x[1] for x in sorted_importances]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=original_feature_values,
        y=original_feature_names,
        orientation='h',
        marker=dict(color='rgba(50, 171, 96, 0.6)',
                    line=dict(color='rgba(50, 171, 96, 1.0)', width=1)),
        name='Feature Importance'
    ))

    fig.update_layout(
        xaxis=dict(title='Feature Importance'),
        yaxis=dict(title='Features'),
        margin=dict(l=200, r=200, t=100, b=100),
        paper_bgcolor='rgb(248, 248, 255)',
        plot_bgcolor='rgb(248, 248, 255)',
        showlegend=False
    )

    return fig

def plot_prediction_probabilities(probabilities):
    labels = ['No Churn', 'Churn']
    fig = go.Figure(data=[go.Pie(labels=labels, values=probabilities, hole=.3)])
    
    fig.update_layout(
        annotations=[dict(text=f'{probabilities[1]:.2%}', x=0.5, y=0.5, font_size=20, showarrow=False)],
        paper_bgcolor='rgb(248, 248, 255)',
        plot_bgcolor='rgb(248, 248, 255)'
    )
    
    return fig

# Load the scaler used during training
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

def scale_user_input(user_input):
    """Scale user input data."""
    return scaler.transform(user_input)

def main():
    st.title("Telco Churn Prediction with Explainability")

    # Get user input
    user_input = user_input_features()

    # Separate numerical and categorical features
    numerical_features = [
        'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges', 'TotalMonthlyCost', 'ChargesPerMonth',
        'NumServices', 'HasDependents', 'IsSeniorCitizen', 'UsesFiberOptic', 'HasMultipleLines',
        'IsMonthToMonthContract', 'UsesElectronicPayment', 'MonthlyChargesRatio',
        'ServiceToTenureRatio', 'StreamingServices', 'SupportServices'
    ]

    # Extract numerical features
    numerical_data = user_input[numerical_features]

    # Load the scaler used during training
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)

    def scale_numerical_features(numerical_data):
        """Scale only the numerical features."""
        return pd.DataFrame(scaler.transform(numerical_data), columns=numerical_features)

    # Scale numerical features
    scaled_numerical_data = scale_numerical_features(numerical_data)

    # Combine scaled numerical features with categorical features
    categorical_features = [col for col in user_input.columns if col not in numerical_features]
    scaled_user_input = pd.concat([scaled_numerical_data, user_input[categorical_features]], axis=1)

    # Apply PCA transformation
    user_input_pca = pca.transform(scaled_user_input)

    # Predict using the loaded model
    prediction = stacking_model.predict(user_input_pca)
    prediction_proba = stacking_model.predict_proba(user_input_pca)

    st.subheader("Model's Prediction")

    if prediction[0] == 1:
        st.error(
            f"The customer is likely to **Churn**.\n"
            f"This means they **may cancel** their subscription in the near future. "
            f"Consider implementing strategies to retain this customer."
        )
    else:
        st.success(
            f"The customer is likely to **not Churn**.\n"
            f"This indicates they are **expected to continue** their subscription in the foreseeable future. "
            f"Nevertheless, it's important to monitor their behavior to ensure ongoing satisfaction."
        )

    # Plot the prediction probabilities
    prob_fig = plot_prediction_probabilities(prediction_proba[0])
    st.plotly_chart(prob_fig)

    # # Initialize LIME explainer
    # explainer = lime.lime_tabular.LimeTabularExplainer(
    #     scaled_user_input.values,  # Use the scaled user input data for LIME explainer
    #     feature_names=scaled_user_input.columns.tolist(),
    #     class_names=['No Churn', 'Churn'],
    #     discretize_continuous=True
    # )

    # exp = explainer.explain_instance(scaled_user_input.values[0], lambda x: stacking_model.predict_proba(pca.transform(x)), num_features=len(scaled_user_input.columns))

    # Plot LIME explanation for the StackingClassifier
    original_feature_names = scaled_user_input.columns.tolist()
    # lime_fig = plot_lime_explanation(exp, pca, original_feature_names, scaled_user_input.values[0], 'StackingClassifier')
    
    # st.subheader("Prediction Feature Importance Plot")
    # st.plotly_chart(lime_fig)

    # Counterfactual explanation
    st.subheader("Counterfactual Explanation")

    try:
        # Generate counterfactual explanation
        counterfactual = generate_counterfactual(
            stacking_model,
            user_input_pca[0],
            stacking_model.predict_proba,
            pca,
            original_feature_names
        )

        # Display counterfactual features with positive values
        positive_features = [name for name, value in zip(original_feature_names, counterfactual) if value > 0]

        if positive_features:
            # Convert the list to a comma-separated string
            positive_features_str = ', '.join(positive_features)
            
            # Define explanations based on predicted class
            if prediction[0] == 1:
                explanation = (
                    f"- In the current prediction scenario of **Churn**, the following features with positive counterfactual values:\n"
                    f"**{positive_features_str}** "
                    "indicate that lowering the impact of these features could prevent churn.\n"
                    f"- The company should consider strategies to **reduce** the impact or cause of these feature values to potentially lower the likelihood of churn. For example, if 'MonthlyCharges' has a positive counterfactual value, lowering this charge might help in reducing the chance of the customer churning.\n"
                )
            else:
                explanation = (
                    f"- In the current prediction scenario of **No Churn**, the following features with positive counterfactual values:\n"
                    f"**{positive_features_str}** "
                    "suggest that maintaining or enhancing the impact of these features could help sustain the prediction of no churn.\n"
                    f"- The company should focus on **retaining** these positive features to ensure continued customer satisfaction and retention. For example, if 'MonthlyCharges' has a positive counterfactual value, keeping or improving this feature may support the customer staying with the service.\n"
                )

            st.success(explanation)

        else:
            st.info("No features with positive counterfactual values.")

        # Plot counterfactual explanation
        counterfactual_fig = plot_counterfactual(counterfactual, original_feature_names, 'StackingClassifier')
        st.plotly_chart(counterfactual_fig)

    except KeyError as e:
        st.write(f"KeyError encountered: {e}. Check the data indexing.")
    except ValueError as e:
        st.write(f"ValueError encountered: {e}. Check the feature dimensions.")

if __name__ == '__main__':
    main()
