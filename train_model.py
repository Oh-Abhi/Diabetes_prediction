"""
Diabetes Prediction Model Training and Evaluation Pipeline
-----------------------------------------------------------
This script builds an end-to-end Machine Learning pipeline using the Pima Indians Diabetes dataset.
It is designed as an educational tool for student interns to learn the lifecycle of an ML project:
1. Data Ingestion & Inspection
2. Data Preprocessing & Cleaning (treating 0s as missing values, duplicate removal)
3. Exploratory Data Analysis (EDA) with visualizations
4. Feature Engineering (splitting data, imputation, and standard scaling)
5. Model Training (Logistic Regression, Decision Tree, Random Forest)
6. Model Evaluation (Accuracy, Precision, Recall, F1 Score, Confusion Matrix)
7. Model Selection & Persistence (saving the best model and scaler)
8. Feature Importance Analysis
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for plots to make them look premium
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16
})

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def main():
    print("=" * 80)
    print("      STARTING DIABETES PREDICTION MACHINE LEARNING PIPELINE      ")
    print("=" * 80)
    
    # Create directory for plots
    os.makedirs('plots', exist_ok=True)
    print("[INFO] Created 'plots/' directory to save exploratory visualizations.\n")
    
    # -------------------------------------------------------------------------
    # STEP 1: DATA LOADING
    # -------------------------------------------------------------------------
    print("-" * 80)
    print("STEP 1: Data Ingestion & Initial Inspection")
    print("-" * 80)
    print("Purpose: To load raw dataset files and inspect their structure (rows, columns, data types).")
    
    csv_path = 'diabetes.csv'
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}. Please place diabetes.csv in the same directory.")
        
    df = pd.read_csv(csv_path)
    
    print(f"\n[Dataset Shape]: {df.shape[0]} rows (samples) and {df.shape[1]} columns (features).")
    print("\n[Column Names]:", list(df.columns))
    
    print("\n[Basic Information]:")
    df.info()
    
    print("\n[First 5 Rows of Dataset]:")
    print(df.head())
    print("\n[Explanation for Interns]:")
    print("  Our target column is 'Outcome' where 1 indicates diabetes and 0 indicates non-diabetes.")
    print("  The other 8 columns are features representing physiological measurements of the patients.\n")
    
    # -------------------------------------------------------------------------
    # STEP 2: DATA PREPROCESSING (CLEANING)
    # -------------------------------------------------------------------------
    print("-" * 80)
    print("STEP 2: Data Preprocessing & Cleaning")
    print("-" * 80)
    
    # A. Check for duplicates
    print("\n--- checking for duplicate records ---")
    duplicate_count = df.duplicated().sum()
    print(f"Number of duplicate rows found: {duplicate_count}")
    if duplicate_count > 0:
        df = df.drop_duplicates()
        print("Dropped duplicate records.")
    
    # B. Treat 0 values as missing in specific columns
    # In columns Glucose, BloodPressure, SkinThickness, Insulin, and BMI, a value of 0 is physiologically 
    # impossible for a living human. Thus, these 0s are actually missing (null) values in disguise.
    cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    
    print("\n--- Treating 0 values as missing (NaN) ---")
    print("0 values count per column before replacement:")
    for col in cols_with_zeros:
        zero_count = (df[col] == 0).sum()
        print(f"  - {col}: {zero_count} zeros ({(zero_count/len(df))*100:.1f}%)")
        # Replace 0 with NaN
        df[col] = df[col].replace(0, np.nan)
        
    print("\nMissing values (NaN) count per column after replacing 0s:")
    print(df.isnull().sum())
    
    # -------------------------------------------------------------------------
    # STEP 3: EXPLORATORY DATA ANALYSIS (EDA)
    # -------------------------------------------------------------------------
    print("-" * 80)
    print("STEP 3: Exploratory Data Analysis (EDA) & Visualizations")
    print("-" * 80)
    print("Purpose: To visually and statistically understand the relationships, distributions, and patterns in our data.")
    
    # Descriptive statistics
    print("\n[Descriptive Statistics (including NaNs)]:")
    print(df.describe())
    
    # Plot 1: Outcome distribution
    plt.figure(figsize=(6, 5))
    ax = sns.countplot(x='Outcome', data=df, palette='viridis')
    plt.title('Distribution of Diabetic vs. Non-Diabetic Patients')
    plt.xlabel('Outcome (0 = Non-Diabetic, 1 = Diabetic)')
    plt.ylabel('Count')
    
    # Add count labels on top of bars
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height() + 5),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points')
                    
    plt.tight_layout()
    plt.savefig('plots/outcome_distribution.png', dpi=300)
    plt.close()
    print("\n[EDA Plot Saved]: plots/outcome_distribution.png")
    print("  Explanation: This bar chart shows the balance of our target classes. We can see that there are more")
    print("               non-diabetic cases than diabetic cases, which represents a class imbalance. This is typical")
    print("               in medical datasets.")
    
    # Plot 2: Correlation heatmap
    plt.figure(figsize=(10, 8))
    # We calculate correlation ignoring missing values
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, square=True)
    plt.title('Correlation Heatmap of Features')
    plt.tight_layout()
    plt.savefig('plots/correlation_heatmap.png', dpi=300)
    plt.close()
    print("\n[EDA Plot Saved]: plots/correlation_heatmap.png")
    print("  Explanation: This correlation matrix tells us which features move together. High positive correlations")
    print("               (closer to +1.0) imply a strong positive relationship. For example, Age and Pregnancies have a")
    print("               moderate correlation, and Glucose has the strongest direct correlation with the target Outcome.")
    
    # Plot 3: Feature distributions
    # We will plot histograms for all features to see their distributions
    features_to_plot = df.columns.drop('Outcome')
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, col in enumerate(features_to_plot):
        sns.histplot(df[col].dropna(), kde=True, ax=axes[i], color='teal', bins=20)
        axes[i].set_title(f'{col} Distribution')
        axes[i].set_xlabel('')
        axes[i].set_ylabel('Density')
        
    # Hide the 9th empty plot
    axes[8].set_visible(False)
    plt.suptitle('Distribution of Health Features', y=0.98, fontsize=16)
    plt.tight_layout()
    plt.savefig('plots/feature_distributions.png', dpi=300)
    plt.close()
    print("\n[EDA Plot Saved]: plots/feature_distributions.png")
    print("  Explanation: These distribution plots show the shape of each feature. Some columns, like Glucose,")
    print("               resemble a normal (bell-shaped) curve, while others, like Insulin, SkinThickness, and")
    print("               DiabetesPedigreeFunction, are highly right-skewed (tail extends right). Knowing this helps us")
    print("               decide on scaling and preprocessing methods.")
    
    # -------------------------------------------------------------------------
    # STEP 4: FEATURE ENGINEERING & DATA SPLITTING
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("STEP 4: Feature Engineering & Dataset Splitting")
    print("-" * 80)
    print("Purpose: To split the data into training/testing sets, impute missing values, and normalize features.")
    
    # A. Separate features and target
    X = df.drop(columns=['Outcome'])
    y = df['Outcome']
    
    # B. Train-Test Split
    # We use a fixed random state (42) to ensure that we get the exact same split every time we run the script.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    print(f"\nSuccessfully split data using an 80/20 train-test ratio:")
    print(f"  - Training Set Features: {X_train.shape}")
    print(f"  - Testing Set Features: {X_test.shape}")
    print("  Note: Stratified splitting is used to maintain the ratio of Diabetic vs Non-Diabetic in both splits.")
    
    # C. Median Imputation
    # CRITICAL: We fit the median imputer on X_train only, and then transform both X_train and X_test.
    # This prevents data leakage (leakage occurs when statistics of test set leak into training phase).
    print("\n--- Imputing Missing Values (Median Strategy) ---")
    imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    
    # Fit on training data
    imputer.fit(X_train)
    
    # Log medians calculated from training data
    medians_dict = dict(zip(X.columns, imputer.statistics_))
    print("Calculated training set medians used for imputation:")
    for feature, med in medians_dict.items():
         print(f"  - {feature}: {med}")
         
    # Transform both training and test data
    X_train_imputed = imputer.transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    
    # D. Feature Scaling
    # Machine Learning models (especially Logistic Regression) perform better when numerical features are 
    # on the same scale. We standardize the features to have a mean of 0 and standard deviation of 1.
    print("\n--- Standardizing Features (StandardScaler) ---")
    scaler = StandardScaler()
    
    # Fit scaler on imputed training features
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    
    # Transform test features
    X_test_scaled = scaler.transform(X_test_imputed)
    
    # Save the scaler so the Streamlit web application can use it for real-time user input
    joblib.dump(scaler, 'scaler.pkl')
    # Save training medians to a file to reconstruct the imputer in streamlit if needed
    # We will actually save the imputer object as well to simplify app logic
    joblib.dump(imputer, 'imputer.pkl')
    print("Saved 'scaler.pkl' and 'imputer.pkl' successfully using joblib.")
    print("  Explanation: The Streamlit app will load these to clean and scale the user's entries in exactly the")
    print("               same way the training set was cleaned and scaled.")
    
    # -------------------------------------------------------------------------
    # STEP 5 & 6: MODEL TRAINING & EVALUATION
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("STEP 5 & 6: Model Training & Evaluation")
    print("-" * 80)
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    # Dictionary to store performance metrics
    results = {}
    
    for name, model in models.items():
        print(f"\n>>> Training Model: {name}...")
        
        # Fit model on scaled training data
        model.fit(X_train_scaled, y_train)
        
        # Make predictions on scaled test data
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        # Save metrics
        results[name] = {
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1 Score': f1,
            'Confusion Matrix': cm,
            'Classification Report': report,
            'Model Object': model
        }
        
        # Output metrics
        print(f"    - Accuracy:  {acc:.4f} (Overall correctness)")
        print(f"    - Precision: {prec:.4f} (Of all predicted positives, how many were correct?)")
        print(f"    - Recall:    {rec:.4f} (Of all actual positives, how many did we find?)")
        print(f"    - F1 Score:  {f1:.4f} (Harmonic mean of Precision and Recall)")
        print("    - Confusion Matrix:")
        print(f"        [[TN={cm[0][0]}, FP={cm[0][1]}],")
        print(f"         [FN={cm[1][0]}, TP={cm[1][1]}]]")
        
    # -------------------------------------------------------------------------
    # STEP 7: MODEL COMPARISON & SELECTION
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("STEP 7: Model Comparison & Selection")
    print("-" * 80)
    
    # Construct a comparison DataFrame
    comparison_df = pd.DataFrame({
        name: {
            'Accuracy': metrics['Accuracy'],
            'Precision': metrics['Precision'],
            'Recall': metrics['Recall'],
            'F1 Score': metrics['F1 Score']
        } for name, metrics in results.items()
    }).T
    
    print("\n[Model Comparison Summary Table]:")
    print(comparison_df.to_string())
    
    # Automatically select the best model
    # Priority metric is F1 Score, followed by Accuracy in case of a tie
    best_model_name = max(results, key=lambda k: (results[k]['F1 Score'], results[k]['Accuracy']))
    best_metrics = results[best_model_name]
    
    print(f"\n[BEST MODEL SELECTED]: {best_model_name}")
    print(f"  F1 Score: {best_metrics['F1 Score']:.4f}")
    print(f"  Accuracy: {best_metrics['Accuracy']:.4f}")
    print("  Explanation: We select the best model using F1 Score because accuracy can be misleading on imbalanced")
    print("               datasets. F1 Score balances both false positives and false negatives, which is crucial in medicine.")
    
    # Save the best model
    joblib.dump(best_metrics['Model Object'], 'diabetes_model.pkl')
    print("\nSaved best model as 'diabetes_model.pkl' successfully using joblib.")
    
    # -------------------------------------------------------------------------
    # STEP 8: FEATURE IMPORTANCE (RANDOM FOREST)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("STEP 8: Feature Importance (Random Forest)")
    print("-" * 80)
    print("Purpose: To understand which features contribute most to the Random Forest model's predictions.")
    
    rf_model = results['Random Forest']['Model Object']
    importances = rf_model.feature_importances_
    
    # Create feature importance dataframe
    feature_importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    print("\n[Feature Importances Table (Sorted)]:")
    print(feature_importance_df.to_string(index=False))
    
    # Plot Feature Importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='mako')
    plt.title('Random Forest Feature Importance')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig('plots/feature_importance.png', dpi=300)
    plt.close()
    
    print("\n[Feature Importance Plot Saved]: plots/feature_importance.png")
    print("  Explanation: This chart shows how much weight the Random Forest classifier places on each health feature.")
    print(f"               The most influential feature in this dataset is '{feature_importance_df.iloc[0]['Feature']}'.")
    print("               This aligns with medical knowledge, as blood glucose level is a critical indicator of diabetes.")
    print("=" * 80)
    print("                    PIPELINE EXECUTED SUCCESSFULLY!                       ")
    print("=" * 80)

if __name__ == "__main__":
    main()
