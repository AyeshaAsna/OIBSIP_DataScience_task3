# Car Price Prediction with Machine Learning
# This script analyzes car data and builds ML models to predict car prices
# based on various features like brand, year, mileage, fuel type, etc.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_data():
    """Load and examine the car dataset"""
    print("Loading Car Dataset...")
    
    try:
        # Load the CSV file
        df = pd.read_csv('datasets/car data.csv')
        
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print("\nFirst few rows:")
        print(df.head())
        
        print("\nDataset Info:")
        print(df.info())
        
        return df
    
    except FileNotFoundError:
        print("Error: File 'datasets/car data.csv' not found!")
        print("Please make sure the file is in the correct location.")
        return None
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def clean_data(df):
    """Clean and preprocess the car data"""
    print("\n" + "="*60)
    print("DATA CLEANING AND PREPROCESSING")
    print("="*60)
    
    # Make a copy to avoid modifying original data
    df_clean = df.copy()
    
    # Check for missing values
    print("Missing values in each column:")
    print(df_clean.isnull().sum())
    
    # Remove rows with missing values
    df_clean = df_clean.dropna()
    print(f"\nShape after removing missing values: {df_clean.shape}")
    
    # Check data types and convert if needed
    print("\nData types:")
    print(df_clean.dtypes)
    
    # Clean column names (remove extra spaces and special characters)
    df_clean.columns = df_clean.columns.str.strip()
    
    # Check for duplicate rows
    duplicates = df_clean.duplicated().sum()
    print(f"\nDuplicate rows: {duplicates}")
    
    if duplicates > 0:
        df_clean = df_clean.drop_duplicates()
        print(f"Shape after removing duplicates: {df_clean.shape}")
    
    # Check for outliers in numerical columns
    print("\nChecking for outliers in numerical columns...")
    numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df_clean[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)]
        print(f"{col}: {len(outliers)} outliers")
    
    return df_clean

def explore_data(df):
    """Explore the dataset with basic statistics and visualizations"""
    print("\n" + "="*60)
    print("DATA EXPLORATION")
    print("="*60)
    
    # Basic statistics for numerical columns
    print("Basic Statistics:")
    print(df.describe())
    
    # Check unique values in categorical columns
    print("\nUnique values in categorical columns:")
    for col in df.select_dtypes(include=['object']).columns:
        print(f"{col}: {df[col].nunique()} unique values")
        print(f"Values: {df[col].unique()}")
        print()
    
    # Create visualizations
    create_exploratory_plots(df)

def create_exploratory_plots(df):
    """Create various exploratory plots for car data"""
    print("Creating exploratory visualizations...")
    
    # Set up the plotting area
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Selling Price Distribution
    plt.subplot(3, 3, 1)
    plt.hist(df['Selling_Price'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Selling Price (Lakhs)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Selling Price')
    plt.grid(True, alpha=0.3)
    
    # 2. Present Price Distribution
    plt.subplot(3, 3, 2)
    plt.hist(df['Present_Price'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    plt.xlabel('Present Price (Lakhs)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Present Price')
    plt.grid(True, alpha=0.3)
    
    # 3. Driven Kilometers Distribution
    plt.subplot(3, 3, 3)
    plt.hist(df['Driven_kms'], bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
    plt.xlabel('Driven Kilometers')
    plt.ylabel('Frequency')
    plt.title('Distribution of Driven Kilometers')
    plt.grid(True, alpha=0.3)
    
    # 4. Selling Price by Fuel Type
    plt.subplot(3, 3, 4)
    fuel_price = df.groupby('Fuel_Type')['Selling_Price'].mean().sort_values(ascending=False)
    bars = plt.bar(range(len(fuel_price)), fuel_price.values, color=['gold', 'lightblue', 'lightgreen'])
    plt.xlabel('Fuel Type')
    plt.ylabel('Average Selling Price (Lakhs)')
    plt.title('Average Selling Price by Fuel Type')
    plt.xticks(range(len(fuel_price)), fuel_price.index)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}', ha='center', va='bottom')
    
    # 5. Selling Price by Transmission
    plt.subplot(3, 3, 5)
    transmission_price = df.groupby('Transmission')['Selling_Price'].mean()
    colors = ['lightblue', 'lightgreen']
    bars = plt.bar(transmission_price.index, transmission_price.values, color=colors)
    plt.xlabel('Transmission')
    plt.ylabel('Average Selling Price (Lakhs)')
    plt.title('Average Selling Price by Transmission')
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}', ha='center', va='bottom')
    
    # 6. Selling Price vs Year
    plt.subplot(3, 3, 6)
    year_price = df.groupby('Year')['Selling_Price'].mean().sort_index()
    plt.plot(year_price.index, year_price.values, marker='o', linewidth=2, markersize=4)
    plt.xlabel('Year')
    plt.ylabel('Average Selling Price (Lakhs)')
    plt.title('Selling Price vs Year')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 7. Correlation Heatmap
    plt.subplot(3, 3, 7)
    # Select only numerical columns for correlation
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    if len(numerical_cols) > 1:
        correlation_matrix = df[numerical_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                    square=True, linewidths=0.5)
        plt.title('Correlation Matrix')
    
    # 8. Box Plot of Selling Price by Fuel Type
    plt.subplot(3, 3, 8)
    df.boxplot(column='Selling_Price', by='Fuel_Type', ax=plt.gca())
    plt.title('Selling Price Distribution by Fuel Type')
    plt.suptitle('')  # Remove default suptitle
    plt.grid(True, alpha=0.3)
    
    # 9. Scatter Plot: Selling Price vs Driven Kilometers
    plt.subplot(3, 3, 9)
    plt.scatter(df['Driven_kms'], df['Selling_Price'], alpha=0.6, color='purple')
    plt.xlabel('Driven Kilometers')
    plt.ylabel('Selling Price (Lakhs)')
    plt.title('Selling Price vs Driven Kilometers')
    plt.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(df['Driven_kms'], df['Selling_Price'], 1)
    p = np.poly1d(z)
    plt.plot(df['Driven_kms'], p(df['Driven_kms']), "r--", alpha=0.8)
    
    plt.tight_layout()
    plt.show()

def prepare_data(df):
    """Prepare data for machine learning"""
    print("\n" + "="*60)
    print("DATA PREPARATION FOR ML")
    print("="*60)
    
    # Make a copy for ML preparation
    df_ml = df.copy()
    
    # Encode categorical variables
    print("Encoding categorical variables...")
    le_fuel = LabelEncoder()
    le_transmission = LabelEncoder()
    le_selling_type = LabelEncoder()
    
    df_ml['Fuel_Type_Encoded'] = le_fuel.fit_transform(df_ml['Fuel_Type'])
    df_ml['Transmission_Encoded'] = le_transmission.fit_transform(df_ml['Transmission'])
    df_ml['Selling_type_Encoded'] = le_selling_type.fit_transform(df_ml['Selling_type'])
    
    # Create feature matrix X and target variable y
    features = ['Year', 'Present_Price', 'Driven_kms', 'Fuel_Type_Encoded', 
                'Transmission_Encoded', 'Selling_type_Encoded', 'Owner']
    
    X = df_ml[features]
    y = df_ml['Selling_Price']
    
    print(f"Features used: {features}")
    print(f"Target variable: Selling_Price")
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target variable shape: {y.shape}")
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Testing set size: {X_test.shape[0]}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, features

def train_models(X_train, X_test, y_train, y_test):
    """Train multiple machine learning models"""
    print("\n" + "="*60)
    print("MODEL TRAINING")
    print("="*60)
    
    # Define models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'Support Vector Regression': SVR(kernel='rbf'),
        'Decision Tree': DecisionTreeRegressor(random_state=42)
    }
    
    # Train and evaluate models
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        results[name] = {
            'model': model,
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'rmse': rmse,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'y_pred': y_pred
        }
        
        print(f"MSE: {mse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R² Score: {r2:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"Cross-validation R²: {cv_mean:.4f} (+/- {cv_std * 2:.4f})")
    
    return results

def evaluate_models(results, y_test):
    """Evaluate and compare model performance"""
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    # Compare R² scores
    r2_scores = {name: result['r2'] for name, result in results.items()}
    
    plt.figure(figsize=(15, 6))
    
    # R² Score comparison
    plt.subplot(1, 2, 1)
    names = list(r2_scores.keys())
    values = list(r2_scores.values())
    bars = plt.bar(names, values, color=['skyblue', 'lightgreen', 'lightcoral', 'gold', 'plum', 'orange', 'red'])
    plt.xlabel('Models')
    plt.ylabel('R² Score')
    plt.title('Model R² Score Comparison')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom')
    
    # RMSE comparison
    plt.subplot(1, 2, 2)
    rmse_scores = {name: result['rmse'] for name, result in results.items()}
    rmse_values = list(rmse_scores.values())
    
    bars = plt.bar(names, rmse_values, color=['skyblue', 'lightgreen', 'lightcoral', 'gold', 'plum', 'orange', 'red'])
    plt.xlabel('Models')
    plt.ylabel('RMSE')
    plt.title('Model RMSE Comparison (Lower is Better)')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, rmse_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    # Find best model
    best_model_name = max(r2_scores, key=r2_scores.get)
    best_model = results[best_model_name]['model']
    best_r2 = r2_scores[best_model_name]
    
    print(f"\nBest Model: {best_model_name}")
    print(f"Best R² Score: {best_r2:.4f}")
    
    return best_model_name, best_model

def detailed_evaluation(best_model, best_model_name, X_test, y_test, features):
    """Detailed evaluation of the best model"""
    print("\n" + "="*60)
    print(f"DETAILED EVALUATION: {best_model_name}")
    print("="*60)
    
    # Make predictions
    y_pred = best_model.predict(X_test)
    
    # Calculate detailed metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    
    # Residual analysis
    residuals = y_test - y_pred
    
    plt.figure(figsize=(15, 10))
    
    # 1. Actual vs Predicted
    plt.subplot(2, 3, 1)
    plt.scatter(y_test, y_pred, alpha=0.6, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Actual vs Predicted Prices')
    plt.grid(True, alpha=0.3)
    
    # 2. Residuals vs Predicted
    plt.subplot(2, 3, 2)
    plt.scatter(y_pred, residuals, alpha=0.6, color='green')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Price')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted Prices')
    plt.grid(True, alpha=0.3)
    
    # 3. Residuals distribution
    plt.subplot(2, 3, 3)
    plt.hist(residuals, bins=30, alpha=0.7, color='orange', edgecolor='black')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Distribution of Residuals')
    plt.grid(True, alpha=0.3)
    
    # 4. Q-Q plot for residuals
    plt.subplot(2, 3, 4)
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot of Residuals')
    plt.grid(True, alpha=0.3)
    
    # 5. Feature importance (for tree-based models)
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = best_model.feature_importances_
        plt.subplot(2, 3, 5)
        bars = plt.bar(features, feature_importance, color='skyblue')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.title(f'Feature Importance - {best_model_name}')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, importance in zip(bars, feature_importance):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{importance:.3f}', ha='center', va='bottom')
    
    # 6. Model performance over different price ranges
    plt.subplot(2, 3, 6)
    price_ranges = pd.cut(y_test, bins=5)
    range_performance = pd.DataFrame({
        'Price_Range': price_ranges,
        'Actual': y_test,
        'Predicted': y_pred
    })
    
    range_rmse = range_performance.groupby('Price_Range').apply(
        lambda x: np.sqrt(mean_squared_error(x['Actual'], x['Predicted']))
    )
    
    plt.bar(range(len(range_rmse)), range_rmse.values, color='lightcoral')
    plt.xlabel('Price Range')
    plt.ylabel('RMSE')
    plt.title('Model Performance by Price Range')
    plt.xticks(range(len(range_rmse)), [str(r) for r in range_rmse.index], rotation=45)
    
    plt.tight_layout()
    plt.show()

def hyperparameter_tuning(X_train, y_train):
    """Perform hyperparameter tuning for the best model"""
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING")
    print("="*60)
    
    # Tune Random Forest (usually performs well on this type of data)
    rf = RandomForestRegressor(random_state=42)
    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    print("Tuning Random Forest hyperparameters...")
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation R² score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def make_predictions(best_model, scaler, features):
    """Make predictions on new car data"""
    print("\n" + "="*60)
    print("MAKING PREDICTIONS")
    print("="*60)
    
    # Example new car data (you can modify these values)
    new_cars = [
        {
            'Year': 2020,
            'Present_Price': 8.5,
            'Driven_kms': 45000,
            'Fuel_Type': 'Petrol',
            'Transmission': 'Manual',
            'Selling_type': 'Individual',
            'Owner': 1
        },
        {
            'Year': 2018,
            'Present_Price': 12.0,
            'Driven_kms': 60000,
            'Fuel_Type': 'Diesel',
            'Transmission': 'Automatic',
            'Selling_type': 'Dealer',
            'Owner': 1
        },
        {
            'Year': 2022,
            'Present_Price': 15.0,
            'Driven_kms': 15000,
            'Fuel_Type': 'Petrol',
            'Transmission': 'Manual',
            'Selling_type': 'Individual',
            'Owner': 0
        }
    ]
    
    # Encode categorical variables
    le_fuel = LabelEncoder()
    le_transmission = LabelEncoder()
    le_selling_type = LabelEncoder()
    
    # Fit encoders on sample data to get the same encoding
    sample_fuel = ['Petrol', 'Diesel', 'CNG']
    sample_transmission = ['Manual', 'Automatic']
    sample_selling_type = ['Individual', 'Dealer']
    
    le_fuel.fit(sample_fuel)
    le_transmission.fit(sample_transmission)
    le_selling_type.fit(sample_selling_type)
    
    print("Predictions on new car data:")
    for i, car in enumerate(new_cars, 1):
        # Encode categorical features
        fuel_encoded = le_fuel.transform([car['Fuel_Type']])[0]
        transmission_encoded = le_transmission.transform([car['Transmission']])[0]
        selling_type_encoded = le_selling_type.transform([car['Selling_type']])[0]
        
        # Create feature vector
        features_vector = [
            car['Year'],
            car['Present_Price'],
            car['Driven_kms'],
            fuel_encoded,
            transmission_encoded,
            selling_type_encoded,
            car['Owner']
        ]
        
        # Scale features
        features_scaled = scaler.transform([features_vector])
        
        # Make prediction
        predicted_price = best_model.predict(features_scaled)[0]
        
        print(f"\nCar {i}:")
        print(f"Year: {car['Year']}")
        print(f"Present Price: {car['Present_Price']} Lakhs")
        print(f"Driven Kilometers: {car['Driven_kms']}")
        print(f"Fuel Type: {car['Fuel_Type']}")
        print(f"Transmission: {car['Transmission']}")
        print(f"Selling Type: {car['Selling_type']}")
        print(f"Owner: {car['Owner']}")
        print(f"Predicted Selling Price: {predicted_price:.2f} Lakhs")

def generate_insights(df, results):
    """Generate insights and recommendations"""
    print("\n" + "="*60)
    print("INSIGHTS AND RECOMMENDATIONS")
    print("="*60)
    
    insights = []
    
    # 1. Overall price statistics
    avg_price = df['Selling_Price'].mean()
    median_price = df['Selling_Price'].median()
    insights.append(f"Average car selling price: {avg_price:.2f} Lakhs")
    insights.append(f"Median car selling price: {median_price:.2f} Lakhs")
    
    # 2. Price by fuel type
    fuel_price = df.groupby('Fuel_Type')['Selling_Price'].mean()
    best_fuel = fuel_price.idxmax()
    best_fuel_price = fuel_price.max()
    insights.append(f"Highest average price by fuel type: {best_fuel} ({best_fuel_price:.2f} Lakhs)")
    
    # 3. Price by transmission
    transmission_price = df.groupby('Transmission')['Selling_Price'].mean()
    if 'Automatic' in transmission_price.index and 'Manual' in transmission_price.index:
        auto_price = transmission_price['Automatic']
        manual_price = transmission_price['Manual']
        insights.append(f"Automatic cars average price: {auto_price:.2f} Lakhs")
        insights.append(f"Manual cars average price: {manual_price:.2f} Lakhs")
        
        if auto_price > manual_price:
            insights.append("Automatic cars command higher prices than manual cars")
        else:
            insights.append("Manual cars command higher prices than automatic cars")
    
    # 4. Year impact on price
    year_correlation = df['Year'].corr(df['Selling_Price'])
    insights.append(f"Correlation between year and selling price: {year_correlation:.3f}")
    
    if year_correlation > 0.5:
        insights.append("Strong positive correlation: Newer cars have higher prices")
    elif year_correlation < -0.5:
        insights.append("Strong negative correlation: Older cars have higher prices")
    else:
        insights.append("Moderate correlation between year and price")
    
    # 5. Model performance insights
    best_model_name = max(results.keys(), key=lambda x: results[x]['r2'])
    best_r2 = results[best_model_name]['r2']
    insights.append(f"Best performing model: {best_model_name} (R² = {best_r2:.3f})")
    
    if best_r2 > 0.8:
        insights.append("Excellent model performance for price prediction")
    elif best_r2 > 0.6:
        insights.append("Good model performance for price prediction")
    else:
        insights.append("Model performance could be improved with feature engineering")
    
    # Print insights
    for i, insight in enumerate(insights, 1):
        print(f"{i}. {insight}")
    
    # Recommendations
    print("\nRECOMMENDATIONS:")
    print("1. Use the best performing model for accurate price predictions")
    print("2. Consider feature engineering to improve model performance")
    print("3. Regular model retraining with new data for better accuracy")
    print("4. Monitor model performance across different price ranges")
    print("5. Use ensemble methods for more robust predictions")

def main():
    """Main function to run the complete car price prediction pipeline"""
    print("CAR PRICE PREDICTION WITH MACHINE LEARNING")
    print("="*70)
    
    try:
        # Load data
        df = load_data()
        if df is None:
            return
        
        # Clean data
        df_clean = clean_data(df)
        
        # Explore data
        explore_data(df_clean)
        
        # Prepare data for ML
        X_train, X_test, y_train, y_test, scaler, features = prepare_data(df_clean)
        
        # Train models
        results = train_models(X_train, X_test, y_train, y_test)
        
        # Evaluate models
        best_model_name, best_model = evaluate_models(results, y_test)
        
        # Detailed evaluation of best model
        detailed_evaluation(best_model, best_model_name, X_test, y_test, features)
        
        # Hyperparameter tuning
        tuned_model = hyperparameter_tuning(X_train, y_train)
        
        # Make predictions
        make_predictions(tuned_model, scaler, features)
        
        # Generate insights
        generate_insights(df_clean, results)
        
        print("\n" + "="*70)
        print("CAR PRICE PREDICTION PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*70)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

















