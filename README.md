# OIBSIP_DataScience_task3
OIBSIP Internship – Data Science Task 3:Car Price Prediction with Machine Learning
# OIBSIP Machine Learning Task 3: Car Price Prediction

#Objective
This task is part of the Oasis Infobyte Internship Program (OIBSIP).  
The goal of Task 3 is to Car Price Prediction with Machine Learning:The price of a car depends on a lot of factors like the goodwill of the brand of the car,
features of the car, horsepower and the mileage it gives and many more. Car price
prediction is one of the major research areas in machine learning. So if you want to learn
how to train a car price prediction model then this project is for you.



# Steps Performed
1. **Data Loading** – Loaded car dataset from CSV file.
2. **Data Cleaning** – Removed missing values, duplicates, and outliers.
3. **Exploratory Data Analysis (EDA)** – Generated statistics and visualizations (price distributions, correlations, comparisons by fuel/transmission).
4. **Feature Engineering & Preparation** – Encoded categorical variables, scaled features, and split dataset into training/testing.
5. **Model Training** – Trained multiple models:
   - Linear Regression, Ridge, Lasso
   - Decision Tree, Random Forest, Gradient Boosting
   - Support Vector Regression (SVR)
6. **Model Evaluation** – Compared R², RMSE, MAE across models.
7. **Hyperparameter Tuning** – Used GridSearchCV to optimize Random Forest.
8. **Prediction on New Data** – Predicted prices for unseen cars.
9. **Insights & Recommendations** – Summarized findings from analysis and model performance.

# Tools Used
- **Python 3.11.9**  
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, SciPy  
- **Dataset:** *Car Data (CSV)*  
- **Editor:** VS Code / Jupyter Notebook  

#Outcome
- Successfully built a machine learning pipeline for car price prediction.
- Identified **Random Forest & Gradient Boosting** as top-performing models with high R² and low RMSE.
- Provided actionable insights on fuel type, transmission, and year impact on car prices.
- Created a flexible script to make predictions on new car data.

#Key Visualizations
- Distributions of selling price, present price, and kilometers driven.  
- Average selling price by **fuel type** and **transmission**.  
- **Correlation heatmap** of numerical features.  
- **Actual vs Predicted** and **residual plots** for model evaluation.  
- **Feature importance** for tree-based models.  

# How to Run
1.Dataset:car data.csv
2.Run the script:
```bash
python task_3.py

