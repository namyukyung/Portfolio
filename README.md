# Crop Recommendation System
## Project Overview
Crop Recommendation System Development: Developed a crop recommendation system by comparing various machine learning models (kNN, Logistic Regression, Decision Tree, Random Forest, and SVM) and evaluated the optimal model.  
Model Performance Analysis: Analyzed model precision and prediction errors using confusion matrices. While most models achieved a precision of 1.0, the kNN model showed one misclassification due to overlapping features.  
Model Validation and Improvement: Evaluated model performance based on test set results and explored improvement strategies to enhance the model’s generalization performance through additional validation processes.

Dataset Source: Utilized a curated dataset from Kaggle containing agricultural data from various hypothetical regions in India.  
Data Quality: Each crop included 100 data entries.  
Data Cleaning: Began with 21 crop types and refined the dataset to 10 unique crops by removing similar varieties.

## Work Flow
### Data Encoding
Standardization: Standardized numerical variables to have a mean of 0 and a variance of 1 to improve model performance.
Label Encoding: Converted the categorical variable representing crop names into numerical values to enable model training.

### Data Splitting
Stratified Split: Split the data using a stratified approach to prevent class imbalance.
Training 70% / Test 30%: Divided the data into a 70:30 ratio for model training and performance evaluation.

### Model Training
Models Used: Conducted comparative experiments using various algorithms, including Decision Tree, Random Forest, kNN (k=3), Logistic Regression, and SVM.
Evaluation Metric: Calculated the precision of each model using the macro averaging method to account for balanced performance across classes.
10-Fold Cross-Validation (CV): Applied 10-fold cross-validation to prevent overfitting and evaluate the models’ generalization performance.

### Model Evaluation
The final model was selected based on performance comparisons using Accuracy, Precision, and Recall.
Considering data imbalance, greater emphasis was placed on Precision and Recall rather than simple accuracy.

## Final Analysis
Most models achieved a high precision of 1.0, indicating that the data structure was clear and the classes were well-separated.
However, the kNN model showed one misclassification, likely due to its non-parametric nature, which makes it more challenging to distinguish between similar feature patterns.

## Scenario Testing
Simulated crop recommendations based on the September climate conditions of two regions.
First region (hot and humid): All models recommended rice.
Second region (warm and moderately humid): Four models recommended banana.
This process demonstrated the consistency and reliability of the models, enhancing their potential for real-world application.
