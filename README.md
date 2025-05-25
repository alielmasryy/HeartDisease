# Heart Disease Project
### Introduction
The ability to predict heart disease accurately is vital for preventive healthcare. The primary
research question for my project is: How can machine learning techniques be used to predict
heart disease based on various medical features?
Additional questions include:
   -What features from a patient's medical history most effectively predict heart disease?
   -Can we optimize machine learning models to enhance prediction accuracy?
   -How do various models compare in terms of performance?
   
### Relevance, Importance, and ML Contribution
Heart disease remains one of the leading causes of death worldwide. Early prediction of heart
disease allows healthcare providers to initiate preventive measures, which can significantly
reduce mortality rates. Machine learning models have the potential to assist doctors by making
predictions based on a patient's medical data and improving the quality of healthcare delivery.
Machine learning (ML) techniques can analyze large datasets with many variables to uncover
patterns that may not be evident to doctors. By using these tools, we can help reduce diagnostic
errors and improve healthcare outcomes. This research contributes to the application of ML in
healthcare, specifically for heart disease prediction.

### Summary of the work done
* Data:
    * Input: ZIP File, description of hospital patients information(resting heart rate, age, cholestral, etc. )
    * Output: Out of a list of all patients, which ones have a heart disease
  * Size: A relatively small dataset with 14 columns and 303 rows
 
#### Preprocessing / Clean up

* A very small number of data points were missing so they were simply dropped as it did not have an impact in mode performance
* New binary features were created to reflect high risk indicators based on clinical thresholds
* Categorical features like sex, cp, and thal were encoded to allow the models to handle them effectively

#### Data Visualization
I am only including few examples of before and after clean up, as the rest can be seen when running the code

* BEFORE:
<img width="447" alt="Screenshot 2025-05-24 at 11 49 33 PM" src="https://github.com/user-attachments/assets/ffbb48f2-4338-49fc-8ebe-bf7f8484ecb1" />
<img width="447" alt="Screenshot 2025-05-24 at 11 49 41 PM" src="https://github.com/user-attachments/assets/461ec7c7-d632-40f2-a04f-99e0db305d96" />
* AFTER:
<img width="734" alt="Screenshot 2025-05-24 at 11 50 55 PM" src="https://github.com/user-attachments/assets/3809e673-3d49-466e-932b-1defe3e39420" />
<img width="707" alt="Screenshot 2025-05-24 at 11 51 12 PM" src="https://github.com/user-attachments/assets/e2d65741-9287-4782-8877-e1331e96e100" />

### Methods:
* Logistic Regression - Logistic Regression was used as a starting point for modeling heart disease prediction. It is a simple, but powerful algorithm for binary classification problems. The model estimates the probability that a patient has heart disease based on their medical features. Because it provides interpretable coefficients, it also helps understand how each variable (like cholesterol or blood pressure) affects the likelihood of heart disease. The model was trained on the cleaned and feature-engineered dataset and evaluated using accuracy, precision, recall, F1-score, and the ROC AUC score to assess both its correctness and ability to distinguish between positive and negative cases.
* Support Vector Machine (SVM) - Support Vector Machine with a linear kernel was used to classify patients with and without heart disease. SVM is effective in high-dimensional spaces and is known for maximizing the decision margin between classes. Hyperparameter tuning was performed using cross-validation to optimize the model. Evaluation was conducted using accuracy, precision, recall, F1-score, and ROC AUC, providing a comprehensive view of model performance.
* Random Forest Classifier - The Decision Tree model splits the data into groups based on different features. It splits the data based on feature thresholds, which makes it easy to interpret but susceptible to overfitting. To reduce complexity, the tree depth and minimum samples for a split were controlled. The model was evaluated using accuracy, precision, recall, F1-score, and ROC AUC to understand how well it identifies heart disease cases compared to the other models.

### Results:
NOTE: I deemed the most important measurement in this case to be recall, more specifically in the positive (class 1) cases, so that was my focus for this project.

#### Logistic Regression:
 * Accuracy: 0.88
 * ROC AUC: 0.89
 * Recall (Class 0-No heart disease): 0.89
 * Precision: (Class 0): 0.91
 * Recall (Class 1-Heart disease): 0.88
 * Precision (Class 1): 0.84
 * F1 (Class 1): 0.86

 * Conclusion: The logistic model performed well overall, particularly in maintaining a strong balance between precision and recall. The results also provided interpretability into how each feature contributes to the prediction, which is valuable in a clinical setting.

#### Support Vector Machine (SVM):
  *  Accuracy: 0.85
  *  ROC AUC: 0.92
  *  Recall (Class 0-No heart disease): 0.92
  *  Precision: (Class 0): 0.85
  *  Recall (Class 1-Heart disease): 0.75
  *  Precision (Class 1): 0.86
 *   F1 (Class 1): 0.80
 * Conclusion: The SVM model with a linear kernel was trained on the same processed data, including binary risk indicators and one-hot encoded features. Compared to logistic regression, the SVM showed slightly lower overall accuracy but a higher ROC AUC score, indicating better overall separation between the two classes. It also achieved higher precision for identifying heart disease cases, though at the cost of some recall.

#### Random Forest Classifier:
 * Accuracy:
 0.88
 * ROC AUC: 0.94
 * Recall (Class 0-No heart disease): 0.90
 * Precision: (Class 0): 0.87
 * Recall (Class 1-Heart disease): 0.86
 * Precision (Class 1): 0.89
 * F1 (Class 1): 0.87
* Conclusion: The Random Forest Classifier was trained on processed data and included one-hot encoded features. It also included hyperparameter tuning which was done through
GridSearch. The Random Forest provided strong results across all metrics but particularly performed well with recall and a high AUC score. We can also see a boost in accuracy and ROC AUC scores compared to SVM. This indicated good separation between classes, and it proved to be a robust model in my project.

## How to reproduce results
   * Download the data from the UCI website (link can be found at the very bottom)
   * Jupyter notebook was what I used to code
   * Import the necessary libraries such as pandas, matplotlib, sckit-learn libraries, and various others which can be seen in the code
   * Run the files attached

### Future Work and limitations
I would like to explore other ensemble models which I felt could help improve recall for heart disease predicition. I also would like to incorporate external data, possibly adding more features such as family history, smoking status, exercise habits, all of which I feel can help improve the model. Overall, these models were trained on a small dataset which might limit generalizability, future work could incorporate larger datasets and more features to improve predicition accuracy. 

### Overview of files 
 * https://archive.ics.uci.edu/ml/datasets/heart+disease
 * Scikit-learn, for model training, evaluation, and hyperparameter tuning
   
 
  
