# Heart Disease Project
### Introduction
The ability to predict heart disease accurately is vital for preventive healthcare. The primary
research question for our project is: How can machine learning techniques be used to predict
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
  
