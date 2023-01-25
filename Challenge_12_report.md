# Module 12 Report Template

## Overview of the Analysis

In this section, describe the analysis you completed for the machine learning models used in this Challenge. This might include:

* Explain the purpose of the analysis.
The purpose of the analysis is to identify the creditworthiness of borrowers given a dataset of historical lending actitivy for a peer-to-peer lending services company. 
* Explain what financial information the data was on, and what you needed to predict.
The lending data was on a csv file and contained historical lending activity that included loan size, interest rate, borrower income, debt to income ratio, number of accounts, derogatory markts, total debt, loan size and loan status (high-risk or healthy loan). We needed to predict the loan status (high-risk = 0 or healthy loan = 1) of future borrowers given the information provided in the dataset (features,) excluding loan status which is what the model is trying to determine.

* Provide basic information about the variables you were trying to predict (e.g., `value_counts`).
Loan status (high-risk or healthy loan). We needed to predict the loan status (high-risk = 0 or healthy loan = 1) of future borrowers given the information provided in the dataset (features,) excluding loan status which is what the model is trying to determine. The value_counts function, for example, told us whether a loan was classified as 0 or 1 organized by loan size.

* Describe the stages of the machine learning process you went through as part of this analysis.
Step 1: Read the lending_data.csv data from the Resources folder into a Pandas DataFrame.
Step 2: Create the labels set (y) from the “loan_status” column, and then create the features (X) DataFrame from the remaining columns.
Step 3: Check the balance of the labels variable (y) by using the value_counts function.
Step 4: Split the data into training and testing datasets by using train_test_split.
* Briefly touch on any methods you used (e.g., `LogisticRegression`, or any resampling method).
First: Create a Logistic Regression Model with the Original Data
Step 1: Fit a logistic regression model by using the training data.
Step 2: Save the predictions on the testing data labels by using the testing feature data and the fitted model, and 
Step 3: Evaluate the model’s performance by doing a calculation of the accuracy score of the model, generated confusion matrix and generated a classification report.

Second: Predict a Logistic Regression Model with Resampled Training Data
Step 1: Use the RandomOverSampler module from the imbalanced-learn library to resample the data. Be sure to confirm that the labels have an equal number of data points.
Step 2: Use the LogisticRegression classifier and the resampled data to fit the model and make predictions.
Step 3: Evaluate the model’s performance by doing a calculation of the accuracy score of the model, generated confusion matrix and generated a classification report.

## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1:
  * Description of Model 1 Accuracy, Precision, and Recall scores.
                precision    recall  f1-score   support

           0       1.00      0.99      1.00     18745
           1       0.85      0.89      0.87       639

    accuracy                           0.99     19384
   macro avg       0.92      0.94      0.93     19384
weighted avg       0.99      0.99      0.99     19384
 

The precision is the ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false positives. The precision is intuitively the ability of the classifier not to label as positive a sample that is negative. The best value is 1 and the worst value is 0. The precision score is 100% for healthy loans and 0.85% for high-risk loans.

The recall is calculated as the ratio between the number of Positive samples correctly classified as Positive to the total number of Positive samples. The recall measures the model's ability to detect Positive samples. The higher the recall, the more positive samples detected. The recall is 99% for healthy loans and 0.89% for high-risk loans.

Accuracy is the number of predictions that the model accurately predicted (number of correct predictions / total number of predictions). The accuracy is 99%. 

* Machine Learning Model 2:
  * Description of Model 2 Accuracy, Precision, and Recall scores.
        precision    recall  f1-score   support

             precision    recall  f1-score   support

           0       1.00      0.99      1.00     18728
           1       0.85      0.91      0.88       656

    accuracy                           0.99     19384
   macro avg       0.92      0.95      0.94     19384
weighted avg       0.99      0.99      0.99     19384


Resampled data:
- The precision score is 100% for healthy loans and 0.85% for high-risk loans.
- The recall is 99% for healthy loans and 0.91% for high-risk loans.
- The accuracy is 99%. 
## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
* Which one seems to perform best? How do you know it performs best?
* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )
Yes, performance depends on the problem we are trying to solve. In this case, it is important to predict 0, which is the high-risk loans for a credit service company. The credit company would like to minimize the likelihood of a borrower defaulting, and thus would like to be able to predict, given a set of features of each borrower, whether the loan will be high-risk or healthy.

If you do not recommend any of the models, please justify your reasoning.
In this particular scenario, we are trying to build a model to accurately predict the 0 but because of the imbalanced data 18745 vs 639 in originl data, and 18728 healthy loans vs 656 high-risk loans in resampled data. The resampled data analysis does not offer a significant change in output and, as such, iIt requires additional testing models to ascertain accuracy.