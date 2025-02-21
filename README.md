# Predicting-Customer-Churn-for-SyriaTel
This project aims to build a classifier to predict whether a customer will ("soon") stop doing business with SyriaTel, a telecommunications company. This is a binary classification problem with 2 possible outcomes:

-The customer will soon stop doing business with SyriaTel.

-The customer will not stop doing business with SyriaTel.

# Tech Stack
-Python

-Pandas

-NumPy

-Matplotlib

-Seaborn

-Scikit-learn 

-DecisionTreeClassifier

-LogisticRegression

-GridSearchCV 

-StandardScaler

-Accuracy Score, Precision, Recall, F1-Score

-Confusion Matrix

-Classification Report

# Business Understanding
The loss of customers in a company such as SyriaTel means loss of revenue and increased costs since acquiring new customers is more expensive than retaining existing ones.

Predicting customer dropout in advance helps SyriaTel identify high-risk customers early therefore implementing cost-effective retention strategies leading to reduced revenue loss and improved customer satisfaction which leads to profitability to the telecommunication company.

## Key Business Questions
What are some of the factors that contribute to customer dropout?

How can SyriaTel leverage on these factors to come up with solutions?

# Data Understanding
The dataset consists of customer records from SyriaTel, a telecommunications company with features that help predict whether a customer will stop using the service.

This dataset was sourced from (https://www.kaggle.com/datasets/becksddf/churn-in-telecoms-dataset) with key features such as:
-Account length

-Area code

-Phone number

-Total day calls

-Total intl calls

-Customer service calls

-Number vmail messages

-International plan

-Total night minutes

Understanding these patterns helps in building a predictive model that assists SyriaTel in retaining valuable customers.

# Data Analysis
The most important features were visualized as follows;

![Alt text](https://github.com/Kipchumba254/Predicting-Customer-Churn-for-SyriaTel/blob/main/Screenshot%202025-02-21%20202416.png)

## Modeling
In this case, our baseline model was Logistic Regression. We obtained a better/ higher AUC in the model after solving class imbalance with SMOTE thus improving the model's performance on the minority class which in this case is churn.

![Alt text](https://github.com/Kipchumba254/Predicting-Customer-Churn-for-SyriaTel/blob/main/Screenshot%202025-02-21%20220745.png)

We also used a confusion matrix which was not suitable for identifying churn due to a poor f1-score indicating a poor performance of the model.

![Alt text](https://github.com/Kipchumba254/Predicting-Customer-Churn-for-SyriaTel/blob/main/Screenshot%202025-02-21%20213917.png)

GridSearch, a hyperparameter tuning technique in decision trees was used to developed a churn prediction model for SyriaTel achieving a high accuracy of 93.55%.

![Alt text](https://github.com/Kipchumba254/Predicting-Customer-Churn-for-SyriaTel/blob/main/Screenshot%202025-02-21%20222345.png)
Decision Tree Accuracy: 93.55%
Best Decision Tree Performance:
              precision    recall  f1-score   support

           0       0.95      0.97      0.96       566
           1       0.82      0.73      0.77       101

    accuracy                           0.94       667
   macro avg       0.89      0.85      0.87       667
weighted avg       0.93      0.94      0.93       667









