# Fintech_Challenge_12
In this challenge, we created a Credit Risk Classification model that can identify the creditworthiness of borrowers from a dataset of historical lending activity.

The following modules and libraries were used:

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import balanced_accuracy_score

STEP 1: we split the Data into Training and Testing Sets
![image](https://user-images.githubusercontent.com/111457110/213896183-1bafd18f-0fc6-464e-bf0d-af6556e94ef9.png)

STEP 2: we created the labels set (y) from the “loan_status” column, and then create the features (X) DataFrame from the remaining columns.
![image](https://user-images.githubusercontent.com/111457110/213896208-0601e9b0-75ac-47dd-a403-b62f6ff688a0.png)

STEP 3: we cheked the balance of the labels variable (y) by using the value_counts function.
![image](https://user-images.githubusercontent.com/111457110/213896268-56148270-20e3-4532-a285-d498f6cb512d.png)

STEP 4: we split the data into training and testing datasets by using train_test_split.
![image](https://user-images.githubusercontent.com/111457110/213896284-9a2b205c-ede2-430c-9b09-ca889590bf44.png)

We then created a Logistic Regression Model with the Original Data and printed the classification report for the data
![image](https://user-images.githubusercontent.com/111457110/213896311-4a76afe7-a9be-423f-aa32-c050bb61bf6e.png)

By reading the scores, we determined that due to the large imbalance between the 0 (healthy loan) and 1 (high-risk loan) was so large, that we needed to resamble the data and oversize the high-risk loan data to see if the predictions were more balanced and thus accurate to use.
![image](https://user-images.githubusercontent.com/111457110/213896434-8c579895-0081-4f1e-b3f6-3c065d68776c.png)
