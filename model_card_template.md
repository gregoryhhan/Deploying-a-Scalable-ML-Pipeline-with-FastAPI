# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

The purpose of the model is to determine whether a person makes over $50,000 or $50,000 and less per year. The model is a GradientBoostingClassifer, a machine learning algorithm that uses ensemble methods on multiple decision trees, along with GridSearchCV, a tool which uses cross-validation to find the optimal hyperparameters, to tune for our model. 

The optimal parameters used for our model were:
* max_depth: 7
* max_features: 'sqrt'
* min_samples_leaf: 4
* min_samples_split: 10
* n_estimators: 200

Model was saved as a pickle file to be accessed by the machine learning pipeline. 

## Intended Use

The intended use of the model is to predict the salary level of an individual based off of a given number of attributes. The model is also available for anyone who wishes to use it for academic, research, or sandbox purposes. 

## Training Data

https://archive.ics.uci.edu/dataset/20/census+income

The Census Income Dataset, sourced from the UCI Machine Learning Repository, contains a CSV file containing 32,561 rows and 15 columns. These columns include the target label "salary", 8 categorical features, and 6 numerical features. Detailed descriptions of each feature can be found at the provided UCI link. The target label "salary" consists of two classes ('<=50K', '>50K'), with a class imbalance ratio of approximately 75% to 25%. 

The dataset was given 80-20 split, dividing it into training and test sets. Stratification was applied to ensure balanced representation of classes within the target label "salary". For training purposes, the categorical features were encoded using a One Hot Encoder, while the target label was binarized using a label binarizer.

## Evaluation Data

20% percent of the dataset was reserved for evaluating the model's performance. Transformation was performed on the categorical features and the target label using the One Hot Encoder and label binarizer, which were fitted on the training set.

## Metrics

The classification performance was evaluated using the precision, recall and fbeta metrics.
The model achieves below scores using the test set:

* precision:0.7840, measuring the accuracy of the positive predictions
* recall:0.6722, measuring the model's ability to capture all positive instances
* fbeta:0.7238, measures the harmonic mean of the precision and recall

## Ethical Considerations

Because the model is able to predict the salary based on a few attributes, it cannot be taken 100% in its accuracy. There are a myriad of other factors that dictate how much a person is making which is not captured by the data. Furthermore, predictions are predictions and should not be assumed to generalize the salary level of different demographics and population categories. 

## Caveats and Recommendations

The dataset is sourced from 1994 and by today's standards is outdated. A more current source of data should be used to get updated data as well as salary levels that have changed due to inflation and state of the economy. Another recommendation is to look into other machine learning models and algorithms which might be able to handle and make better predictions based on categorical values. 