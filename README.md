# Description

See the report file for the full process.

## Goal

Given physical measurements (features) of a patient, predict his/her diabetic level.

## Data
The training data includes:

train.csv: 1 column indicating patient ID (column name: 'id'), 64 features columns (column name: 'Col 1', …, 'Col 64'), and 1 ground-truth diabetic level (column name: 'y') of 160 patients

The test data includes:

x_test.csv: 1 id column and 64 feature columns of 82 patients

## Task
Train a regression model (e.g., linear regression, Lasso, Ridge, etc) on training data and make prediction on test data.

## Submission
The prediction on test data, summarized it in y_test.csv file, with similar format as sample_submission.csv, i.e. 1 column indicating patient ID (column name: 'id') and 1 column showing the predicted diabetic level (column name: 'y'). Note the column 'id' should match 'id' in x_test.csv.

## Evaluation
The evaluation metric reported is MSE (Mean Square Error).
