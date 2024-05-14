
# SageMaker Mobile Price Classification Project

## Table of Contents
- [Introduction](#introduction)
- [Data Preparation](#data-preparation)
- [Model Training](#model-training)
- [Model Deployment](#model-deployment)
- [Prediction](#prediction)
- [Conclusion](#conclusion)
- [Future Work](#future-work)

## Introduction

This project aims to classify mobile phones into different price ranges using a RandomForest model. The project leverages Amazon SageMaker for training, deploying, and making predictions with the model.

## Data Preparation

The dataset used in this project contains various features of mobile phones, including battery power, clock speed, RAM, etc. The target variable is the price range, which categorizes phones into different price brackets.

### Data Handling

- The data is split into training and testing sets.
- The features and target variable are separated.
- Missing values are checked and handled appropriately.

## Model Training

The model used for classification is a RandomForestClassifier from scikit-learn. The training script (`train.py`) performs the following steps:

1. Parse command-line arguments for hyperparameters and data paths.
2. Load the training and testing data.
3. Train the RandomForest model.
4. Save the trained model to a specified directory.

### Hyperparameters

- `n_estimators`: Number of trees in the forest.
- `random_state`: Random seed for reproducibility.

## Model Deployment

The deployment script (`deploy.py`) performs the following steps:

1. Define the model and endpoint names using the current timestamp.
2. Create an `SKLearnModel` object with the trained model's S3 path.
3. Deploy the model to a SageMaker endpoint.
4. Print the endpoint name and details.

## Prediction

The prediction script (`predict.py`) performs the following steps:

1. Initialize the SageMaker runtime client.
2. Prepare the input data for prediction.
3. Convert the input data to a CSV string.
4. Invoke the endpoint to get predictions.
5. Print the prediction results.

## Conclusion

This project demonstrates how to use Amazon SageMaker for training, deploying, and making predictions with a machine learning model. The RandomForest model provides accurate classifications for mobile phone price ranges based on their features.

## Future Work

- Experiment with different machine learning models to improve accuracy.
- Implement feature engineering techniques to enhance model performance.
- Automate the entire workflow using AWS Step Functions or SageMaker Pipelines.
