# SageMaker RandomForest Model Machine Learning Project - Phone Price Classification 

This repository contains the code for a machine learning project that uses Amazon SageMaker to train, deploy, and make predictions with a RandomForest model for mobile phone price classification.

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Setup](#setup)
- [Usage](#usage)
- [License](#license)

## Overview

The goal of this project is to classify mobile phones into different price ranges using a RandomForest model. The project involves the following steps:
1. Data preparation and preprocessing.
2. Training a RandomForest model using SageMaker.
3. Deploying the trained model to a SageMaker endpoint.
4. Making predictions using the deployed model.
5. Cleaning up the deployed endpoint.

## Project Structure

```
.
├── data_handling.py
├── deploy.py
├── predict.py
├── train.py
├── README.md
└── PROJECT.md
```

- `data_handling.py`: Script for handling data preprocessing.
- `deploy.py`: Script for deploying the trained model to a SageMaker endpoint.
- `predict.py`: Script for making predictions using the deployed model.
- `train.py`: Script for training the RandomForest model.
- `README.md`: This file.
- `PROJECT.md`: Detailed project report.

## Requirements

- Python 3.x
- boto3
- pandas
- scikit-learn
- joblib
- argparse
- SageMaker Python SDK

## Setup

1. **Clone the repository**:
    ```sh
    git clone https://github.com/yourusername/your-repository-name.git
    cd your-repository-name
    ```

2. **Install dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

3. **Set up AWS credentials**:
    Ensure you have configured your AWS credentials by running:
    ```sh
    aws configure
    ```

## Usage

### Training the Model
To train the RandomForest model, run the `train.py` script:
```sh
python train.py
```

### Deploying the Model
To deploy the trained model to a SageMaker endpoint, run the `deploy.py` script:
```sh
python deploy.py
```
### Making Predictions
To make predictions using the deployed model, run the `predict.py` script:
```sh
python predict.py
```
### Cleaning Up
To delete the deployed endpoint, uncomment the line in the `deploy.py` script that deletes the endpoint.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
