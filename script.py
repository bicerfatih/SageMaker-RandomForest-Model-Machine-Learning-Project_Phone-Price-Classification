# Import necessary libraries
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score
import joblib
import sklearn
import argparse


# Function to load the model from the specified directory
def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf


if __name__ == "__main__":
    print("Extracting arguments")

    # Parse command-line arguments for hyperparameters and data paths
    parser = argparse.ArgumentParser()

    # Hyperparameters sent by the client
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--random_state", type=int, default=0)

    # Data, model, and output directories
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    parser.add_argument('--train-file', type=str, default="train_v1.csv")
    parser.add_argument('--test-file', type=str, default="test_v1.csv")

    args, _ = parser.parse_known_args()

    # Print versions of libraries
    print("SKLearn Version: ", sklearn.__version__)
    print("Joblib Version: ", joblib.__version__)

    print("[INFO] Reading data")

    # Print paths to training and testing data
    print("Train path:", args.train)
    print("Train file:", args.train_file)
    print("Test path:", args.test)
    print("Test file:", args.test_file)

    # Load the training and test data
    train_df = pd.read_csv(os.path.join(args.train, args.train_file))
    test_df = pd.read_csv(os.path.join(args.test, args.test_file))

    # Define the target and features
    features = list(train_df.columns)
    label = features.pop(-1)

    print("Building training and testing datasets")

    X_train = train_df[features]
    X_test = test_df[features]
    y_train = train_df[label]
    y_test = test_df[label]

    print('Column order:')
    print(features)

    print("Label column is:", label)

    print("Data Shape:")
    print()
    print("... - SHAPE OF TRAINING DATA (85%)")
    print(X_train.shape)
    print(y_train.shape)

    print("... - SHAPE OF TESTING DATA (15%)")
    print(X_test.shape)
    print(y_test.shape)

    print("Training RandomForest Model....")
    model = RandomForestClassifier(n_estimators=args.n_estimators, random_state=args.random_state, verbose=0)
    model.fit(X_train, y_train)

    # Save the trained model to the specified directory
    model_path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, model_path)
    print("Model persisted at " + model_path)

    # Make predictions on the test data
    y_pred_test = model.predict(X_test)

    # Calculate accuracy and generate a classification report
    test_acc = accuracy_score(y_test, y_pred_test)
    test_rep = classification_report(y_test, y_pred_test)

    print("Metrics results for testing data")

    print("Total Rows are:", X_test.shape[0])
    print('[TESTING] Model Accuracy is:', test_acc)
    print('[TESTING] Testing Report:', test_rep)
