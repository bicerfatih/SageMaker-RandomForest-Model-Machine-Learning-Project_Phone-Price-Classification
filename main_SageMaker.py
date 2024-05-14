# Import SKLearn estimator from SageMaker library
from sagemaker.sklearn.estimator import SKLearn

# Import custom module for data paths and boto3
from dataSender_S3bucket import trainpath, testpath, sagemaker_runtime
import boto3

# Define the framework version for scikit-learn
FRAMEWORK_VERSION = "0.23-1"

# Create an SKLearn estimator object
sklearn_estimator = SKLearn(
    entry_point="script.py",  # Specify the entry point script for training
    role="arn:aws:iam::471112704729:role/SageMaker_1",  # IAM role for SageMaker
    instance_count=1,  # Number of instances for training
    instance_type="ml.m5.large",  # Type of instance to use
    framework_version=FRAMEWORK_VERSION,  # Framework version to use
    base_job_name="custom-sklearn8",  # Base name for the training job
    hyperparameters={
        "n_estimators": 100, # Hyperparameter for number of estimators
        "random state": 0 # Hyperparameter for random state
    },
    use_spot_instances=True,  # Use spot instances to save cost
    max_wait=7200,  # Maximum wait time in seconds for spot instances
    max_run=3600  # Maximum run time in seconds for the training job
)

# Launch the training job
sklearn_estimator.fit({"train": trainpath, "test": testpath}, wait=True)

# Wait for the training job to complete and disable logs
sklearn_estimator.latest_training_job.wait(logs="None")

# Retrieve and print the location of the model artifact in S3
artifact = sagemaker_runtime.describe_training_job(
    TrainingJobName=sklearn_estimator.latest_training_job.name)["ModelArtifacts"]["S3ModelArtifacts"]

print("Model artifact persisted at " + artifact)
