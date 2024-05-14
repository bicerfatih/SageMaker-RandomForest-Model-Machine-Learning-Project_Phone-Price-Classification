# Import necessary libraries
import sagemaker
import boto3

# Create a SageMaker client using boto3 to communicate with S3 bucket
sagemaker_runtime = boto3.client("sagemaker")

# Define the AWS region
region = 'eu-north-1'

# Create a SageMaker session
sess = sagemaker.Session()

# Define the S3 bucket name to store the data
bucket = 'sagemakerfbicer'

# Print the name of the bucket being used
print("Using bucket " + bucket)

# Define the S3 key prefix for storing training data
sk_prefix = "your-s3-bucket-key-prefix-here"

# Upload the training data to S3 and get the S3 URI for the training data
trainpath = sess.upload_data(
    path="data/train_v1.csv", bucket=bucket, key_prefix=sk_prefix
)

# Upload the testing data to S3 and get the S3 URI for the testing data
testpath = sess.upload_data(
    path="data/test_v1.csv", bucket=bucket, key_prefix=sk_prefix
)

# Print the S3 URIs for the training and testing data
print(trainpath)
print(testpath)
