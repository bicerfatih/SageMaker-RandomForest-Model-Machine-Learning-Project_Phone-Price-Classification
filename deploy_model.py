# Import necessary libraries
from sagemaker.sklearn.model import SKLearnModel
from time import gmtime, strftime
from data_handling import testX, features

# Define the framework version for scikit-learn
FRAMEWORK_VERSION = "0.23-1"

# Generate a unique model name using the current timestamp
model_name = "Custom-sklearn-model-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())

# Create an SKLearnModel object
model = SKLearnModel(
    name=model_name,
    model_data="s3://sagemaker-eu-north-1-471112704729/custom-sklearn8-2024-05-14-11-17-29-125/output/model.tar.gz",
    role="arn:aws:iam::471112704729:role/SageMaker_1",
    entry_point="script.py",
    framework_version=FRAMEWORK_VERSION,
)

# Print the model details
print("Model is deployed")
print(model)
print(model_name)

# Generate a unique endpoint name using the current timestamp
endpoint_name = "Custom-sklearn-model-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
print("EndpointName={}".format(endpoint_name))

# Deploy the model to an endpoint
predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.xlarge",
    endpoint_name=endpoint_name,
)

# Print the predictor details
print(predictor)

# Prepare the input data for prediction
input_data = testX[features][0:2].values.tolist()
print(input_data)

# Make predictions using the deployed model
predictions = predictor.predict(input_data)
print(predictions)

## Deleting the endpoint (Uncomment the line below to delete the endpoint)
# sm_boto3.delete_endpoint(EndpointName=endpoint_name)
