# Import necessary libraries
import boto3
import csv
from io import StringIO
from data_handling import testX, features

# Initialize the SageMaker runtime client
sagemaker_runtime = boto3.client('sagemaker-runtime')

# Specify your endpoint name
endpoint_name = 'Custom-sklearn-model-2024-05-14-13-00-21'

# Prepare the input data for prediction
input_data = testX[features][0:2].values.tolist()

# Convert the list of lists to a CSV string
csv_buffer = StringIO()
csv_writer = csv.writer(csv_buffer)
csv_writer.writerows(input_data)
csv_data = csv_buffer.getvalue()

# Set content type to 'text/csv'
content_type = 'text/csv'

# Make a prediction using the deployed model
response = sagemaker_runtime.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType=content_type,
    Body=csv_data.encode('utf-8')  # Encode the CSV string to bytes
)

# Read and decode the response
result = response['Body'].read().decode('utf-8')
print("Prediction:", result)
