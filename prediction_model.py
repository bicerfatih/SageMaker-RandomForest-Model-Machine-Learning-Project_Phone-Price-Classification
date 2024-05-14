from data_handling import testX, features

print(testX)
testX[features][0:2].values.tolist()
print(testX[features][0:2].values.tolist())
print(predictor.predict(testX[features][0:2].values.tolist()))


sagemaker = boto3.client('sagemaker-runtime')
response = sagemaker.invoke_endpoint(
    EndpointName='your-endpoint-name',
    Body=b'{"input_data2": [1, 2, 3]}',
    ContentType='application/json',
)
print(response['Body'].read().decode())

