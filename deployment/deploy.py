import boto3
from sagemaker import Model
from sagemaker import Session

region = 'eu-north-1'

boto3_session = boto3.Session(region_name=region)
sagemaker_session = Session(boto_session=boto3_session)

role = 'arn:aws:iam::641801338804:role/service-role/AmazonSageMaker-ExecutionRole-20240717T115438'

# TensorFlow 2.16.1 GPU image URI for the 'eu-north-1' region
image_uri = '641801338804.dkr.ecr.eu-north-1.amazonaws.com/fb-asr-tf:latest'

model = Model(
    model_data='s3://fb-asr/model_pt.tar.gz',
    role=role,
    image_uri=image_uri,
    sagemaker_session=sagemaker_session
)

predictor = model.deploy(
    instance_type='ml.g4dn.xlarge',
    initial_instance_count=1
)