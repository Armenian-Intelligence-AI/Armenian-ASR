# Deployment Guide (STT SageMaker)

This guide provides step-by-step instructions for deploying your model using Docker, AWS ECR, and AWS SageMaker.

## 1. Docker Image Preparation

### 1.1. Build the Docker Image

Build the Docker image using the Dockerfile in your project directory:

```sh
docker build --platform linux/amd64 --build-arg CACHEBUST=$(date +%s) -t fb-asr-tf:latest .
```

### 1.2. Tag the Docker Image

Tag the Docker image for your AWS ECR repository:

```sh
docker tag fb-asr-tf:latest 641801338804.dkr.ecr.eu-north-1.amazonaws.com/fb-asr-tf:latest
```

### 1.3. Login to AWS ECR

Authenticate your Docker CLI to your Amazon ECR registry:

```sh
aws ecr get-login-password --region eu-north-1 | docker login --username AWS --password-stdin 641801338804.dkr.ecr.eu-north-1.amazonaws.com
```

### 1.4. Push the Docker Image to ECR

Push your Docker image to the ECR repository:

```sh
docker push 641801338804.dkr.ecr.eu-north-1.amazonaws.com/fb-asr-tf:latest
```

## 2. Preparing Model Artifacts

### 2.1. Navigate to the Deployment Directory

Change to the directory containing your model artifacts:

```sh
cd deploy_data
```

### 2.2. Create a `.tar.gz` Archive

Archive the necessary files into `model_pt.tar.gz`:

```sh
tar -czvf model_pt.tar.gz whisper_arm_stt audio_classifier_model ner_model
```

### 2.3. Upload to S3

Upload the `model_pt.tar.gz` file to fb-asr S3 bucket

## 3. Deploying the Model

Run the deployment script to deploy your model on SageMaker:

```sh
python deploy.py
```


## Note:

`deploy_data` folder should include 2 folders, whisper_arm_stt ner_model, they are located in fb-asr S3 bucket