#!/bin/bash
set -e

# Configuration
AWS_REGION="${AWS_REGION:-us-east-2}"
AWS_ACCOUNT_ID="${AWS_ACCOUNT_ID:-818273938349}"
ECR_REPO="seed-voken"
IMAGE_TAG="${IMAGE_TAG:-$(git rev-parse --short HEAD)}"

ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
FULL_IMAGE="${ECR_URI}/${ECR_REPO}:${IMAGE_TAG}"

echo "Building SEED-Voken Docker image..."
echo "  Image: ${FULL_IMAGE}"

# Login to ECR
echo "Logging in to ECR..."
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${ECR_URI}

# Create repository if it doesn't exist
echo "Ensuring ECR repository exists..."
aws ecr describe-repositories --repository-names ${ECR_REPO} --region ${AWS_REGION} 2>/dev/null || \
    aws ecr create-repository --repository-name ${ECR_REPO} --region ${AWS_REGION}

# Build the image
echo "Building Docker image..."
docker build -t ${FULL_IMAGE} -t ${ECR_URI}/${ECR_REPO}:latest .

# Push to ECR
echo "Pushing to ECR..."
docker push ${FULL_IMAGE}
docker push ${ECR_URI}/${ECR_REPO}:latest

echo ""
echo "Successfully built and pushed:"
echo "  ${FULL_IMAGE}"
echo "  ${ECR_URI}/${ECR_REPO}:latest"
echo ""
echo "Update k8s/seed-voken-training.yaml with this image tag if needed."
