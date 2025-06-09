#!/bin/bash

# Exit on error
set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

# Configuration
NAMESPACE="automation"
REGISTRY="your-registry.com"
IMAGE_TAG="latest"

echo -e "${GREEN}Starting Kubernetes deployment...${NC}"

# Create namespace if it doesn't exist
echo "Creating namespace..."
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# Build and push Docker image
echo "Building and pushing Docker image..."
docker build -t $REGISTRY/automation:$IMAGE_TAG .
docker push $REGISTRY/automation:$IMAGE_TAG

# Update image in deployment
echo "Updating deployment configuration..."
sed -i "s|image: automation:latest|image: $REGISTRY/automation:$IMAGE_TAG|" kubernetes/deployment.yaml

# Apply Kubernetes configurations
echo "Applying Kubernetes configurations..."
kubectl apply -f kubernetes/deployment.yaml

# Wait for deployment to be ready
echo "Waiting for deployment to be ready..."
kubectl rollout status deployment/automation -n $NAMESPACE

# Check service health
echo "Checking service health..."
SERVICE_URL=$(kubectl get ingress automation -n $NAMESPACE -o jsonpath='{.spec.rules[0].host}')
if curl -s https://$SERVICE_URL/health | grep -q "healthy"; then
    echo -e "${GREEN}Application is healthy${NC}"
else
    echo -e "${RED}Application health check failed${NC}"
    exit 1
fi

echo -e "${GREEN}Deployment completed successfully!${NC}"
echo "Access the application at:"
echo "- Web Interface: https://$SERVICE_URL"
echo "- Monitoring Dashboard: https://$SERVICE_URL/dashboard"

# Show deployment status
echo -e "\n${GREEN}Deployment status:${NC}"
kubectl get pods -n $NAMESPACE
kubectl get services -n $NAMESPACE
kubectl get ingress -n $NAMESPACE

# Show logs
echo -e "\n${GREEN}Showing application logs...${NC}"
kubectl logs -f deployment/automation -n $NAMESPACE 