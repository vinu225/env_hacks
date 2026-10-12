# AWS Production Deployment Guide: Heat-Risk FastAPI on ECS Fargate

## Architecture Overview

```
                      Client / Demonstration Browser
                                   │
                                   ▼ HTTP (Port 8000)
                     [ Public IP / Application Load Balancer ]
                                   │
                     ┌─────────────▼─────────────┐
                     │   AWS ECS Fargate Cluster │
                     │ ┌───────────────────────┐ │
                     │ │  heat-risk-api        │ │
                     │ │  - FastAPI Uvicorn    │ │
                     │ │  - LightGBM Model v1  │ │
                     │ │  - Port 8000 (Non-root│ │
                     │ └───────────────────────┘ │
                     └───────────────────────────┘
                                   │
                     ┌─────────────┴─────────────┐
                     ▼                           ▼
            [ Amazon CloudWatch ]       [ Amazon ECR Registry ]
            (/ecs/heat-risk-api)        (heat-risk-api:latest)
```

---

## 1. Prerequisites

Before running deployment commands, ensure you have:
1. An AWS Account with Administrator or DevOps permissions.
2. AWS CLI installed and configured:
   ```bash
   aws configure
   ```
3. Docker or Docker Desktop running locally.

### Configurable Placeholders
Replace these placeholders across the commands below:
- `<AWS_REGION>`: e.g., `ap-south-1` (Mumbai) or `us-east-1` (N. Virginia)
- `<ACCOUNT_ID>`: Your 12-digit AWS Account ID (retrieve via `aws sts get-caller-identity --query Account --output text`)
- `<VPC_ID>`: Your default or demo VPC ID
- `<SUBNET_1>`, `<SUBNET_2>`: Public Subnet IDs in your VPC

---

## 2. Step-by-Step Deployment Instructions

### Step 2.1 — Create Amazon ECR Repository
```bash
aws ecr create-repository \
  --repository-name heat-risk-api \
  --region <AWS_REGION> \
  --image-scanning-configuration scanOnPush=true
```

### Step 2.2 — Build, Tag, and Push Docker Image to ECR
Authenticate Docker with your ECR private registry:
```bash
aws ecr get-login-password --region <AWS_REGION> | docker login --username AWS --password-stdin <ACCOUNT_ID>.dkr.ecr.<AWS_REGION>.amazonaws.com
```

Build the multi-platform or Linux AMD64 image matching ECS Fargate:
```bash
docker build --platform linux/amd64 -t heat-risk-api:latest .
```

Tag and push to ECR:
```bash
docker tag heat-risk-api:latest <ACCOUNT_ID>.dkr.ecr.<AWS_REGION>.amazonaws.com/heat-risk-api:latest
docker push <ACCOUNT_ID>.dkr.ecr.<AWS_REGION>.amazonaws.com/heat-risk-api:latest
```

---

### Step 2.3 — Create CloudWatch Log Group
```bash
aws logs create-log-group \
  --log-group-name /ecs/heat-risk-api \
  --region <AWS_REGION>
```

---

### Step 2.4 — Create IAM Task Execution Role (If Not Already Present)
Create `ecsTaskExecutionRole` with standard permissions to pull from ECR and write to CloudWatch:
```bash
# 1. Create trust policy
cat <<EOF > /tmp/ecs-tasks-trust-policy.json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": { "Service": "ecs-tasks.amazonaws.com" },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

# 2. Create the IAM role
aws iam create-role \
  --role-name ecsTaskExecutionRole \
  --assume-role-policy-document file:///tmp/ecs-tasks-trust-policy.json

# 3. Attach AWS managed execution policy
aws iam attach-role-policy \
  --role-name ecsTaskExecutionRole \
  --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy
```

---

### Step 2.5 — Create Security Group for ECS Fargate Task
```bash
# Create Security Group
SG_ID=$(aws ec2 create-security-group \
  --group-name heat-risk-api-sg \
  --description "Allow inbound traffic on port 8000 for Heat-Risk API demo" \
  --vpc-id <VPC_ID> \
  --query 'GroupId' \
  --output text)

# Allow inbound port 8000 from public or your IP
aws ec2 authorize-security-group-ingress \
  --group-id $SG_ID \
  --protocol tcp \
  --port 8000 \
  --cidr 0.0.0.0/0
```

---

### Step 2.6 — Create ECS Cluster & Register Task Definition
Create ECS cluster:
```bash
aws ecs create-cluster \
  --cluster-name heat-risk-cluster \
  --region <AWS_REGION>
```

Register task definition using the project template:
```bash
# Substitute Account ID and Region into task definition template
sed -e 's/<ACCOUNT_ID>/'"<ACCOUNT_ID>"'/g' \
    -e 's/<AWS_REGION>/'"<AWS_REGION>"'/g' \
    infra/ecs-task-definition.json > /tmp/task-def-ready.json

aws ecs register-task-definition \
  --cli-input-json file:///tmp/task-def-ready.json \
  --region <AWS_REGION>
```

---

### Step 2.7 — Launch ECS Fargate Service
Launch with public IP assignment for direct demonstration:
```bash
aws ecs create-service \
  --cluster heat-risk-cluster \
  --service-name heat-risk-service \
  --task-definition heat-risk-api-task \
  --desired-count 1 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[<SUBNET_1>,<SUBNET_2>],securityGroups=[$SG_ID],assignPublicIp=ENABLED}" \
  --region <AWS_REGION>
```

---

## 3. Verify Public Deployment

### Retrieve Public IP Address
```bash
# 1. Get running Task ARN
TASK_ARN=$(aws ecs list-tasks --cluster heat-risk-cluster --service-name heat-risk-service --query 'taskArns[0]' --output text)

# 2. Get Network Interface ID
ENI_ID=$(aws ecs describe-tasks --cluster heat-risk-cluster --tasks $TASK_ARN --query 'tasks[0].attachments[0].details[?name==`networkInterfaceId`].value' --output text)

# 3. Get Public IP
PUBLIC_IP=$(aws ec2 describe-network-interfaces --network-interface-ids $ENI_ID --query 'NetworkInterfaces[0].Association.PublicIp' --output text)

echo "Application is live at: http://$PUBLIC_IP:8000"
```

### Endpoints Smoke Test
```bash
# Liveness Health Check
curl -i http://$PUBLIC_IP:8000/api/v1/health

# Readiness Check
curl -i http://$PUBLIC_IP:8000/api/v1/ready

# Live ML Heat Prediction for Delhi
curl -s http://$PUBLIC_IP:8000/api/v1/locations/loc_delhi_central/heat | jq .
```

---

## 4. Operational Monitoring & Service Updates

### Inspect Live CloudWatch Logs
```bash
aws logs tail /ecs/heat-risk-api --follow --region <AWS_REGION>
```

### Deploying a New Image / Update
```bash
# 1. Build and push new image
docker build --platform linux/amd64 -t heat-risk-api:latest .
docker tag heat-risk-api:latest <ACCOUNT_ID>.dkr.ecr.<AWS_REGION>.amazonaws.com/heat-risk-api:latest
docker push <ACCOUNT_ID>.dkr.ecr.<AWS_REGION>.amazonaws.com/heat-risk-api:latest

# 2. Force new deployment on ECS
aws ecs update-service \
  --cluster heat-risk-cluster \
  --service heat-risk-service \
  --force-new-deployment \
  --region <AWS_REGION>
```

---

## 5. Teardown / Resource Cleanup (Avoid AWS Charges)

Run after the hackathon demonstration to terminate all provisioned resources:

```bash
# 1. Scale down and delete ECS service
aws ecs update-service --cluster heat-risk-cluster --service heat-risk-service --desired-count 0 --region <AWS_REGION>
aws ecs delete-service --cluster heat-risk-cluster --service heat-risk-service --region <AWS_REGION>

# 2. Delete ECS Cluster
aws ecs delete-cluster --cluster heat-risk-cluster --region <AWS_REGION>

# 3. Delete Security Group
aws ec2 delete-security-group --group-id $SG_ID

# 4. Delete ECR Repository & images
aws ecr delete-repository --repository-name heat-risk-api --force --region <AWS_REGION>

# 5. Delete CloudWatch Log Group
aws logs delete-log-group --log-group-name /ecs/heat-risk-api --region <AWS_REGION>
```
