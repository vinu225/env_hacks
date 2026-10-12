# Infrastructure & Cloud Deployment Architecture (AWS Target)

This directory is designated for AWS Infrastructure-as-Code (Terraform / AWS CDK) to support cloud deployment in subsequent phases.

## Target AWS Architecture

```
                                      Internet
                                         │
                                   [ Route 53 ]
                                         │
                                [ CloudFront CDN ]
                                         │
                       ┌─────────────────┴─────────────────┐
                       ▼                                   ▼
             [ S3: Static Frontend ]          [ Application Load Balancer ]
                                                           │
                                              [ AWS ECS Fargate / App Runner ]
                                              (FastAPI Modular Monolith)
                                                           │
                      ┌───────────────────────┬────────────┴───────────────┐
                      ▼                       ▼                            ▼
             [ Amazon DynamoDB ]      [ Amazon S3 ]               [ AWS Bedrock ]
             - Locations              - ML Model Artifacts        - Claude / Llama
             - Risk Predictions       - Raw Observations          - Natural Language
             - Community Reports                                    Advisories
                      ▲
                      │
             [ Amazon EventBridge ] ──► [ AWS Lambda Ingestion Worker ]
             (Hourly Weather Trigger)   (Open-Meteo / IMD Feed Ingest)
```

## Storage Decoupling & DynamoDB Single-Table Design

The backend uses repository abstractions (`LocationRepository`, `RiskRepository`, `ReportRepository`) that operate seamlessly against:
1. In-Memory store (used for local testing and CI/CD without cloud dependencies).
2. Amazon DynamoDB (designed for single-table access patterns):
   - **Locations Table**: Partition Key `PK = LOCATION#<id>`, Sort Key `SK = METADATA`.
   - **Risk Predictions Table**: Partition Key `PK = LOCATION#<id>`, Sort Key `SK = RISK#<category>#<timestamp>`.
   - **Community Reports Table**: Partition Key `PK = LOCATION#<id>`, Sort Key `SK = REPORT#<timestamp>`.
   - **ML Artifacts**: S3 Bucket `bharat-env-ml-artifacts/models/<category>/<version>/`.

## Environment Variables for Cloud

All AWS configurations are loaded through environment variables (or IAM Task Roles when deployed to ECS/App Runner). Local developers never need AWS credentials to run the full application suite.
