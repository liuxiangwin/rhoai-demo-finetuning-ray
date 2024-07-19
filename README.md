# RHOAI Demo Fine-tuning
This repository contains a demonstration of fine-tuning using RHOAI with Codeflare and Ray.
### Prerequisites
- A GPU-enabled RHOAI platform
- A data connection for the workbench is required. I am using MinIO and you could try the minio.yaml for your deployment and set the data connection to something like the following:
  - Access key: minio
  - Secret key: minio123
  - Endpoint: http://minio-api.minio.svc.cluster.local:9000
- Enable the Codeflare and Ray operators by running the following commands:
```bash
oc patch dsc/default-dsc --type merge -p '{"spec":{"components":{"codeflare":{"managementState": "Managed"}}}}'
oc patch dsc/default-dsc --type merge -p '{"spec":{"components":{"ray":{"managementState": "Managed"}}}}'
```
### Tested Version
The following versions have been tested:
- rhods-operator: 2.10.0
- Standard Data Science Workbench: Version 2024.1
- GPU: Nvidia Tesla V100-SXM2-16GB
## Instruction
To get started, clone this repository and use the provided Jupyter Notebooks:
- **ray_job_submission.ipynb:** for distribution fine-tuning demo
- **local_fine_tuning.ipynb:** for local fine-tuning demo