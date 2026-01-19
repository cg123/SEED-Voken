# Kubernetes Deployment Guide for SEED-Voken

This guide covers deploying SEED-Voken training on your 4-node 8xH100 cluster.

## Prerequisites

- Kubeflow PyTorchJob operator installed
- ECR access configured
- Kubernetes secrets with HuggingFace and WandB tokens
- ImageNet data available (S3 or local)

## Quick Start

### 1. Build and Push Docker Image

```bash
# Set your AWS region if different
export AWS_REGION=us-east-2

# Build and push to ECR
./k8s/build-and-push.sh
```

This creates an image at `818273938349.dkr.ecr.us-east-2.amazonaws.com/seed-voken:latest`

### 2. Prepare ImageNet Data

**Option A: Download to each node's local SSD**

Edit `k8s/download-imagenet.yaml` and set your S3 path:
```yaml
- name: IMAGENET_S3_PATH
  value: "s3://your-bucket/imagenet/ILSVRC/"
```

Then deploy:
```bash
kubectl apply -f k8s/download-imagenet.yaml

# Wait for downloads (check all nodes)
kubectl get pods -l app=imagenet-downloader
kubectl logs -l app=imagenet-downloader -f
```

**Option B: Use HuggingFace streaming dataset**

Update `configs/k8s/imagenet_16k_distributed.yaml` to use:
```yaml
train:
  target: src.IBQ.data.huggingface.HuggingFaceStreamingDatasetTrain
  params:
    config:
      dataset_name: "imagenet-1k"
      size: 256
      token: "${HF_TOKEN}"
```

### 3. Deploy etcd (if not already running)

```bash
kubectl apply -f k8s/etcd.yaml
```

Verify it's running:
```bash
kubectl get pods -l app=etcd
```

### 4. Start Training

```bash
kubectl apply -f k8s/seed-voken-training.yaml
```

### 5. Monitor Training

**Watch logs:**
```bash
# All workers
kubectl logs -l app=seed-voken -f

# Specific worker
kubectl logs seed-voken-ibq-16k-worker-0 -f
```

**Check pod status:**
```bash
kubectl get pods -l app=seed-voken
```

**WandB:** Check https://wandb.ai/your-username/seed-voken

### 6. Access Checkpoints

Checkpoints are saved to S3:
```bash
aws s3 ls s3://torchtitan-us-west-2-ablations/seed-voken/ibq-16k/checkpoints/
```

Download latest checkpoint:
```bash
aws s3 cp s3://torchtitan-us-west-2-ablations/seed-voken/ibq-16k/checkpoints/last.ckpt .
```

## Configuration Details

### Training Config

The distributed training config is at `configs/k8s/imagenet_16k_distributed.yaml`:

- **Nodes:** 4
- **GPUs per node:** 8
- **Total GPUs:** 32
- **Batch size per GPU:** 48
- **Global batch size:** 1536
- **Mixed precision:** 16-bit (fp16)

### Customization

**Change batch size:**
```yaml
data:
  init_args:
    batch_size: 32  # per GPU
```

**Change S3 checkpoint path:**
```yaml
trainer:
  callbacks:
    - class_path: lightning.pytorch.callbacks.ModelCheckpoint
      init_args:
        dirpath: "s3://your-bucket/your-path/"
```

**Disable torch.compile (if issues arise):**
```yaml
model:
  init_args:
    compile_model: false
```

**Change number of nodes:**

Edit `k8s/seed-voken-training.yaml`:
```yaml
spec:
  elasticPolicy:
    minReplicas: 2
    maxReplicas: 2
  pytorchReplicaSpecs:
    Worker:
      replicas: 2
```

And update the config:
```yaml
trainer:
  num_nodes: 2
```

## Troubleshooting

### Pods stuck in Pending
```bash
kubectl describe pod <pod-name>
```
Common causes: GPU resources unavailable, image pull errors, PVC not mounted

### Training crashes immediately
Check logs:
```bash
kubectl logs <pod-name>
```

### NaN losses
The model now has NaN protection that skips bad updates. Watch for warnings:
```
WARNING: disc_loss is NaN/Inf at step 1234 (count: 1)
```

If you see this repeatedly (>10 times), training will stop automatically. This usually indicates:
- Learning rate too high
- Bad initialization
- Data corruption

### Check distributed setup
Look for these in logs:
```
Initializing distributed: GLOBAL_RANK: 0, MEMBER: 1/32
```

All 32 ranks should initialize properly.

## Performance Optimizations Enabled

- ✅ `torch.compile()` on encoder, decoder, discriminator
- ✅ Mixed precision (fp16)
- ✅ Persistent workers
- ✅ Prefetch factor = 2
- ✅ cuDNN benchmark autotuning
- ✅ Drop last batch for consistent sizes
- ✅ EFA for fast inter-node communication

## Cleanup

```bash
# Stop training
kubectl delete pytorchjob seed-voken-ibq-16k

# Remove ImageNet downloader
kubectl delete daemonset imagenet-downloader

# Remove etcd (if not used by other jobs)
kubectl delete -f k8s/etcd.yaml
```
