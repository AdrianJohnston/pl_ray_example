# pl_ray_example

Run Locally with:

```
XAMIN_RUN_LOCAL=1 AWS_DEFAULT_REGION='us-west-2' JOB_BUCKET="xamin-us-west-2-jobs" XAMIN_ORG_ID=$(uuidgen) XAMIN_USER_ID=$(uuidgen) XAMIN_JOB_ID=$(uuidgen) python train.py --model.rootpath=./ --model.dataset_folder=./data --model.network_backbone=LeNet --model.dataset_version=0 --model.dataset_name=MNIST --trainer.gpus 1 --trainer.max_epochs 10
```