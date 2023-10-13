# GLUE tasks

Here are the corresponding GLUE scores on BiGS if we freeze token type embeddings in Pytorch.

# Experiments 

GLUE is made up of a total of 9 different tasks, we finetune BiGS on a single 24G titanrtx.

```
export TASK_NAME=cola

python run_glue_pytorch_freeze.py \
  --model_name_or_path JunxiongWang/BiGS_128 \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 16 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --overwrite_output_dir \
  --output_dir BiGS_128_$TASK_NAME/
  --ignore_mismatched_sizes \
  --save_total_limit 2 \
  --output_dir BiGS_128_$TASK_NAME/
```

Those give us the following result

Using `pykeops` package

| Task  | Metric                       | Result      |
|-------|------------------------------|-------------|
| CoLA  | Matthews corr                | 69.37|
| MRPC  | Accuracy/F1                  | 80.64/86.36|
| STS-B | Pearson/Spearman corr.       | 89.10/88.98|
| RTE   | Accuracy                     | 69.68|

Without `pykeops` package

| Task  | Metric                       | Result      |
|-------|------------------------------|-------------|
| CoLA  | Matthews corr                | 65.46|
| MRPC  | Accuracy/F1                  | 81.86/87.29|
| STS-B | Pearson/Spearman corr.       | 88.96/88.95|
| RTE   | Accuracy                     | 69.31|