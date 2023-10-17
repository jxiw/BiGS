# GLUE tasks

Here are the corresponding GLUE scores on BiGS in Pytorch.

# Experiments 

GLUE is made up of a total of 9 different tasks, we finetune BiGS on a single 24G titanrtx.

```
export TASK_NAME=cola

python run_glue_pytorch.py \
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

Without `pykeops` package

| Task  | Metric                       | Result      |
|-------|------------------------------|-------------|
| CoLA  | Matthews corr                |65.5|
| SST-2 | Accuracy                     |93.1|
| QQP   | Accuracy/F1                  |91.2/88.2|
| MNLI  | Matched acc./Mismatched acc. |86.0|
| QNLI  | Accuracy                     |90.9|

Notice that, our Pytorch MNLI models are port from JAX models. So it has higher accuracy 86.4.

For MRPC, STS-B and RTE, we finetune on the MNLI model

```
export TASK_NAME=cola

python run_glue_pytorch.py \
  --model_name_or_path JunxiongWang/BiGS_128_MNLI \
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

Without `pykeops` package

| Task  | Metric                       | Result      |
|-------|------------------------------|-------------|
| MRPC  | Accuracy/F1                  |82.4/87.5|
| STS-B | Pearson/Spearman corr.       |89.8/89.9|
| RTE   | Accuracy                     |78.3|

Using `pykeops` package

| Task  | Metric                       | Result      |
|-------|------------------------------|-------------|
| MRPC  | Accuracy/F1                  |83.3/88.1|
| STS-B | Pearson/Spearman corr.       |89.8/89.9|
| RTE   | Accuracy                     |80.1|



