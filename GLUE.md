# GLUE tasks

Here are the corresponding GLUE scores on BiGS.

# Experiments 

GLUE is made up of a total of 9 different tasks. Here is how to run the script on one of them:

We finetune BiGS on TPU-v3 with 8 cores. Since the batch size per device is 2, the total number of batch size is 16.

```
export TASK_NAME=cola

python run_glue.py \
    --model_name_or_path JunxiongWang/BiGS_512 \
    --task_name $TASK_NAME \
    --max_seq_length 512 \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --logging_steps 100 \
    --eval_steps 500 \
    --weight_decay 0.01 \
    --output_dir BiGS_$TASK_NAME/
```

Those give us the following result

| Task  | Metric                       | Result      |
|-------|------------------------------|-------------|
| CoLA  | Matthews corr                | 67.9 |
| SST-2 | Accuracy                     | 93.8 |
| QQP   | Accuracy/F1                  | 91.4/88.4 |
| MNLI  | Matched acc./Mismatched acc. | 86.5 |
| QNLI  | Accuracy                     | 91.6 |


For MRPC, STS-B and RTE, we finetune on the MNLI model

```
export TASK_NAME=mrpc

python run_glue.py \
    --model_name_or_path JunxiongWang/BiGS_512_MNLI \
    --task_name $TASK_NAME \
    --max_seq_length 512 \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --logging_steps 100 \
    --eval_steps 500 \
    --weight_decay 0.01 \
    --output_dir BiGS_$TASK_NAME/
```

| Task  | Metric                       | Result      |
|-------|------------------------------|-------------|
| MRPC  | F1/Accuracy                  | 88.3/83.3 |
| STS-B | Pearson/Spearman corr.       | 89.9/89.7 |
| RTE   | Accuracy                     | 79.8      |



