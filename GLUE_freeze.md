# GLUE tasks

Here are the corresponding GLUE scores on BiGS if we freeze the kernel and token type embeddings.

# Experiments 

GLUE is made up of a total of 9 different tasks.

We finetune BiGS on TPU-v3 with 8 cores. Since the batch size per device is 2, the total number of batch size is 16.

```
export TASK_NAME=cola

python run_glue_freeze.py \
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
| CoLA  | Matthews corr                | 66.03 |
| SST-2 | Accuracy                     | 94.27 |
| QQP   | Accuracy/F1                  | 91.83/88.96 |
| MNLI  | Matched acc./Mismatched acc. | 86.14 |
| QNLI  | Accuracy                     | 91.54 |
| MRPC  | F1/Accuracy                  | 86.35/80.39 |
| STS-B | Pearson/Spearman corr.       | 89.05/89.04 |
| RTE   | Accuracy                     | 73.29 |



