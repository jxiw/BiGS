# Pre-training

## Pretrain corpus

Our pretraining text corpus can be find [here](https://drive.google.com/drive/folders/18JGg5DSSnnSNdezd2P7J7ZNJW19oHNuE?usp=sharing).

Each file is either a train shard or a test shard in the hdf5 format.
And in each shard, it contains a two dimension array and its shape is (5, number of examples in this shard).

```
inputs[0] is the input_ids, using the BERT bert-large-uncased tokenization.
inputs[1] is the input_masks, where 1 is for non-padding elements and 0 is for padding elements (it is also called attention_mask in BERT).
inputs[2] is the segment_ids, where 0 is the first segment and 1 is the second segment. Since we don’t consider the next sentence prediction, it is always 0.
inputs[3] is masked_lm_positions: positions that we mask, where 0 is for padding and should be ignored.
inputs[4] is masked_lm_ids: the true token values in those masked positions, where 0 is for padding and should be ignored.
```

Here is the [code](https://colab.research.google.com/drive/1xZS56HEkqguSdzofC6ZujO5odhYfUQaM?usp=sharing) to look at the first example in the train shard 0.

If you want to build you own training corpus, pleas refer to [academic-budget-bert](https://github.com/IntelLabs/academic-budget-bert/tree/main/dataset)

## GPU

You can first download this corpus to your local server.

After you download this corpus, you can use the following script to train BiGS with 100k steps.

```
python pretrain.py \
--output_dir=BiGS_100k/ \
--dataset_path=PATH_OF_YOUR_TRAIN_CORPUS/ \
--model_type=s4-bert \
--max_seq_length=128 \
--weight_decay=0.05 \
--per_device_train_batch_size=131 \
--per_device_eval_batch_size=131 \
--learning_rate=6e-4 \
--overwrite_output_dir \
--max_train_shards=3000 \
--adam_beta1=0.9 \
--adam_beta2=0.98 \
--logging_steps=100 \
--save_steps=5000 \
--eval_steps=8000 \
--n_layers=23 \
--warmup_steps=2000 \
--num_total_train_steps=100000 \
--hidden_dropout_prob=0.1 \
--hidden_size=1024 \
--intermediate_size=3072 \
--schedule_type=cosine
```

## TPU

Due to the limited disk space on TPU, you need to store your data into a Google Cloud bucket.

Then you can train BiGS using the following script for example.

And please change ```from bigs_dataset.pretraining_dataset import PreTrainingDataset``` in pretrain.py to ```from bigs_dataset.pretraining_dataset_gcb import PreTrainingDataset```

```
python pretrain.py \
--output_dir=BiGS_100k/ \
--dataset_path=BUCKET_NAME/CORPUS_FOLDER \
--model_type=s4-bert \
--max_seq_length=128 \
--weight_decay=0.05 \
--per_device_train_batch_size=131 \
--per_device_eval_batch_size=131 \
--learning_rate=6e-4 \
--overwrite_output_dir \
--max_train_shards=3000 \
--adam_beta1=0.9 \
--adam_beta2=0.98 \
--logging_steps=100 \
--save_steps=5000 \
--eval_steps=8000 \
--n_layers=23 \
--warmup_steps=2000 \
--num_total_train_steps=100000 \
--hidden_dropout_prob=0.1 \
--hidden_size=1024 \
--intermediate_size=3072 \
--schedule_type=cosine
```

It roughly takes 3 days with a single TPU-v3. 
