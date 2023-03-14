Most of code is from [academic-budget-bert](https://github.com/IntelLabs/academic-budget-bert)

Here are PreTrainingDataset classes **if you want to download text corpus locally and train BiGS with it.**

# Pretrain corpus

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

If you want to build you own training corpus, pleas refer to https://github.com/IntelLabs/academic-budget-bert/tree/main/dataset 