# BiGS (Pretraining Without Attention)

<img width="537" alt="BiGS" src="https://user-images.githubusercontent.com/16102460/221464744-06b6538a-7e84-4c95-909f-239eab1dba71.png">

This repository contains a library for Bidirectional Gated State Space Model (BiGS). BiGS combines SSM layers with a multiplicative gating architecture that has been effective in simplified sequence modeling architectures. 

So far,  BiGS is able to match BERT accuracy on GLUE while BiGS is a **linear** space/memory model. 

# Released Models

Sentence length: 128

|**Training Tokens**|**Model**|
|----------|----------|
|~11B|[https://drive.google.com/drive/folders/1-nhzeWVgpXwMyNEQ5j-MwJxSzwKyT2an?usp=sharing](https://drive.google.com/drive/folders/1-nhzeWVgpXwMyNEQ5j-MwJxSzwKyT2an?usp=sharing)
|~29B|[https://drive.google.com/drive/folders/10Mtl8_XUJb2mmHLyRC9x1wltdIWy6aaP?usp=sharing](https://drive.google.com/drive/folders/10Mtl8_XUJb2mmHLyRC9x1wltdIWy6aaP?usp=sharing)
|~97B|[https://huggingface.co/JunxiongWang/BiGS_128](https://huggingface.co/JunxiongWang/BiGS_128)

MNLI checkpoint:

|**Training Tokens**|**Model**|
|----------|----------|
|~11B|[https://drive.google.com/drive/folders/1-tn5ar_tRi9DnK_bNMZtPpappUdNnVET?usp=sharing](https://drive.google.com/drive/folders/1-tn5ar_tRi9DnK_bNMZtPpappUdNnVET?usp=sharing)
|~29B|[https://drive.google.com/drive/folders/116JwMbChYp9tBuPTz5jbiaulhXrXt1P2?usp=sharing](https://drive.google.com/drive/folders/116JwMbChYp9tBuPTz5jbiaulhXrXt1P2?usp=sharing)
|~97B|[https://huggingface.co/JunxiongWang/BiGS_128_MNLI](https://huggingface.co/JunxiongWang/BiGS_128_MNLI)

Sentence length: 512

|**Training Tokens**|**Model**|
|----------|----------|
|~108B|[https://huggingface.co/JunxiongWang/BiGS_512](https://huggingface.co/JunxiongWang/BiGS_512)

MNLI checkpoint:

|**Training Tokens**|**Model**|
|----------|----------|
|~108B|[https://huggingface.co/JunxiongWang/BiGS_512_MNLI](https://huggingface.co/JunxiongWang/BiGS_512_MNLI)

Sentence length: 1024

|**Training Tokens**|**Model**|
|----------|----------|
|~110B|[https://huggingface.co/JunxiongWang/BiGS_1024](https://huggingface.co/JunxiongWang/BiGS_1024)

Sentence length: 4096

|**Training Tokens**|**Model**|
|----------|----------|
|~110B|[https://huggingface.co/JunxiongWang/BiGS_4096](https://huggingface.co/JunxiongWang/BiGS_4096)

# Example Usage

## Load Masked Language Model

```python
import jax
from jax import numpy as jnp
from transformers import BertTokenizer
from BiGS.modeling_flax_bigs import FlaxBiGSForMaskedLM
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
model = FlaxBiGSForMaskedLM.from_pretrained('JunxiongWang/BiGS_128')
text = "The goal of life is [MASK]."
encoded_input = tokenizer(text, return_tensors='np', padding='max_length', max_length=128)
output = model(**encoded_input)
tokenizer.convert_ids_to_tokens(jnp.flip(jnp.argsort(jax.nn.softmax(output.logits[encoded_input['input_ids']==103]))[0])[:10])
# output: ['happiness', 'love', 'peace', 'perfection', 'life', 'enlightenment', 'god', 'survival', 'freedom', 'good']
jnp.flip(jnp.sort(jax.nn.softmax(output.logits[encoded_input['input_ids']==103]))[0])[:10]
# probability: [0.16052087, 0.04306792, 0.03651363, 0.03468223, 0.02927081, 0.02549769, 0.02385132, 0.02261189, 0.01672831, 0.01619471]
text = "Paris is the [MASK] of France."
encoded_input = tokenizer(text, return_tensors='np', padding='max_length', max_length=128)
output = model(**encoded_input)
tokenizer.convert_ids_to_tokens(jnp.flip(jnp.argsort(jax.nn.softmax(output.logits[encoded_input['input_ids']==103]))[0])[:8])
# output: ['capital', 'centre', 'center', 'city', 'capitol', 'prefecture', 'headquarters', 'president', 'metropolis', 'heart']
jnp.flip(jnp.sort(jax.nn.softmax(output.logits[encoded_input['input_ids']==103]))[0])[:10]
# probability: [0.9981787 , 0.00034076, 0.00026992, 0.00026926, 0.00017787, 0.00004816, 0.00004256, 0.00003716, 0.00003634, 0.00002893]
``` 

## Load Sequence Classification Model

```python
from BiGS.modeling_flax_bigs import FlaxBiGSForSequenceClassification
model = FlaxBiGSForSequenceClassification.from_pretrained('JunxiongWang/BiGS_512')
```

## Load Question Answering Model

```python
from BiGS.modeling_flax_bigs import FlaxBiGSForQuestionAnswering
model = FlaxBiGSForQuestionAnswering.from_pretrained('JunxiongWang/BiGS_512')
```

## Load Multiple Choice Classification Model

```python
from BiGS.modeling_flax_bigs import FlaxBiGSForMultipleChoice
model = FlaxBiGSForMultipleChoice.from_pretrained('JunxiongWang/BiGS_512')
```

# Pretrain

See [pretrain.md](pretrain.md)

# Finetune

### GLUE

See [GLUE.md](GLUE.md) and [GLUE_freeze.md](GLUE_freeze.md)

# Citation

```
@article{wang2022pretraining,
  title={Pretraining Without Attention},
  author={Wang, Junxiong and Yan, Jing Nathan and Gu, Albert and Rush, Alexander M},
  journal={arXiv preprint arXiv:2212.10544},
  year={2022}
}
```