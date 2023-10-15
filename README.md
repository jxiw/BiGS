## Pretraining Without Attention(BiGS)
## Bidirectional Language Modeling with State Space Model<br>

**Update: Pytorch models port from JAX**

**Torch Masked Language Model**

```python
import torch
from BiGS.modeling_bigs import BiGSForMaskedLM
model = BiGSForMaskedLM.from_pretrained('JunxiongWang/BiGS_128')
```

**Torch Sequence Classification Model**

```python
import torch
from BiGS.modeling_bigs import BiGSForSequenceClassification
model = BiGSForSequenceClassification.from_pretrained('JunxiongWang/BiGS_128')
```

For GLUE task, please see [GLUE_torch.md](GLUE_torch.md) and [GLUE_torch_freeze.md](GLUE_torch_freeze.md). If you don't want to use MNLI checkpoints to finetune MRPC, RTE, STS-B, please run [GLUE_torch_freeze.md](GLUE_torch_freeze.md). Notice that torch version has slight worse results compared with Jax version.

### Official JAX Implementation

### [Paper](https://arxiv.org/abs/2212.10544) | [![Hugging Face Hub](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Hub-blue)](https://huggingface.co/JunxiongWang) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Fz3OSRF3PZEF_dlnyJ3KZ8Bq35DfUrIB?usp=sharing) 
 
<img width="537" alt="BiGS" src="https://user-images.githubusercontent.com/16102460/221464744-06b6538a-7e84-4c95-909f-239eab1dba71.png">

This repository contains BiGS's jax model definitions, pretrained models weights, training and fintuning code for our paper exploring using state space models for pretraining. You can find more details in our paper. 

> [**Pretraining Without Attention**](https://arxiv.org/abs/2212.10544)<br>
> [Junxiong Wang](), [Jing Nathan Yan](), [Albert Gu](), [Alexander M.Rush]()
> <br>Cornell University, Cornell Tech, DeepMind<br>

Transformers have been essential to pretraining success in NLP. While other architectures have been used, downstream accuracy is either significantly worse, or requires attention layers to match standard benchmarks such as GLUE. This work explores pretraining without attention by using recent advances in sequence routing based on state-space models (SSMs). Our proposed model, Bidirectional Gated SSM (BiGS), combines SSM layers with a multiplicative gating architecture that has been effective in simplified sequence modeling architectures. The model learns static layers that do not consider pair-wise interactions. Even so, BiGS is able to match BERT pretraining accuracy on GLUE and can be extended to long-form pretraining of 4096 tokens without approximation. Analysis shows that while the models have similar accuracy, the approach has significantly different inductive biases than BERT in terms of interactions and syntactic representations. 

This repo contains:
* 🪐 JAX implementation of BiGS and its variants,
* 🛸 Pre-trained BiGS Models of various lengths,
* 💥 Training scripts to train BiGS from scratch,
* 💫 Fine-tuning scripts for GLUE tasks

## Setup

You can run our models on both GPUs and TPUs. 

For TPUs,
```
pip install -r requirements-tpu.txt
```

For GPUs,
```
pip install -r requirements-gpu.txt
```

## Download Models

### Pretrained Models
|**Sentence Length**|**Trained Tokens**|**Link**|
|----------|----------|----------|
|128|~11B|[BiGS-11B-128](https://drive.google.com/drive/folders/1-nhzeWVgpXwMyNEQ5j-MwJxSzwKyT2an?usp=sharing)
|128|~29B|[BiGS-29B-128](https://drive.google.com/drive/folders/10Mtl8_XUJb2mmHLyRC9x1wltdIWy6aaP?usp=sharing)
|128|~97B|[BiGS-97B-128](https://huggingface.co/JunxiongWang/BiGS_128)
|512|~108B|[BiGS-108B-512](https://huggingface.co/JunxiongWang/BiGS_512)
|1024|~110B|[BiGS-110B-1024](https://huggingface.co/JunxiongWang/BiGS_1024)
|4096|~110B|[BiGS-110B-4096](https://huggingface.co/JunxiongWang/BiGS_4096)

### MNLI Checkpoints

|**Sentence Length**|**Trained Tokens**|**Model**|
|----------|----------|----------|
|128|~11B|[BiGS-11B-128MNLI](https://drive.google.com/drive/folders/1-tn5ar_tRi9DnK_bNMZtPpappUdNnVET?usp=sharing)
|128|~29B|[BiGS-29B-128MNLI](https://drive.google.com/drive/folders/116JwMbChYp9tBuPTz5jbiaulhXrXt1P2?usp=sharing)
|128|~97B|[BiGS-97B-128MNLI](https://huggingface.co/JunxiongWang/BiGS_128_MNLI)
|512|~108B|[BiGS-108B-512MNLI](https://huggingface.co/JunxiongWang/BiGS_512_MNLI)

<!-- Sentence length: 128

|**Training Tokens**|**Model**|
|----------|----------|
|~11B|[https://drive.google.com/drive/folders/1-nhzeWVgpXwMyNEQ5j-MwJxSzwKyT2an?usp=sharing](https://drive.google.com/drive/folders/1-nhzeWVgpXwMyNEQ5j-MwJxSzwKyT2an?usp=sharing)
|~29B|[https://drive.google.com/drive/folders/10Mtl8_XUJb2mmHLyRC9x1wltdIWy6aaP?usp=sharing](https://drive.google.com/drive/folders/10Mtl8_XUJb2mmHLyRC9x1wltdIWy6aaP?usp=sharing)
|~97B|[https://huggingface.co/JunxiongWang/BiGS_128](https://huggingface.co/JunxiongWang/BiGS_128)
 -->

<!-- Sentence length: 512

|**Training Tokens**|**Model**|
|----------|----------|
|~108B|[https://huggingface.co/JunxiongWang/BiGS_512](https://huggingface.co/JunxiongWang/BiGS_512) -->

<!-- MNLI checkpoint:

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
 -->
## Example Usage


### Load Masked Language Model

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

### Load Sequence Classification Model

```python
from BiGS.modeling_flax_bigs import FlaxBiGSForSequenceClassification
model = FlaxBiGSForSequenceClassification.from_pretrained('JunxiongWang/BiGS_512')
```

### Load Question Answering Model

```python
from BiGS.modeling_flax_bigs import FlaxBiGSForQuestionAnswering
model = FlaxBiGSForQuestionAnswering.from_pretrained('JunxiongWang/BiGS_512')
```

### Load Multiple Choice Classification Model

```python
from BiGS.modeling_flax_bigs import FlaxBiGSForMultipleChoice
model = FlaxBiGSForMultipleChoice.from_pretrained('JunxiongWang/BiGS_512')
```

## Pretraining

See [pretrain.md](pretrain.md)

## Finetuning 

### GLUE

See [GLUE.md](GLUE.md) and [GLUE_freeze.md](GLUE_freeze.md). If you don't want to use MNLI checkpoints to finetune MRPC, RTE, STS-B, please run [GLUE_freeze.md](GLUE_freeze.md).

## Citation

```
@article{wang2022pretraining,
  title={Pretraining Without Attention},
  author={Wang, Junxiong and Yan, Jing Nathan and Gu, Albert and Rush, Alexander M},
  journal={arXiv preprint arXiv:2212.10544},
  year={2022}
}
```
