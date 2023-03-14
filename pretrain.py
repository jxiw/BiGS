#!/usr/bin/env python
# coding=utf-8
# Copyright 2021. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for masked language modeling (BERT, ALBERT, RoBERTa...) with whole word masking on a
text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=fill-mask
"""
import json
import logging
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
# You can also adapt this script on your own masked language modeling task. Pointers for this are left as comments.
from pathlib import Path
from typing import Optional, Callable

import jax
import jax.numpy as jnp
import optax
import wandb
from flax import jax_utils, traverse_util
from flax.training import train_state
from flax.training.common_utils import get_metrics, onehot, shard
from huggingface_hub import Repository
from tqdm import tqdm

from BiGS.configuration_bigs import BiGSConfig
from BiGS.modeling_flax_bigs import FlaxBiGSForMaskedLM

from bigs_dataset.pretraining_dataset import PreTrainingDataset
# if you want to train it on TPU, change it to `from bigs_dataset.pretraining_dataset_gcb import PreTrainingDataset`

from transformers import (
    FLAX_MODEL_FOR_MASKED_LM_MAPPING,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)
from transformers.utils import get_full_repo_name

MODEL_CONFIG_CLASSES = list(FLAX_MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class TrainingArguments:
    output_dir: str = field(
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help": (
                "Overwrite the content of the output directory. "
                "Use this to continue training if output_dir points to a checkpoint directory."
            )
        },
    )
    do_train: bool = field(default=True, metadata={"help": "Whether to run training."})
    do_eval: bool = field(default=True, metadata={"help": "Whether to run eval on the dev set."})
    per_device_train_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )
    learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate for AdamW."})
    weight_decay: float = field(default=0.01, metadata={"help": "Weight decay for AdamW if we apply some."})
    adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for AdamW optimizer"})
    adam_beta2: float = field(default=0.98, metadata={"help": "Beta2 for AdamW optimizer"})
    adam_epsilon: float = field(default=1e-6, metadata={"help": "Epsilon for AdamW optimizer."})
    adafactor: bool = field(default=False, metadata={"help": "Whether or not to replace AdamW by Adafactor."})
    max_train_shards: int = field(default=1000, metadata={"help": "Total number of training epochs to perform."})
    warmup_steps: int = field(default=0, metadata={"help": "Linear warmup over warmup_steps."})
    logging_steps: int = field(default=500, metadata={"help": "Log every X updates steps."})
    save_steps: int = field(default=500, metadata={"help": "Save checkpoint every X updates steps."})
    eval_steps: int = field(default=None, metadata={"help": "Run an evaluation every X steps."})
    seed: int = field(default=42, metadata={"help": "Random seed that will be set at the beginning of training."})
    push_to_hub: bool = field(
        default=False, metadata={"help": "Whether or not to upload the trained model to the model hub after training."}
    )
    hub_model_id: str = field(
        default=None, metadata={"help": "The name of the repository to keep in sync with the local `output_dir`."}
    )
    hub_token: str = field(default=None, metadata={"help": "The token to use to push to the Model Hub."})

    num_total_train_steps: Optional[int] = field(
        default=2000,
        metadata={"help": "Maximum number of training steps of effective batch size to complete."},
    )

    checkpoint_epoch: int = field(default=50000, metadata={"help": "checkpoint epoch"})

    schedule_type: str = field(
        default='linear',
        metadata={"help": "Type of scheduler"}
    )

    cosine_schedule_alpha: float = field(
        default=0.1,
        metadata={"help": "Cosine schedule alpha"}
    )

    def __post_init__(self):
        if self.output_dir is not None:
            self.output_dir = os.path.expanduser(self.output_dir)

    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values (for JSON serialization support). It obfuscates
        the token values by removing their value.
        """
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
                d[k] = [x.value for x in v]
            if k.endswith("_token"):
                d[k] = f"<{k.upper()}>"
        return d


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
                    "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    dtype: Optional[str] = field(
        default="float32",
        metadata={
            "help": "Floating-point format in which the model weights should be initialized and trained. Choose one of `[float32, float16, bfloat16]`."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )
    n_layers: int = field(
        default=12,
        metadata={"help": "Layer of S4 Model."}
    )
    num_ssm: int = field(
        default=64,
        metadata={"help": "Number of SSM."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={"help": "Max length of S4 Model."}
    )
    hidden_dropout_prob: float = field(
        default=0.1,
        metadata={"help": "Dropout prob."}
    )
    hidden_size: int = field(
        default=1024,
        metadata={"help": "Hidden size."}
    )
    intermediate_size: int = field(
        default=4096,
        metadata={"help": "Hidden size."}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_path: Optional[str] = field(
        default="books_wiki_en_corpus", metadata={"help": "pretrain_dataset"}
    )
    num_workers: Optional[int] = field(default=4, metadata={"help": "num of dataloader workers"})

    async_worker: Optional[bool] = field(default=True, metadata={"help": "async_worker"})

    data_loader_type: Optional[str] = field(
        default="per_device",
        metadata={
            "help": "Dataloader to use: dist=distributed, per_device=local per device",
            "choices": ["dist", "per_device"],
        },
    )

    max_predictions_per_seq: Optional[int] = field(
        default=20,
        metadata={"help": "The maximum number of masked tokens in a sequence to be predicted."},
    )

    train_batch_size: Optional[int] = field(
        default=None,
        metadata={"help": "The total number of train batch size."},
    )


# As we're using Flax, we also write a utility function to return a default TrainState object.
# This function initializes model parameters, as well as our optimizer. Note that for S4 models,
# we use a custom learning rate for parameters of the S4 kernel (lr = 0.001, no weight decay).
def map_nested_fn(fn):
    """Recursively apply `fn to the key-value pairs of a nested dict / pytree."""

    def map_fn(nested_dict):
        return {
            k: (map_fn(v) if hasattr(v, "keys") else fn(k, v))
            for k, v in nested_dict.items()
        }

    return map_fn


def create_learning_rate_fn(
        num_total_train_steps: int, num_warmup_steps: int, learning_rate: float, schedule_type: str = 'linear',
        cosine_schedule_alpha: float = 0.1,
) -> Callable[[int], jnp.array]:
    """Returns a linear warmup, linear_decay learning rate function."""
    warmup_fn = optax.linear_schedule(init_value=0.0, end_value=learning_rate, transition_steps=num_warmup_steps)
    if schedule_type == 'constant':
        decay_fn = optax.constant_schedule(value=learning_rate)
    elif schedule_type == 'linear':
        decay_fn = optax.linear_schedule(
            init_value=learning_rate, end_value=0, transition_steps=num_total_train_steps - num_warmup_steps
        )
    elif schedule_type == 'cosine':
        decay_fn = optax.cosine_decay_schedule(
            init_value=learning_rate, decay_steps=num_total_train_steps - num_warmup_steps, alpha=cosine_schedule_alpha,
        )
    schedule_fn = optax.join_schedules(schedules=[warmup_fn, decay_fn], boundaries=[num_warmup_steps])
    return schedule_fn


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
        datefmt="[%X]",
    )

    # Log on each process the small summary:
    logger = logging.getLogger(__name__)

    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Handle the repository creation
    if training_args.push_to_hub:
        if training_args.hub_model_id is None:
            repo_name = get_full_repo_name(
                Path(training_args.output_dir).absolute().name, token=training_args.hub_token
            )
        else:
            repo_name = training_args.hub_model_id
        repo = Repository(training_args.output_dir, clone_from=repo_name)

    train_batch_size = int(training_args.per_device_train_batch_size) * jax.device_count()

    # we load the dataset from 24h bert
    data_args.train_batch_size = train_batch_size

    pretrain_dataset_provider = PreTrainingDataset(data_args)
    validation_dataset_provider = PreTrainingDataset(data_args, data_prefix="test") if training_args.do_eval else None

    # Load pretrained model and tokenizer
    config = BiGSConfig(
        cache_dir=model_args.cache_dir,
        num_hidden_layers=model_args.n_layers,
        num_ssm=model_args.num_ssm,
        hidden_size=model_args.hidden_size,
        intermediate_size=model_args.intermediate_size,
        hidden_dropout_prob=model_args.hidden_dropout_prob,
        # for 24hbert, they fix the sequence length to 128
        max_position_embeddings=model_args.max_seq_length,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        # follow the 24hbert, we fix the tokenizer to bert-large-uncased
        "bert-large-uncased",
        do_lower_case=True,
        max_len=model_args.max_seq_length,
    )

    # Initialize our training
    rng = jax.random.PRNGKey(training_args.seed)
    dropout_rngs = jax.random.split(rng, jax.local_device_count())

    model = FlaxBiGSForMaskedLM(
        config=config,
        seed=training_args.seed,
        dtype=getattr(jnp, model_args.dtype),
    )

    # Create learning rate schedule
    linear_decay_lr_schedule_fn = create_learning_rate_fn(
        training_args.num_total_train_steps,
        training_args.warmup_steps,
        training_args.learning_rate,
        training_args.schedule_type,
        training_args.cosine_schedule_alpha,
    )

    # We use Optax's "masking" functionality to not apply weight decay
    # to bias and LayerNorm scale parameters. decay_mask_fn returns a
    # mask boolean with the same structure as the parameters.
    # The mask is True for parameters that should be decayed.
    # Note that this mask is specifically adapted for FlaxBERT-like models.
    # For other models, one should correct the layer norm parameter naming
    # accordingly.
    def decay_mask_fn(params):
        flat_params = traverse_util.flatten_dict(params)
        flat_mask = {path: (path[-1] != "bias" and path[-2:] != ("LayerNorm", "scale")) for path in flat_params}
        return traverse_util.unflatten_dict(flat_mask)

    # create adam optimizer
    if training_args.adafactor:
        # We use the default parameters here to initialize adafactor,
        # For more details about the parameters please check https://github.com/deepmind/optax/blob/ed02befef9bf81cbbf236be3d2b0e032e9ed4a40/optax/_src/alias.py#L74
        optimizer = optax.adafactor(
            learning_rate=linear_decay_lr_schedule_fn,
        )
    else:
        optimizer = optax.adamw(
            learning_rate=linear_decay_lr_schedule_fn,
            b1=training_args.adam_beta1,
            b2=training_args.adam_beta2,
            eps=training_args.adam_epsilon,
            weight_decay=training_args.weight_decay,
            mask=decay_mask_fn,
        )

    # Setup train state
    state = train_state.TrainState.create(apply_fn=model.__call__, params=model.params, tx=optimizer)
    print("model.params", jax.tree_map(jax.numpy.shape, model.params))

    # Print parameter count
    _is_complex = lambda x: x.dtype in [jnp.complex64, jnp.complex128]
    param_sizes = map_nested_fn(
        lambda k, param: param.size * (2 if _is_complex(param) else 1)
    )(model.params)
    print(f"[*] Trainable Parameters: {sum(jax.tree_leaves(param_sizes))}")

    # Define gradient update step fn
    def train_step(state, batch, dropout_rng):
        dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)

        def loss_fn(params):
            labels = batch.pop("labels")
            logits = state.apply_fn(**batch, params=params, dropout_rng=dropout_rng, train=True)[0]

            # compute loss, ignore padded input tokens
            label_mask = (labels > 0)
            loss = optax.softmax_cross_entropy(logits, onehot(labels, logits.shape[-1])) * label_mask

            # take average
            loss = loss.sum() / label_mask.sum()

            return loss

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grad = grad_fn(state.params)
        grad = jax.lax.pmean(grad, "batch")
        new_state = state.apply_gradients(grads=grad)

        metrics = jax.lax.pmean(
            {"loss": loss, "learning_rate": linear_decay_lr_schedule_fn(state.step)}, axis_name="batch"
        )

        return new_state, metrics, new_dropout_rng

    # Create parallel version of the train step
    p_train_step = jax.pmap(train_step, "batch", donate_argnums=(0,))

    # Define eval fn
    def eval_step(params, batch):
        labels = batch.pop("labels")

        logits = model(**batch, params=params, train=False)[0]

        # compute loss, ignore padded input tokens
        label_mask = (labels > 0)
        loss = optax.softmax_cross_entropy(logits, onehot(labels, logits.shape[-1])) * label_mask

        # compute accuracy
        accuracy = jnp.equal(jnp.argmax(logits, axis=-1), labels) * label_mask

        # summarize metrics
        metrics = {"loss": loss.sum(), "accuracy": accuracy.sum(), "normalizer": label_mask.sum()}
        metrics = jax.lax.psum(metrics, axis_name="batch")

        return metrics

    p_eval_step = jax.pmap(eval_step, "batch", donate_argnums=(0,))

    # Replicate the train state on each device
    state = jax_utils.replicate(state)

    wandb.init(project="BiGS_pretrain")
    wandb.config.update(training_args)

    epochs = tqdm(range(training_args.max_train_shards), desc=f"Epoch ... (1/{training_args.max_train_shards})",
                  position=0, smoothing=1)

    cur_step = 0
    total_data_sample = 0
    before_train_time = time.time()

    validation_shard_index = 0
    for epoch in epochs:
        if cur_step > training_args.num_total_train_steps:
            break
        # ======================== Training ================================
        train_metrics = []

        train_shard_index = epoch
        train_dataset_iterator, total_length = pretrain_dataset_provider.get_shard(train_shard_index)
        pretrain_dataset_provider.prefetch_shard(train_shard_index + 1)

        # Gather the indexes for creating the batch and do a training step
        for batch_index_number, batch_index in enumerate(tqdm(train_dataset_iterator, desc="Training...", position=1)):

            train_input_ids, train_input_mask, train_token_type_ids, labels = pretrain_dataset_provider.get_batch(
                batch_index)

            # change batch_data to jax numpy

            model_inputs = {
                "input_ids": jnp.array(train_input_ids.numpy()),
                "attention_mask": jnp.array(train_input_mask.numpy()),
                "token_type_ids": jnp.array(train_token_type_ids.numpy()),
                "labels": jnp.array(labels.numpy())
            }

            total_data_sample += train_batch_size

            # shard the input to different device
            model_inputs = shard(model_inputs)

            # Model forward
            state, train_metric, dropout_rngs = p_train_step(state, model_inputs, dropout_rngs)
            train_metrics.append(train_metric)

            cur_step += 1

            if cur_step % training_args.logging_steps == 0:
                # Save metrics
                train_metric = jax_utils.unreplicate(train_metric)

                epochs.write(
                    f"Step... ({cur_step} | Train Shard Index: {train_shard_index} | Process Data Samples: {total_data_sample} | Train Loss: {train_metric['loss']}, Learning Rate: {train_metric['learning_rate']}, SamplesPerSec: {total_data_sample / (time.time() - before_train_time)})"
                )

                wandb.log(
                    {
                        "Train Loss": train_metric['loss'],
                        'Learning Rate': train_metric['learning_rate'],
                        'Time': time.time() - before_train_time,
                        'Epoch': epoch,
                        'Train Shard Index': train_shard_index,
                    }, step=cur_step
                )

                train_metrics = []

            if cur_step % training_args.save_steps == 0 and cur_step > 0:
                # save checkpoint after each epoch and push checkpoint to the hub
                if jax.process_index() == 0:
                    params = jax.device_get(jax.tree_map(lambda x: x[0], state.params))
                    model.save_pretrained(training_args.output_dir, params=params)
                    tokenizer.save_pretrained(training_args.output_dir)
                    if training_args.push_to_hub:
                        repo.push_to_hub(commit_message=f"Saving weights and logs of step {cur_step}", blocking=False)

        pretrain_dataset_provider.release_shard(train_shard_index)

        if (epoch + 1) % training_args.eval_steps == 0:
            # ======================== Evaluating ==============================
            validation_dataset_iterator, total_length = validation_dataset_provider.get_shard(
                validation_shard_index)
            validation_dataset_provider.prefetch_shard(validation_shard_index + 1)

            eval_metrics = []
            for val_batch_index_number, val_batch_index in enumerate(
                    tqdm(validation_dataset_iterator, desc="Evaluating ...", position=2, smoothing=1)):
                test_input_ids, test_input_mask, test_token_type_ids, labels = validation_dataset_provider.get_batch(
                    val_batch_index)

                data_to_remove = test_input_ids.shape[0] % jax.device_count()
                if data_to_remove != 0:
                    test_input_ids = test_input_ids[:-data_to_remove]
                    test_input_mask = test_input_mask[:-data_to_remove]
                    test_token_type_ids = test_token_type_ids[:-data_to_remove]
                    labels = labels[:-data_to_remove]

                # change batch to jax
                model_inputs = {
                    "input_ids": jnp.array(test_input_ids.numpy()),
                    "attention_mask": jnp.array(test_input_mask.numpy()),
                    "token_type_ids": jnp.array(test_token_type_ids.numpy()),
                    "labels": jnp.array(labels.numpy())
                }

                # Model forward
                model_inputs = shard(model_inputs)
                metrics = p_eval_step(state.params, model_inputs)
                eval_metrics.append(metrics)

            # normalize eval metrics
            eval_metrics = get_metrics(eval_metrics)
            eval_metrics = jax.tree_map(jnp.sum, eval_metrics)
            eval_normalizer = eval_metrics.pop("normalizer")
            eval_metrics = jax.tree_map(lambda x: x / eval_normalizer, eval_metrics)

            # Update progress bar
            epochs.write(
                f"Step... ({cur_step} | Validation Loss: {eval_metrics['loss']}, Validation Acc: {eval_metrics['accuracy']})")

            wandb.log(
                {
                    "Step": cur_step,
                    "Validation Loss": eval_metrics['loss'],
                    "Validation Accuracy": eval_metrics['accuracy'],
                    "Validation Perplexity": math.exp(eval_metrics["loss"])
                }, step=cur_step
            )

            validation_dataset_provider.release_shard(validation_shard_index)

            validation_shard_index += 1

    print(f"Training Finish, Total Train Time SamplesPerSec: {total_data_sample / (time.time() - before_train_time)}")

    # save final model
    if jax.process_index() == 0:
        params = jax.device_get(jax.tree_map(lambda x: x[0], state.params))
        model.save_pretrained(training_args.output_dir, params=params)
        tokenizer.save_pretrained(training_args.output_dir)
        if training_args.push_to_hub:
            repo.push_to_hub(commit_message=f"Saving weights and logs of step {cur_step}", blocking=False)

    # Eval after training
    if training_args.do_eval:
        # run another validation
        validation_dataset_iterator, total_length = validation_dataset_provider.get_shard(
            validation_shard_index)

        eval_metrics = []
        for batch_index_number, batch_index in enumerate(
                tqdm(validation_dataset_iterator, desc="Evaluating ...", position=2, smoothing=1)):
            test_input_ids, test_input_mask, test_token_type_ids, labels = validation_dataset_provider.get_batch(
                batch_index)

            data_to_remove = test_input_ids.shape[0] % jax.device_count()
            if data_to_remove != 0:
                test_input_ids = test_input_ids[:-data_to_remove]
                test_input_mask = test_input_mask[:-data_to_remove]
                test_token_type_ids = test_token_type_ids[:-data_to_remove]
                labels = labels[:-data_to_remove]

            # change batch to jax
            model_inputs = {
                "input_ids": jnp.array(test_input_ids.numpy()),
                "attention_mask": jnp.array(test_input_mask.numpy()),
                "token_type_ids": jnp.array(test_token_type_ids.numpy()),
                "labels": jnp.array(labels.numpy())
            }

            # Model forward
            model_inputs = shard(model_inputs)
            metrics = p_eval_step(state.params, model_inputs)
            eval_metrics.append(metrics)

        # normalize eval metrics
        eval_metrics = get_metrics(eval_metrics)
        eval_metrics = jax.tree_map(lambda metric: jnp.sum(metric).item(), eval_metrics)
        eval_normalizer = eval_metrics.pop("normalizer")
        eval_metrics = jax.tree_map(lambda x: x / eval_normalizer, eval_metrics)

        try:
            perplexity = math.exp(eval_metrics["loss"])
        except OverflowError:
            perplexity = float("inf")
        eval_metrics["perplexity"] = perplexity

        if jax.process_index() == 0:
            eval_metrics = {f"eval_{metric_name}": value for metric_name, value in eval_metrics.items()}
            path = os.path.join(training_args.output_dir, "eval_results.json")
            with open(path, "w") as f:
                json.dump(eval_metrics, f, indent=4, sort_keys=True)


if __name__ == "__main__":
    main()
