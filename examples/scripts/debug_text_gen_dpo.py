"""
1. First checkout the trl branch:

git clone https://github.com/huggingface/trl.git
git checkout debug-dpo

2. Install deps with:

make dev

Then install latest versions of transformers / accelerate / deepspeed

pip install transformers==4.39.1 accelerate==0.28.0 deepspeed==0.14.0

See examples/scripts/requirements.txt for exact versions.

3. Run with:

TRANSFORMERS_VERBOSITY=info ACCELERATE_LOG_LEVEL=info accelerate launch --config_file=examples/accelerate_configs/deepspeed_zero3.yaml examples/scripts/debug_text_gen_dpo.py

If you change `gradient_accumulation_steps=1` in the `TrainingArguments` and `examples/accelerate_configs/deepspeed_zero3.yaml` config it runs fine. But with `gradient_accumulation_steps=2` it fails with the following error:

Traceback (most recent call last):
  File "/fsx/lewis/git/hf/trl/examples/scripts/debug_text_gen_dpo.py", line 141, in <module>
    main()
  File "/fsx/lewis/git/hf/trl/examples/scripts/debug_text_gen_dpo.py", line 137, in main
    trainer.train()
  File "/fsx/lewis/miniconda3/envs/trl/lib/python3.10/site-packages/transformers/trainer.py", line 1624, in train
    return inner_training_loop(
  File "/fsx/lewis/miniconda3/envs/trl/lib/python3.10/site-packages/transformers/trainer.py", line 1961, in _inner_training_loop
    tr_loss_step = self.training_step(model, inputs)
  File "/fsx/lewis/miniconda3/envs/trl/lib/python3.10/site-packages/transformers/trainer.py", line 2902, in training_step
    loss = self.compute_loss(model, inputs)
  File "/fsx/lewis/git/hf/trl/examples/scripts/debug_text_gen_dpo.py", line 43, in compute_loss
    with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
  File "/fsx/lewis/miniconda3/envs/trl/lib/python3.10/contextlib.py", line 142, in __exit__
    next(self.gen)
  File "/fsx/lewis/git/hf/trl/trl/models/utils.py", line 146, in unwrap_model_for_generation
    with deepspeed.zero.GatheredParameters(model.parameters()):
  File "/fsx/lewis/miniconda3/envs/trl/lib/python3.10/site-packages/deepspeed/runtime/zero/partition_parameters.py", line 2177, in __exit__
    self.params[0].partition(param_list=self.params, has_been_updated=False)
  File "/fsx/lewis/miniconda3/envs/trl/lib/python3.10/site-packages/deepspeed/runtime/zero/partition_parameters.py", line 1325, in partition
    self._partition(param_list, has_been_updated=has_been_updated)
  File "/fsx/lewis/miniconda3/envs/trl/lib/python3.10/site-packages/deepspeed/runtime/zero/partition_parameters.py", line 1474, in _partition
    self._partition_param(param, has_been_updated=has_been_updated)
  File "/fsx/lewis/miniconda3/envs/trl/lib/python3.10/site-packages/deepspeed/utils/nvtx.py", line 15, in wrapped_fn
    ret_val = func(*args, **kwargs)
  File "/fsx/lewis/miniconda3/envs/trl/lib/python3.10/site-packages/deepspeed/runtime/zero/partition_parameters.py", line 1507, in _partition_param
    free_param(param)
  File "/fsx/lewis/miniconda3/envs/trl/lib/python3.10/site-packages/deepspeed/utils/nvtx.py", line 15, in wrapped_fn
    ret_val = func(*args, **kwargs)
  File "/fsx/lewis/miniconda3/envs/trl/lib/python3.10/site-packages/deepspeed/runtime/zero/partition_parameters.py", line 279, in free_param
    assert not param.ds_active_sub_modules, param.ds_summary()
AssertionError: {'id': 0, 'status': 'AVAILABLE', 'numel': 25755648, 'ds_numel': 25755648, 'shape': (50304, 512), 'ds_shape': (50304, 512), 'requires_grad': True, 'grad_shape': None, 'persist': False, 'active_sub_modules': {182}, 'ds_tensor.shape': torch.Size([3219456])}
"""
import warnings
from contextlib import nullcontext
from typing import Any, Dict, Tuple, Union

import torch
import torch.nn as nn
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    TrainingArguments,
)

from trl import DPOTrainer
from trl.models.utils import unwrap_model_for_generation


class MyDPOTrainer(DPOTrainer):
    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if not self.use_dpo_data_collator:
            warnings.warn(
                "compute_loss is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )

        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            generations = unwrapped_model.generate(inputs["prompt_input_ids"])
            print(f"Generations: {self.tokenizer.batch_decode(generations)}")

        compute_loss_context_manager = torch.cuda.amp.autocast if self._peft_has_been_casted_to_bf16 else nullcontext

        with compute_loss_context_manager():
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="train")

        # Make sure to move the loss to the device the original accumulating loss is at back in the `Trainer` class:
        loss = loss.to(self.args.device)
        # force log the metrics
        self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return (loss, metrics)
        return loss


def main():
    training_args = TrainingArguments(
        output_dir="scratch/dummy-model",
        per_device_train_batch_size=2,
        max_steps=3,
        remove_unused_columns=False,
        gradient_accumulation_steps=2,  # Runs fine with gradient_accumulation_steps=1
        learning_rate=9e-1,
        evaluation_strategy="steps",
        bf16=True,
    )

    # fmt: off
    dummy_dataset_dict = {
            "prompt": [
                "hello",
                "how are you",
                "What is your name?",
                "What is your name?",
                "Which is the best programming language?",
                "Which is the best programming language?",
                "Which is the best programming language?",
                "[INST] How is the stock price? [/INST]",
                "[INST] How is the stock price? [/INST] ",
            ],
            "chosen": [
                "hi nice to meet you",
                "I am fine",
                "My name is Mary",
                "My name is Mary",
                "Python",
                "Python",
                "Python",
                "$46 as of 10am EST",
                "46 as of 10am EST",
            ],
            "rejected": [
                "leave me alone",
                "I am not fine",
                "Whats it to you?",
                "I dont have a name",
                "Javascript",
                "C++",
                "Java",
                " $46 as of 10am EST",
                " 46 as of 10am EST",
            ],
        }
    dummy_dataset = Dataset.from_dict(dummy_dataset_dict)

    model_id = "HuggingFaceH4/pythia-70m-sft"
    model_revision = "v0.0"
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=model_revision)
    ref_model = AutoModelForCausalLM.from_pretrained(model_id, revision=model_revision)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=model_revision)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    trainer = MyDPOTrainer(
                model=model,
                ref_model=ref_model,
                beta=0.1,
                loss_type="sigmoid",
                args=training_args,
                tokenizer=tokenizer,
                train_dataset=dummy_dataset,
                eval_dataset=dummy_dataset,
                precompute_ref_log_probs=False,
            )

    trainer.train()


if __name__ == "__main__":
    main()
