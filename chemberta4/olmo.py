from typing import Dict, Any, Tuple, Optional
from deepchem.models.torch_models import HuggingFaceModel
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, OlmoConfig, OlmoForCausalLM, BitsAndBytesConfig
from deepchem.models.torch_models.olmo_layers import OlmoForSequenceClassification
from transformers.modeling_utils import PreTrainedModel
try:
    import torch
    has_torch = True
except:
    has_torch = False
import gc, torch
import torch.nn as nn
from transformers.modeling_layers import GenericForSequenceClassification
from transformers import OlmoPreTrainedModel, AutoModel
import torch.nn as nn


class OlmoForSequenceClassification(GenericForSequenceClassification, OlmoPreTrainedModel):
    """
    OLMo model adapted for sequence classification tasks.

    This class extends the base OLMo model by adding a lightweight classification
    head on top of the pooled sequence representation. The head consists of a single
    linear layer that maps the hidden representation to the desired number of labels.

    The model is suitable for tasks such as regression (num_labels=1) or
    classification (num_labels > 1).

    Example:
    -------
    >>> model = OlmoForSequenceClassification.from_pretrained(  "allenai/OLMo-7b-hf",
    ...                                                        num_labels=1,
    ...                                                        torch_dtype=torch.float16,)   
    >>> tokenizer = AutoTokenizer.from_pretrained('allenai/olmo-7b-hf', trust_remote_code=True) 
    >>> input = tokenizer(["CCCl"],
    ...                    return_tensors="pt")
    >>> output = model(**input)
    >>> output.logits
    >>> output.loss
    """
    base_model_prefix = "model"

    def __init__(self, config):
        super(GenericForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        # Similar to `self.model = AutoModel.from_config(config)` but allows to change the base 
        # model name if needed in the child class
        setattr(self, self.base_model_prefix, AutoModel.from_config(config))
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Linear layer gets initialised in full precision even when the model's parameters are
        # half precision. This leads to an error, so the linear layer's weights' dtype is coverted to the 
        # model's parameters' dtype
        self.score = self.score.to(next(self.parameters()).dtype)

        # Initialize weights and apply final processing
        self.post_init()
    


class Olmo(HuggingFaceModel):
 
    """This class enables training and prediction using Olmo, a decoder-only transformer through the DeepChem API.

    It supports pretraining via causal language modeling, finetuning via regression, 
    classification and multitask regression and they can be specified using 'clm', `regression`, 
    `classification` and `mtr` as arguments to the `task` keyword during model initialisation.

    It uses a tokenizer to create input tokens for the models.
    The default tokenizer model is GPTNeoXTokenizerFast.

    Example
    -------
    >>> import os
    >>> import tempfile
    >>> import shutil
    >>> tempdir = tempfile.mkdtemp()
    >>> import pandas as pd
    >>> import deepchem as dc
    >>>
    >>> # preparing dataset
    >>> smiles = ["CCN(CCSC)C(=O)N[C@@](C)(CC)C(F)(F)F","CC1(C)CN(C(=O)Nc2cc3ccccc3nn2)C[C@@]2(CCOC2)O1"]
    >>> labels = [3.112,2.432]
    >>> df = pd.DataFrame(list(zip(smiles, labels)), columns=["smiles", "task1"])
    >>> with dc.utils.UniversalNamedTemporaryFile(mode='w') as tmpfile:
    ...     df.to_csv(tmpfile.name)
    ...     loader = dc.data.CSVLoader(["task1"], feature_field="smiles", featurizer=dc.feat.DummyFeaturizer())
    ...     dataset = loader.create_dataset(tmpfile.name)
    >>>
    >>> model = Olmo(task="regression",
                tokenizer_path=t"allenai/olmo-7b-hf",
                finetune_strategy = 'qlora',
                config = {'torch_dtype': torch.float16},
                batch_size = 2)
    >>>
    >>> from deepchem.models.lightning import LightningTorchModel
    >>> trainer = LightningTorchModel(model=model,
    ...                             batch_size=2,
    ...                             max_epochs=1,
    ...                             enable_progress_bar=True,
    ...                             accelerator="gpu",
    ...                             strategy = "ddp",
    ...                             devices = -1,
    ...                             log_every_n_steps=1
    ...                             )
    >>>
    >>> trainer.fit(dataset, num_workers=0)
    >>> predictions = trainer.predict(dataset)
    """

    def __init__(self,
                 task: str,
                 tokenizer_path: str = 'allenai/olmo-7b-hf',
                 n_tasks: int = 1,
                 config: Dict[Any, Any] = {},
                 finetune_strategy: str = 'full_finetune',
                 **kwargs):
        if finetune_strategy not in ('lora', 'qlora', 'full_finetune'):
            raise ValueError(f"finetune_strategy must be 'lora', 'qlora', or 'full_finetune', got '{finetune_strategy}'")
        self.n_tasks = n_tasks
        self.finetune_strategy = finetune_strategy
        """
        Parameters
        ----------
        task: str
            The task defines the type of learning task in the model. The supported tasks are
            - `clm` - causal language modeling commonly used in pretraining
            - `mtr` - multitask regression - a task used for both pretraining base models and finetuning
            - `regression` - use it for regression tasks, like property prediction
            - `classification` - use it for classification tasks
        tokenizer_path: str
            Path containing pretrained tokenizer used to tokenize SMILES string for model inputs. The tokenizer path can either be a huggingFace tokenizer model or a path in the local machine containing the tokenizer.
        n_tasks: int, default 1
            Number of prediction targets for a multitask learning model
        config: Dict[Any, Any], default {}
            Additional keyword arguments forwarded to OlmoConfig. Use this to override
            architecture defaults such as `hidden_size`, `num_hidden_layers`, or
            `torch_dtype`. When loading from a HuggingFace checkpoint these same keys
            are also forwarded to `from_pretrained`.
        """

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,
                                                  trust_remote_code=True)
        self.model: PreTrainedModel
        olmo_config = OlmoConfig(vocab_size=tokenizer.vocab_size,
                                         **config)

        if task == 'clm':
           self.model = OlmoForCausalLM(olmo_config)
        elif task == 'mtr' or task == 'regression':
            olmo_config.problem_type = 'regression'
            olmo_config.num_labels = n_tasks
            self.model = OlmoForSequenceClassification(olmo_config)
        elif task == 'classification':
            if n_tasks == 1:
                olmo_config.problem_type = 'single_label_classification'
            else:
                olmo_config.problem_type = 'multi_label_classification'
                olmo_config.num_labels = n_tasks
            self.model = OlmoForSequenceClassification(olmo_config)
        else:
            raise ValueError('invalid task specification')
        self.config = olmo_config

        super(Olmo, self).__init__(model=self.model,
                                        task=task,
                                        tokenizer=tokenizer,
                                        **kwargs)

    def _prepare_batch(self, batch: Tuple[Any, Any, Any]):
        """
        Prepares a batch of data for the model based on the specified task. It overrides the _prepare_batch
        of parent class for the following condition:-

        - When n_task == 1 and task == 'classification', CrossEntropyLoss is used which takes input in
        long int format.
        - When n_task > 1 and task == 'classification', BCEWithLogitsLoss is used which takes input in
        float format.
        """

        smiles_batch, y, w = batch

        if w is not None:
            w = torch.tensor(w, dtype=torch.float).to(self.device)

        tokens = self.tokenizer(smiles_batch[0].tolist(),
                                padding=True,
                                return_tensors="pt")

        if self.task == 'clm':
            inputs, labels = self.data_collator.torch_mask_tokens(
                tokens['input_ids'])
            inputs = {
                'input_ids': inputs.to(self.device),
                'labels': labels.to(self.device),
                'attention_mask': tokens['attention_mask'].to(self.device),
            }
            return inputs, None, w
        elif self.task in ['regression', 'classification', 'mtr']:
            if y is not None:
                # y is None during predict
                y = torch.from_numpy(y[0])
                if self.task == 'regression' or self.task == 'mtr':
                    y = y.float().to(self.device)
                elif self.task == 'classification':
                    if self.n_tasks == 1:
                        y = y.long().to(self.device)
                    else:
                        y = y.float().to(self.device)
            for key, value in tokens.items():
                tokens[key] = value.to(self.device)

            inputs = {**tokens, 'labels': y}
            return inputs, y, w

    def build_bnb_config(self) -> Optional[BitsAndBytesConfig]:
        """Build and return a BitsAndBytesConfig for qlora, or None for other strategies.

        Returns
        -------
        Optional[BitsAndBytesConfig]
            A 4-bit quantization config when finetune_strategy is 'qlora', else None.
        """
        if self.finetune_strategy == 'qlora':
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        return None

    def apply_peft(self, model: PreTrainedModel, task_type: str) -> PreTrainedModel:
        """Optionally apply k-bit training preparation and LoRA/QLoRA adapters to the model.

        Parameters
        ----------
        model: PreTrainedModel
            The loaded pretrained model.
        task_type: str
            PEFT task type string — "CAUSAL_LM" or "SEQ_CLS".

        Returns
        -------
        PreTrainedModel
            The model, possibly wrapped with PEFT adapters.
        """
        if self.finetune_strategy == 'qlora':
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=True
            )
        if self.finetune_strategy in ('lora', 'qlora'):
            lora_cfg = LoraConfig(
                r=32,
                lora_alpha=64,
                target_modules=["q_proj", "k_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type=task_type,
            )
            model = get_peft_model(model, lora_cfg)
        return model

    def load_from_pretrained(  # type: ignore
            self,
            model_dir: Optional[str] = None,
            from_hf_checkpoint: bool = False):
        """Load pretrained OLMo weights into the current model instance.

        Overrides `HuggingFaceModel.load_from_pretrained` to support two loading paths:

        **HuggingFace checkpoint** (`from_hf_checkpoint=True`):
            Loads weights from a HuggingFace Hub model ID or a local directory created
            by `save_pretrained`. The randomly-initialised model built during `__init__`
            is deleted first to avoid holding two copies of the model in memory
            simultaneously. The correct model class is selected based on `task`:

            - ``clm``: `AutoModelForCausalLM`
            - ``regression`` / ``mtr`` / ``classification``: `OlmoForSequenceClassification`
              — a custom head not present in the HuggingFace transformers library, which
              adds a linear scoring layer on top of the decoder.

            After loading, `apply_peft` is called to optionally wrap the model with LoRA
            or QLoRA adapters depending on `finetune_strategy`.

        **Local DeepChem checkpoint** (`from_hf_checkpoint=False`):
            Loads weights from a checkpoint saved by `save_checkpoint`. The
            ``module.`` prefix added by PyTorch's DistributedDataParallel is stripped
            from state-dict keys automatically. Output projection weights
            (``classifier.out_proj``, ``classifier.dense``) are dropped before loading
            so that a pretrain checkpoint with a different number of output labels can
            be used to initialise a finetuning model (`strict=False`).

        Parameters
        ----------
        model_dir: str, optional
            HuggingFace Hub model ID, path to a `save_pretrained` directory, or path
            to a directory containing DeepChem checkpoints. Defaults to `self.model_dir`
            if not provided.
        from_hf_checkpoint: bool, default False
            When True, load from a HuggingFace Hub checkpoint using `from_pretrained`.
            When False, load from a local DeepChem checkpoint file.

        Example
        -------
        >>> # Loading from HuggingFace Hub and applying QLoRA adapters:
        >>> model = Olmo(task='regression', tokenizer_path='allenai/olmo-7b-hf',
        ...              finetune_strategy='qlora', config={'torch_dtype': torch.float16})
        >>> model.load_from_pretrained('allenai/olmo-7b-hf', from_hf_checkpoint=True)
        """

        if model_dir is None:
            model_dir = self.model_dir

        if from_hf_checkpoint:
            # FIXME Transformers library has an api like AutoModel.from_pretrained. It allows to
            # initialise and create a model instance directly without requiring a class instance initialisation step.
            # To use `load_from_pretrained` in DeepChem, we need to follow a two step process
            # of initialising class instance and then loading weights via `load_from_pretrained`.

            # init function creates a randomly initialised model. It is deleted before the
            # pretained weights are loaded to reduce peak memory usage (having 2 copies of the model at the same time).

            del self.model
            gc.collect()
            torch.cuda.empty_cache()

            bnb_config = self.build_bnb_config()

            if self.task == 'clm':
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_dir,
                    quantization_config=bnb_config,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.float16,
                    **self.config)
                task_type = "CAUSAL_LM"

            elif self.task in ['mtr', 'regression', 'classification']:
                self.model = OlmoForSequenceClassification.from_pretrained(
                    model_dir,
                    quantization_config=bnb_config,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.float16,
                    problem_type='regression',
                    num_labels=self.n_tasks,
                    **self.config)
                task_type = "SEQ_CLS"

            else:
                self.model = AutoModel.from_pretrained(
                    model_dir,
                    quantization_config=bnb_config,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.float16,
                    **self.config)
                task_type = "CAUSAL_LM"

            self.model = self.apply_peft(self.model, task_type)

        elif not from_hf_checkpoint:
            checkpoints = sorted(self.get_checkpoints(model_dir))
            if len(checkpoints) == 0:
                raise ValueError('No checkpoint found')
            else:
                checkpoint = checkpoints[0]
                data = torch.load(checkpoint, map_location=self.device)
                # Delete keys of output projection layer (last layer) as the number of
                # tasks (projections) in pretrain model and the current model
                # might vary.

                # When using Distributed Data Parallel (DDP) for training models, PyTorch automatically
                # wraps model parameters in a module. prefix. This can cause issues when loading or
                # saving model states because the key names in state_dict differ from their original
                # single-GPU counterparts. To address this, model_state_dict is updated by removing
                # the "module." prefix when saving or loading models.

                data['model_state_dict'] = {
                    key.replace("module.", ""): value
                    for key, value in data['model_state_dict'].items()
                }
                keys = data['model_state_dict'].keys()
                if 'classifier.out_proj.weight' in keys:
                    del data['model_state_dict']['classifier.out_proj.weight']
                if 'classifier.out_proj.bias' in keys:
                    del data['model_state_dict']['classifier.out_proj.bias']
                if 'classifier.dense.bias' in keys:
                    del data['model_state_dict']['classifier.dense.bias']
                if 'classifier.dense.weight' in keys:
                    del data['model_state_dict']['classifier.dense.weight']
                self.model.load_state_dict(data['model_state_dict'],
                                           strict=False)