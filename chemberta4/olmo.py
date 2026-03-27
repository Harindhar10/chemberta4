from typing import Dict, Any, Tuple, Optional
#from deepchem.models.torch_models.hf_models import HuggingFaceModel
from deepchem.models import HuggingFaceModel
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

from transformers import AutoTokenizer, OlmoConfig, OlmoForCausalLM, BitsAndBytesConfig, AutoModelForCausalLM, AutoModel
from modeling_olmo import OlmoForSequenceClassification
from transformers.modeling_utils import PreTrainedModel
try:
    import torch
    has_torch = True
except:
    has_torch = False
import gc, torch



class Olmo(HuggingFaceModel):
    """This class enables training and prediction using Olmo, a decoder-only transformer through the DeepChem API.

    It supports pretraining via causal language modeling, finetuning via regression, 
    classification and multitask regression and they can be specified using 'clm', `regression`, 
    `classification` and `mtr` as arguments to the `task` keyword during model initialisation.

    It uses a tokenizer to create input tokens for the models.
    The default tokenizer model is GPTNeoXTokenizerFast.

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

    Example
    -------
    >>> import os
    >>> import tempfile
    >>> import shutil
    >>> tempdir = tempfile.mkdtemp()

    >>> # preparing dataset
    >>> import pandas as pd
    >>> import deepchem as dc
    >>> smiles = ["CCN(CCSC)C(=O)N[C@@](C)(CC)C(F)(F)F","CC1(C)CN(C(=O)Nc2cc3ccccc3nn2)C[C@@]2(CCOC2)O1"]
    >>> labels = [3.112,2.432]
    >>> df = pd.DataFrame(list(zip(smiles, labels)), columns=["smiles", "task1"])
    >>> with dc.utils.UniversalNamedTemporaryFile(mode='w') as tmpfile:
    ...     df.to_csv(tmpfile.name)
    ...     loader = dc.data.CSVLoader(["task1"], feature_field="smiles", featurizer=dc.feat.DummyFeaturizer())
    ...     dataset = loader.create_dataset(tmpfile.name)

    >>> # pretraining
    >>> from chemberta4.olmo import Olmo
    >>> pretrain_model_dir = os.path.join(tempdir, 'pretrain-model')
    >>> tokenizer_path = "allenai/olmo-7b-hf"
    >>> config = {'torch_dtype': torch.float16}
    >>> pretrain_model = Olmo(task="regression",
            tokenizer_path="allenai/Olmo-7b-hf",
            config = config)
    >>> pretraining_loss = pretrain_model.fit(dataset, nb_epoch=1)

    >>> # finetuning in regression mode
    >>> finetune_model_dir = os.path.join(tempdir, 'finetune-model')
    >>> pretrain_model = Olmo(task="regression",
            tokenizer_path="allenai/Olmo-7b-hf",
            config = config)
    >>> finetune_model.load_from_pretrained(pretrain_model_dir)
    >>> finetuning_loss = finetune_model.fit(dataset, nb_epoch=1)

    >>> # prediction and evaluation
    >>> result = finetune_model.predict(dataset)
    >>> eval_results = finetune_model.evaluate(dataset, metrics=dc.metrics.Metric(dc.metrics.mae_score))

    >>> # removing temporary directory
    >>> if os.path.exists(tempdir):
    ...     shutil.rmtree(tempdir)

    """

    def __init__(self,
                 task: str,
                 tokenizer_path: str = 'allenai/olmo-7b-hf',
                 n_tasks: int = 1,
                 config: Dict[Any, Any] = {},
                 **kwargs):
        self.n_tasks = n_tasks

        self.finetune_strategy = kwargs.get("finetune_strategy", "qlora")
        if self.finetune_strategy not in {"qlora", "lora", "full_finetune"}:
            raise ValueError("Invalid finetune_strategy")

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,
                                                  trust_remote_code=True)
        self.model: PreTrainedModel
        chemberta_config = OlmoConfig(vocab_size=tokenizer.vocab_size,
                                         **config)
        print(chemberta_config)
        if task == 'clm':
            self.model = OlmoForCausalLM(chemberta_config)
        if task == 'mtr':
            chemberta_config.problem_type = 'regression'
            chemberta_config.num_labels = n_tasks
            self.model = OlmoForSequenceClassification(chemberta_config)
        elif task == 'regression':
            chemberta_config.problem_type = 'regression'
            chemberta_config.num_labels = n_tasks
            self.model = OlmoForSequenceClassification(chemberta_config)
        elif task == 'classification':
            if n_tasks == 1:
                chemberta_config.problem_type = 'single_label_classification'
            else:
                chemberta_config.problem_type = 'multi_label_classification'
                chemberta_config.num_labels = n_tasks
                self.model = OlmoForSequenceClassification(chemberta_config)
        else:
            raise ValueError('invalid task specification')

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

    def load_from_pretrained(  # type: ignore
            self,
            model_dir: Optional[str] = None,
            from_hf_checkpoint: bool = False):
        """Load HuggingFace model from a pretrained checkpoint.

        The utility can be used for loading a model from a checkpoint.
        Given `model_dir`, it checks for existing checkpoint in the directory.
        If a checkpoint exists, the models state is loaded from the checkpoint.

        If the option `from_hf_checkpoint` is set as True, then it loads a pretrained
        model using HuggingFace models `from_pretrained` method. This option
        interprets model_dir as a model id of a pretrained model hosted inside a model repo
        on huggingface.co or path to directory containing model weights saved using `save_pretrained`
        method of a HuggingFace model.

        Parameter
        ----------
        model_dir: str
            Directory containing model checkpoint
        from_hf_checkpoint: bool, default False
            Loads a pretrained model from HuggingFace checkpoint.

        Example
        -------
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("allenai/olmo-7b-hf")

        >>> from deepchem.models.torch_models.hf_models import HuggingFaceModel
        >>> from transformers.models.olmo import OlmoForCausalLM, OlmoModel, OlmoConfig
        >>> config = OlmoConfig(vocab_size=tokenizer.vocab_size)
        >>> model = OlmoForCausalLM(config)
        >>> pretrain_model = HuggingFaceModel(model=model, tokenizer=tokenizer, task='clm', model_dir='model-dir')
        >>> pretrain_model.save_checkpoint()

        >>> from modelling_olmo import OlmoForSequenceClassification
        >>> config = OlmoConfig(vocab_size=tokenizer.vocab_size)
        >>> model = OlmoForSequenceClassification(config)
        >>> finetune_model = HuggingFaceModel(model=model, task='classification', tokenizer=tokenizer, model_dir='model-dir')

        >>> finetune_model.load_from_pretrained()

        Note
        ----
        1. Use `load_from_pretrained` method only to load a pretrained model - a
            model trained on a different task like Masked Language Modeling or
            Multitask Regression. To `restore` a model, use the `restore` method.

        2. A pretrain model has different number of target tasks for pretraining and a finetune
            model has different number of target tasks for finetuning. Thus, they both have different
            number of projection outputs in the last layer. To avoid a mismatch
            in the weights of the output projection layer (last layer) between
            the pretrain model and current model, we delete the projection
            layers weights.
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

            self.model.to("cpu")   # not required, but safer
            del self.model
            gc.collect()
            torch.cuda.empty_cache()

            bnb_config = None
            if self.finetune_strategy == 'qlora':
                self.bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16
                )

            if self.task == 'clm':
                self.model = AutoModelForCausalLM.from_pretrained(
                    "allenai/olmo-7b-hf", 
                    quantization_config = self.bnb_config,
                    trust_remote_code=True,
                    device_map = 'cpu',
                    low_cpu_mem_usage = True,
                    **self.config)
    
                self.task_type = "CAUSAL_LM"


            elif self.task in ['mtr', 'regression', 'classification']:
                self.model = OlmoForSequenceClassification.from_pretrained(
                            "allenai/olmo-7b-hf",
                            quantization_config = self.bnb_config,
                            trust_remote_code=True, 
                            device_map = 'cpu',
                            low_cpu_mem_usage = True,
                            **self.config)
    
                self.task_type = "SEQ_CLS"
    
            else:
                self.model = AutoModel.from_pretrained("allenai/olmo-7b-hf",
                                                       quantization_config = self.bnb_config,
                                                       trust_remote_code=True,
                                                       device_map = 'cpu',
                                                       low_cpu_mem_usage = True,
                                                       **self.config)
                self.task_type = "CAUSAL_LM"
    
            if self.finetune_strategy == "qlora":
                self.model = prepare_model_for_kbit_training(
                    self.model, use_gradient_checkpointing=True
                )

            if self.finetune_strategy != "full_finetune":
                lora_cfg = LoraConfig(
                    r=32,
                    lora_alpha=64,
                    target_modules=["q_proj", "k_proj", "v_proj"],
                    lora_dropout=0.05,
                    bias="none",
                    task_type=self.task_type,
                )
            self.model = get_peft_model(self.model, lora_cfg)

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
        