import deepchem as dc
import pandas as pd
import numpy as np
import os
import tempfile
from chemberta4.olmo import Olmo
import torch
import pytest

try:
    import torch
    gpu_available = torch.cuda.is_available() and torch.cuda.device_count() > 1
except ImportError:
    gpu_available = False

try:
    import lightning as L
    from deepchem.models.lightning.trainer import LightningTorchModel
    PYTORCH_LIGHTNING_IMPORT_FAILED = False
except ImportError:
    PYTORCH_LIGHTNING_IMPORT_FAILED = True


os.environ["TOKENIZERS_PARALLELISM"] = "false"  # to avoid deadlocks in tokenization due to parallel processing already done by dataloader.


pytestmark = [
    pytest.mark.skipif(not gpu_available,
                       reason="No GPU available for testing"),
    pytest.mark.skipif(PYTORCH_LIGHTNING_IMPORT_FAILED,
                       reason="PyTorch Lightning is not installed")
]


@pytest.fixture(scope="function")
def smiles_regression_dataset(tmpdir):
    """Creates a single-task regression dataset with two SMILES molecules and continuous labels."""
    smiles = [
        "CCN(CCSC)C(=O)N[C@@](C)(CC)C(F)(F)F",
        "CC1(C)CN(C(=O)Nc2cc3ccccc3nn2)C[C@@]2(CCOC2)O1"
    ]
    labels = [3.112, 2.432]
    df = pd.DataFrame(list(zip(smiles, labels)), columns=["smiles", "task1"])
    filepath = os.path.join(tmpdir, 'smiles_reg.csv')
    df.to_csv(filepath)

    loader = dc.data.CSVLoader(["task1"],
                               feature_field="smiles",
                               featurizer=dc.feat.DummyFeaturizer())
    dataset = loader.create_dataset(filepath)
    return dataset

@pytest.fixture(scope="function")
def smiles_multitask_regression_dataset(tmpdir):
    """Creates a two-task regression dataset with two SMILES molecules and two sets of continuous labels."""
    smiles = ["CCN(CCSC)C(=O)N[C@@](C)(CC)C(F)(F)F","CC1(C)CN(C(=O)Nc2cc3ccccc3nn2)C[C@@]2(CCOC2)O1"]
    labels1 = [3.112,2.432]
    labels2 = [7.222,9.124]
    df = pd.DataFrame(list(zip(smiles, labels1, labels2)), columns=["smiles", "task0", "task1"])
    filepath = os.path.join(tmpdir, 'smiles_mtr.csv')
    df.to_csv(filepath)

    loader = dc.data.CSVLoader(["task0","task1"],
                               feature_field="smiles",
                               featurizer=dc.feat.DummyFeaturizer())
    dataset = loader.create_dataset(filepath)
    return dataset


@pytest.mark.torch
def test_olmo_pretraining(smiles_regression_dataset):
    """Test causal language model pretraining completes without error."""
    tokenizer_path = 'allenai/olmo-7b-hf'
    model = Olmo(task='clm', tokenizer_path=tokenizer_path)
    model.load_from_pretrained('allenai/olmo-7b-hf',from_hf_checkpoint=True)

    dataset = smiles_regression_dataset
    loss = model.fit(dataset, nb_epoch=1)
    assert loss

@pytest.mark.torch
def test_olmo_regression(smiles_regression_dataset):
    """Test single-task regression fit, evaluate, and predict."""
    tokenizer_path = 'allenai/olmo-7b-hf'
    model = Olmo(task="regression",
                n_tasks=1,
                tokenizer_path=tokenizer_path,
                config = {'torch_dtype': torch.float16},
                batch_size=2)

    model.load_from_pretrained(model_dir='allenai/olmo-7b-hf',from_hf_checkpoint=True)
    dataset = smiles_regression_dataset

    loss = model.fit(dataset, nb_epoch=1)
    eval_score = model.evaluate(dataset,
                                metrics=dc.metrics.Metric(
                                dc.metrics.mean_absolute_error))

    assert loss, eval_score
    prediction = model.predict(dataset)
    assert prediction.shape == dataset.y.shape

@pytest.mark.torch
def test_olmo_classification(smiles_regression_dataset):
    """Test single-task classification fit, evaluate, and predict."""
    dataset = smiles_regression_dataset
    y = np.random.choice([0, 1], size=dataset.y.shape)

    dataset = dc.data.NumpyDataset(X=dataset.X,
                                   y=y,
                                   w=dataset.w,
                                   ids=dataset.ids)

    model = Olmo(task="classification", 
                n_tasks=1,
                tokenizer_path= 'allenai/olmo-7b-hf', 
                config = {'torch_dtype': torch.float16},
                batch_size=2)
    model.load_from_pretrained(model_dir='allenai/olmo-7b-hf',from_hf_checkpoint=True)

    loss = model.fit(dataset, nb_epoch=1)
    eval_score = model.evaluate(dataset,
                                metrics=dc.metrics.Metric(
                                    dc.metrics.recall_score))
    assert eval_score, loss
    prediction = model.predict(dataset)
    # logit scores
    assert prediction.shape == (dataset.y.shape[0], 2)


@pytest.mark.torch
def test_chemberta_save_reload(tmpdir):
    """Test that a saved checkpoint is restored with identical model weights."""
    tokenizer_path = 'allenai/olmo-7b-hf'
    model = Olmo(task='regression',
                      tokenizer_path=tokenizer_path,
                      model_dir=tmpdir)

    model._ensure_built()
    model.save_checkpoint()

    model_new = Olmo(task='regression',
                          tokenizer_path=tokenizer_path,
                          model_dir=tmpdir)
    model_new.restore()

    old_state = model.model.state_dict()
    new_state = model_new.model.state_dict()
    matches = [
        torch.allclose(old_state[key], new_state[key])
        for key in old_state.keys()
    ]

    # all keys values should match
    assert all(matches)


@pytest.mark.torch
def test_olmo_multi_task_regression(smiles_multitask_regression_dataset):
    """Test multi-task regression fit, evaluate, and predict."""
    tokenizer_path = 'allenai/olmo-7b-hf'
    model = Olmo(task="mtr",
                n_tasks=2,
                tokenizer_path=tokenizer_path,
                config = {'torch_dtype': torch.float16},
                batch_size=2)

    model.load_from_pretrained(from_hf_checkpoint=True)

    dataset = smiles_multitask_regression_dataset

    loss = model.fit(dataset, nb_epoch=1)
    eval_score = model.evaluate(dataset,
                                metrics=dc.metrics.Metric(
                                dc.metrics.mean_absolute_error))

    assert loss, eval_score
    prediction = model.predict(dataset)
    assert prediction.shape == dataset.y.shape


@pytest.mark.torch
def test_olmo_multitask_classification():
    """Test multi-task classification fit, evaluate, and predict on ClinTox."""
    loader = dc.molnet.load_clintox(featurizer=dc.feat.DummyFeaturizer())
    tasks, dataset, transformers = loader
    train, val, test = dataset

    train_sample = train.select(range(10))
    test_sample = test.select(range(10))
    
    model = Olmo(task="classification",
            n_tasks=len(tasks),
            tokenizer_path="allenai/Olmo-7b-hf",
            config = {'torch_dtype': torch.float16,},
            batch_size = 2)

    model.load_from_pretrained(model_dir='allenai/olmo-7b-hf',from_hf_checkpoint=True)


    loss = model.fit(train_sample, nb_epoch=1)
    eval_score = model.evaluate(test_sample,
                                metrics=dc.metrics.Metric(
                                    dc.metrics.roc_auc_score))
    assert eval_score, loss
    prediction = model.predict(test_sample)
    # logit scores
    assert prediction.shape == (test_sample.y.shape[0], len(tasks))


@pytest.mark.torch
def test_olmo_lightning_fit_and_predict(smiles_regression_dataset):
    """Test QLoRA regression training and prediction via PyTorch Lightning DDP."""
    from deepchem.models.lightning import LightningTorchModel

    tokenizer_path = 'allenai/olmo-7b-hf'

    model = Olmo(task="regression",
                tokenizer_path=tokenizer_path,
                finetune_strategy = 'qlora',
                config = {'torch_dtype': torch.float16},
                batch_size = 2)

    model.load_from_pretrained(model_dir='allenai/olmo-7b-hf',from_hf_checkpoint=True)

    dataset = smiles_regression_dataset
    
    trainer = LightningTorchModel(
    model=model,
    batch_size=2,
    max_epochs=1,
    enable_progress_bar=True,
    accelerator="gpu",
    strategy = "ddp",
    devices = -1,
    log_every_n_steps=1
    )

    trainer.fit(dataset, num_workers=0)
    predictions = trainer.predict(dataset)

    assert len(predictions) > 0
    assert isinstance(predictions, np.ndarray)
    # The final prediction shape should be (n_samples, n_tasks)
    assert predictions.shape == (2, 1)

@pytest.mark.torch
def test_olmo_load_from_pretrained(tmpdir):
    """Test that base model weights are correctly transferred from a pretrained CLM checkpoint 
    to a regression model."""
    pretrain_model_dir = os.path.join(tmpdir, 'pretrain')
    finetune_model_dir = os.path.join(tmpdir, 'finetune')
    tokenizer_path = 'allenai/olmo-7b-hf'
    pretrain_model = Olmo(task='clm',
                               tokenizer_path=tokenizer_path,
                               model_dir=pretrain_model_dir)
    pretrain_model.save_checkpoint()

    finetune_model = Olmo(task='regression',
                               tokenizer_path=tokenizer_path,
                               model_dir=finetune_model_dir)
    finetune_model.load_from_pretrained(pretrain_model_dir)

    # check weights match
    pretrain_model_state_dict = pretrain_model.model.state_dict()
    finetune_model_state_dict = finetune_model.model.state_dict()

    pretrain_base_model_keys = [
        key for key in pretrain_model_state_dict.keys() if 'olmo' in key
    ]
    matches = [
        torch.allclose(pretrain_model_state_dict[key],
                       finetune_model_state_dict[key])
        for key in pretrain_base_model_keys
    ]

    assert all(matches)

@pytest.mark.torch
def test_lora_qlora():
    """Test that LoRA and QLoRA adapters are applied at init with correct trainable parameter structure."""
    from peft import PeftModel

    for strategy in ('lora', 'qlora'):
        model = Olmo(task='regression',
                     finetune_strategy=strategy,
                     tokenizer_path='allenai/olmo-7b-hf',
                     config={'torch_dtype': torch.float16})

        model.load_from_pretrained(model_dir='allenai/olmo-7b-hf',from_hf_checkpoint=True)

        assert isinstance(model.model, PeftModel)

        trainable = [n for n, p in model.model.named_parameters() if p.requires_grad]
        assert len(trainable) > 0

        # All trainable params must be either LoRA adapter weights or the task head
        assert all('lora_' in n or 'score' in n for n in trainable)

        # At least some LoRA adapter params must be trainable
        assert any('lora_' in n for n in trainable)

        # Base transformer backbone weights must be frozen
        frozen = [n for n, p in model.model.named_parameters() if not p.requires_grad]
        assert len(frozen) > 0

