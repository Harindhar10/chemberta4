import deepchem as dc
import pandas as pd
import numpy as np
import os
import tempfile
from chemberta4.olmo import Olmo
import torch


def smiles_regression_dataset(tmpdir):
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


def smiles_multitask_regression_dataset(tmpdir):
    
    smiles = ["CCN(CCSC)C(=O)N[C@@](C)(CC)C(F)(F)F","CC1(C)CN(C(=O)Nc2cc3ccccc3nn2)C[C@@]2(CCOC2)O1"]
    labels1 = [3.112,2.432]
    labels2 = [7.222,9.124]
    df = pd.DataFrame(list(zip(smiles, labels1, labels2)), columns=["smiles", "task0", "task1"])
    filepath = os.path.join(tmpdir, 'smiles_mtr.csv')
    df.to_csv(filepath)

    loader = dc.data.CSVLoader(["task1","task2"],
                               feature_field="smiles",
                               featurizer=dc.feat.DummyFeaturizer())
    dataset = loader.create_dataset(filepath)
    return dataset


def test_olmo_pretraining(smiles_regression_dataset):
    """Test causal language model pretraining completes without error."""
    tokenizer_path = 'allenai/olmo-7b-hf'
    model = Olmo(task='clm', tokenizer_path=tokenizer_path)
    model.load_from_pretrained('allenai/olmo-7b-hf',from_hf_checkpoint=True)
    
    dataset = smiles_multitask_regression_dataset(tempfile.mkdtemp())
    loss = model.fit(dataset, nb_epoch=1)
    assert loss

def test_olmo_regression():
    """Test single-task regression fit, evaluate, and predict."""
    tokenizer_path = 'allenai/olmo-7b-hf'
    model = Olmo(task="regression", 
                n_tasks=1,
                tokenizer_path=tokenizer_path, 
                config = {'torch_dtype': torch.float16},
                batch_size=2)

    dataset = smiles_regression_dataset(tempfile.mkdtemp())

    loss = model.fit(dataset, nb_epoch=1)
    eval_score = model.evaluate(dataset,
                                metrics=dc.metrics.Metric(
                                dc.metrics.mean_absolute_error))

    assert loss, eval_score
    prediction = model.predict(dataset)
    assert prediction.shape == dataset.y.shape

def test_olmo_classification():
    """Test single-task classification fit, evaluate, and predict."""
    dataset = smiles_regression_dataset(tempfile.mkdtemp())
    y = np.random.choice([0, 1], size=smiles_regression_dataset.y.shape)
    
    dataset = dc.data.NumpyDataset(X=smiles_regression_dataset.X,
                                   y=y,
                                   w=smiles_regression_dataset.w,
                                   ids=smiles_regression_dataset.ids)

    model = Olmo(task="classification", 
                n_tasks=1,
                tokenizer_path= 'allenai/olmo-7b-hf', 
                config = {'torch_dtype': torch.float16},
                batch_size=2)
    loss = model.fit(dataset, nb_epoch=1)
    eval_score = model.evaluate(dataset,
                                metrics=dc.metrics.Metric(
                                    dc.metrics.recall_score))
    assert eval_score, loss
    prediction = model.predict(dataset)
    # logit scores
    assert prediction.shape == (dataset.y.shape[0], 2)


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


def test_olmo_multi_task_regression():
    """Test multi-task regression fit, evaluate, and predict."""
    tokenizer_path = 'allenai/olmo-7b-hf'
    model = Olmo(task="mtr", 
                n_tasks=2,
                tokenizer_path=tokenizer_path, 
                config = {'torch_dtype': torch.float16},
                batch_size=2)
    
    dataset = smiles_multitask_regression_dataset(tempfile.mkdtemp())

    loss = model.fit(dataset, nb_epoch=1)
    eval_score = model.evaluate(dataset,
                                metrics=dc.metrics.Metric(
                                dc.metrics.mean_absolute_error))

    assert loss, eval_score
    prediction = model.predict(dataset)
    assert prediction.shape == dataset.y.shape


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

    loss = model.fit(train_sample, nb_epoch=1)
    eval_score = model.evaluate(test_sample,
                                metrics=dc.metrics.Metric(
                                    dc.metrics.roc_auc_score))
    assert eval_score, loss
    prediction = model.predict(test_sample)
    # logit scores
    assert prediction.shape == (test_sample.y.shape[0], len(tasks))


def test_olmo_lightning_fit_and_predict():
    """Test QLoRA regression training and prediction via PyTorch Lightning DDP."""
    from deepchem.models.lightning import LightningTorchModel

    tokenizer_path = 'allenai/olmo-7b-hf'

    model = Olmo(task="regression",
                tokenizer_path=tokenizer_path,
                finetune_strategy = 'qlora',
                config = {'torch_dtype': torch.float16},
                batch_size = 2)

    dataset = smiles_regression_dataset(tempfile.mkdtemp())
    
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

def test_lora_qlora():
    """Test that LoRA and QLoRA adapters are applied at init with correct trainable parameter structure."""
    from peft import PeftModel

    for strategy in ('lora', 'qlora'):
        model = Olmo(task='regression',
                     finetune_strategy=strategy,
                     tokenizer_path='allenai/olmo-7b-hf',
                     config={'torch_dtype': torch.float16})

        assert isinstance(model.model, PeftModel)

        model.model.print_trainable_parameters()

        trainable = [n for n, p in model.model.named_parameters() if p.requires_grad]
        assert len(trainable) > 0

        # All trainable params must be either LoRA adapter weights or the task head
        assert all('lora_' in n or 'score' in n for n in trainable)

        # At least some LoRA adapter params must be trainable
        assert any('lora_' in n for n in trainable)

        # Base transformer backbone weights must be frozen
        frozen = [n for n, p in model.model.named_parameters() if not p.requires_grad]
        assert len(frozen) > 0

