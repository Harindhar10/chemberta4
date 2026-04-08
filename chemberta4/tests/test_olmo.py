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
    filepath = os.path.join(tmpdir, 'smiles.csv')
    df.to_csv(filepath)

    loader = dc.data.CSVLoader(["task1"],
                               feature_field="smiles",
                               featurizer=dc.feat.DummyFeaturizer())
    dataset = loader.create_dataset(filepath)
    return dataset

def test_olmo_fit_and_predict():
    """
    
    """
    from deepchem.models.torch_models.olmo import Olmo
    import torch
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


def test_chemberta_load_weights_from_hf_hub():
    pretrained_model_path = 'allenai/olmo-7b-hf'
    tokenizer_path = 'allenai/olmo-7b-hf'
    model = Olmo(task='regression', tokenizer_path=tokenizer_path, config = {'torch_dtype': torch.float16})
    old_model_id = id(model.model)
    model.load_from_pretrained(pretrained_model_path, from_hf_checkpoint=True)
    new_model_id = id(model.model)
    # new model's model attribute is an entirely new model initiated by AutoModel.load_from_pretrained
    # and hence it should have a different identifier
    assert old_model_id != new_model_id
