chemberta4.data
===============

Dataset class for classification.
It returns dictionaries of ``torch.Tensor`` compatible with PyTorch DataLoader.

.. class:: MoleculeNetDataset(Dataset)

   Unified dataset for MoleculeNet classification and regression tasks.
   Handles single-task, multi-task, and LM-head prompt formatting.
   Supports optional z-score normalization for regression labels.

   .. method:: __init__(df, tokenizer, task_columns, prompt, task_type, experiment_type, max_len=128, use_lm_head=False, label_stats=None, smiles_column="smiles")

      Initializes the dataset from a DataFrame with SMILES strings and target columns.

   .. method:: __len__()

      Returns the number of samples in the dataset.

   .. method:: __getitem__(idx)

      Returns a dict of tensors with ``input_ids``, ``attention_mask``, ``labels``,
      and optionally ``label_mask`` for multi-task datasets.
