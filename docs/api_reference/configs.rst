configs/tasks.yaml
==================

Task registry used to generate prompts and configure experiments. Each entry
defines a dataset by name and maps it to the columns, prompt template, task
type, and monitoring metric consumed by the training pipeline.

When an experiment is launched via ``run_experiment.py --datasets <name>``, the
``get_task()`` utility reads this file and returns a ``SimpleNamespace`` whose
fields drive dataset construction, prompt generation, and trainer setup.

Schema
------

Every task entry supports the following keys:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Key
     - Description
   * - ``task_columns``
     - List of target column names in the dataset CSV. For classification and
       regression tasks these are the label columns. For generation tasks this
       is an empty list. For instruction tuning this contains the instruction,
       input, and output column names.
   * - ``prompt``
     - Natural-language prompt prepended to each SMILES string before
       tokenization. This is the mechanism used to generate task-specific
       prompts for the model (e.g. *"Does this molecule permeate the
       blood-brain barrier?"*). For generation tasks the prompt is simply
       ``"SMILES:"``.
   * - ``task_type``
     - One of ``single_task``, ``multi_task``, ``regression``, or
       ``generation``. Controls how labels are loaded and how the model head
       is configured.
   * - ``experiment_type``
     - One of ``classification``, ``regression``, ``pretraining``, or
       ``instruction``. Selects which training runner to invoke.
   * - ``monitor_metric``
     - Metric watched by ``EarlyStopping`` and ``ModelCheckpoint``
       (e.g. ``val/roc_auc``, ``val/rmse``, ``train/loss``).
   * - ``monitor_mode``
     - ``max`` or ``min`` — whether higher or lower metric values are better.
   * - ``target_column`` *(regression only)*
     - Column name used to compute label normalization statistics.

Supported Tasks
---------------

**Single-Task Classification** — ``bbbp``, ``bace_classification``, ``hiv``,
``clintox``

**Multi-Task Classification** — ``sider`` (27 side-effect endpoints),
``tox21`` (12 toxicity assays)

**Regression** — ``clearance``, ``delaney``, ``freesolv``, ``lipo``,
``bace_regression``

**Generation (Pretraining)** — ``zinc20``, ``pubchem``

**Instruction Tuning** — ``uspto``

Adding a New Task
-----------------

Append an entry to ``configs/tasks.yaml``::

    my_dataset:
      task_columns: [label]
      prompt: "Predict the property of this molecule."
      task_type: single_task
      experiment_type: classification
      monitor_metric: val/roc_auc
      monitor_mode: max

Then run::

    python scripts/run_experiment.py --datasets my_dataset
