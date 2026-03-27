
from transformers.modeling_layers import GenericForSequenceClassification
from transformers import OlmoPreTrainedModel, AutoModel
import torch.nn as nn


class OlmoForSequenceClassification(GenericForSequenceClassification, OlmoPreTrainedModel):
    """
    Olmo Model with a sequence classification head on top (a linear layer on top of the pooled output).
    
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
        # Similar to `self.model = AutoModel.from_config(config)` but allows to change the base model name if needed in the child class
        setattr(self, self.base_model_prefix, AutoModel.from_config(config))
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Linear layer gets initialised in full precision even when the model's parameters are
        # half precision. This leads to an error, so the linear layer's weights' dtype is coverted to the 
        # model's parameters' dtype
        self.score = self.score.to(next(self.parameters()).dtype)

        # Initialize weights and apply final processing
        self.post_init()
    