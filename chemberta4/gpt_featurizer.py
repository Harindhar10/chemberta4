import pandas as pd
from deepchem.feat import Featurizer
from typing import List
from deepchem.feat import Featurizer
from typing import List
from prompt_templates import PROMPT_TEMPLATES



class PromptFeaturizer(Featurizer):
    """This Featurizer adds prompts to smiles strings in MoleculeNet datasets.
    The prompts are dataset specific and are defined in prompt_templates.py

    Examples
    --------

    >>> import pandas as pd
    >>> import deepchem as dc
    >>> 
    >>> smiles = ["CCN(CCSC)C(=O)N[C@@](C)(CC)C(F)(F)F","CC1(C)CN(C(=O)Nc2cc3ccccc3nn2)C[C@@]2(CCOC2)O1"]
    >>> labels = [3.112,2.432]
    >>> df = pd.DataFrame(list(zip(smiles, labels)), columns=["smiles", "task1"])

    >>> df.to_csv("input_file.csv",index=False)
    >>> loader = dc.data.CSVLoader(["task1"], feature_field="smiles", featurizer=PromptFeaturizer(task_name='bbbp'))
    >>> dataset = loader.create_dataset("/input_file.csv")

    """

    def __init__(self, task_name):

        self.task_name = task_name

    def _featurize(self, datapoint: str, **kwargs) -> List[List[int]]:
        """
        This function adds prompt to a smiles string based on the template
        defined in prompt_templates.py
    
        Parameters
        ----------
        datapoint: str
            Arbitrary smiles sequence to be featurized

        Returns
        -------
        str:
            Text with prompt and smiles
        """

        prompt_template = PROMPT_TEMPLATES[self.task_name]
        
        text = prompt_template.format(smiles=datapoint)

        return text
    

