# DLCF-DCA (PyABSA-based)

Codes for paper Combining Dynamic Local Context Focus and Dependency Cluster Attention for Aspect-level sentiment classification. submitted to 《Neurocomputing》.

We exploit a efficient and easy-to-use aspect-based sentiment analysis framework PyABSA. Futhermore, we integrate the optimized DLCF-DCA model into this framework.

You can easily train our DLCF-DCA models and design your models based on PyABSA.

To use PyABSA, install the latest version from pip or source code:
```
pip install pyabsa==1.1.24
```



我们开发了一个高效易用的方面级情感分析框架PyABSA，并将优化后的DLCF-DCA模型整合到这个框架之中。

您可以基于PyABSA快速地开始训练DLCF-DCA模型并设计您自己的模型。

您可以通过以下代码来安装PyABSA ：
```
pip install pyabsa==1.1.24
```
## Requirement
* Python >= 3.6 <br> 
* PyTorch >= 1.0 <br> 
* transformers >= 2.4.0 <br> 
* SpaCy >= 2.2

To use our models, you need download `en_core_web_sm` by
`python -m spacy download en_core_web_sm`


##  Model Architecture
![dlcf_dca](pic/dlcf_dca.png)


## Note
Some important scripts to note:
* [dlcf_dca_bert.py](https://github.com/XuMayi/DLCF-DCA/blob/main/pyabsa/core/apc/models/dlcf_dca_bert.py): the source code of DLCF_DCA model.
* [apc_utils_for_dlcf_dca.py](https://github.com/XuMayi/DLCF-DCA/blob/main/pyabsa/core/apc/dataset_utils/apc_utils_for_dlcf_dca.py): preprocess the tokens and calculates the shortest distance to target words and cluster via the Dependency Syntax Parsing Tree.
* [apc_utils.py](https://github.com/XuMayi/DLCF-DCA/blob/main/pyabsa/core/apc/dataset_utils/apc_utils.py): calculates the SynRD from aspect term to target words via the Dependency Syntax Parsing Tree.
* [apc_trainer.py](https://github.com/XuMayi/DLCF-DCA/blob/main/pyabsa/core/apc//training/apc_trainer.py): training process instruction.

## Dataset
Our code will automatically download the datasets in intergrated_datasets folder
* integrated_datasets/apc_datasets/SemEval/laptop14/*.seg: Preprocessed training and testing sentences in SemEval-2014 laptop dataset.
* integrated_datasets/apc_datasets/SemEval/restaurant14/*.seg: Preprocessed training and testing sentences in SemEval-2014 restaurant dataset.
* integrated_datasets/apc_datasets/SemEval/restaurant15/*.seg: Preprocessed training and testing sentences in SemEval-2015 restaurant dataset.
* integrated_datasets/apc_datasets/SemEval/restaurant16/*.seg: Preprocessed training and testing sentences in SemEval-2016 restaurant dataset.
* integrated_datasets/apc_datasets/TShirt/*.seg: Preprocessed training and testing sentences in Tshirt dataset.
* integrated_datasets/apc_datasets/Television/*.seg: Preprocessed training and testing sentences in Television dataset.


## Quick Start of Training and Testing
### 1. Import necessary entries
```
from pyabsa.functional import Trainer
from pyabsa.functional import APCConfigManager
from pyabsa.functional import ABSADatasetList
from pyabsa.functional import APCModelList
```
### 2. Choose a base param_dict
```
apc_config_english = APCConfigManager.get_apc_config_english()
```
### 3. Specify an APC model and alter some hyper-parameters
```
apc_config_english.model = APCModelList.DLCF_DCA_BERT
apc_config_english.lcf = "cdm" # or "cdw"
apc_config_english.dlcf_a = 2
apc_config_english.dca_p = 1
apc_config_english.dca_layer = 3
apc_config_english.dropout = 0.5
apc_config_english.num_epoch = 10
apc_config_english.l2reg = 0.00001
apc_config_english.seed = {0, 1, 2, 3}
apc_config_english.evaluate_begin = 0
```
### 4. Configure runtime setting and running training
```
dataset_path = ABSADatasetList.Restaurant14
sent_classifier = Trainer(config=apc_config_english,
                          dataset=dataset_path,  # train set and test set will be automatically detected
                          checkpoint_save_mode=1,  # =None to avoid save model
                          auto_device=True  # automatic choose CUDA or CPU
                          )
```

## Quick Start of Inferring
We share some checkpoints for the DLCF-DCA models in Google drive.

Our codes will automatically download the checkpoint.

|      checkpoint name        | Laptop14 (acc) |  Laptop14 (f1) |
| :------------------: | :------------: | :-----------: |
| ['dlcf-dca-bert1'](https://drive.google.com/file/d/1w-NtWujPglsvZu4-jC6Vmu8Iz8CvX-1u/view?usp=sharing) |      81.50     |   78.03      |

| checkpoint name       | Restaurant14 (acc) |  Restaurant14 (f1) |
| :--------------------: | :--------------: | :-----------: |
| ['dlcf-dca-bert2'](https://drive.google.com/file/d/1py52V7GmkvjWxrpKICY6h7XaUh9Anw7A/view?usp=sharing)  |     86.79      |    80.53     |

### 1. Import necessary entries
```
import os
from pyabsa import APCCheckpointManager, ABSADatasetList
os.environ['PYTHONIOENCODING'] = 'UTF8'
```

### 2. Assume the sent_classifier and checkpoint
```
sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive', -999: ''}

sent_classifier = APCCheckpointManager.get_sentiment_classifier(checkpoint='dlcf-dca-bert1', #or 'dlcf-dca-bert2'
                                                                auto_device='cuda',  # Use CUDA if available
                                                                sentiment_map=sentiment_map
                                                                )
```
### 3. Configure inferring setting
```
# batch inferring_tutorials returns the results, save the result if necessary using save_result=True
inference_sets = ABSADatasetList.Laptop14
results = sent_classifier.batch_infer(target_file=inference_sets,
                                      print_result=True,
                                      save_result=True,
                                      ignore_error=True,
                                      )
```
### 4. some inferring cases
```
Apple is unmatched in  product quality  , aesthetics , craftmanship , and customer service .  
product quality --> Positive  Real: Positive (Correct)
 Apple is unmatched in product quality ,  aesthetics  , craftmanship , and customer service .  
aesthetics --> Positive  Real: Positive (Correct)
 Apple is unmatched in product quality , aesthetics ,  craftmanship  , and customer service .  
craftmanship --> Positive  Real: Positive (Correct)
 Apple is unmatched in product quality , aesthetics , craftmanship , and  customer service  .  
customer service --> Positive  Real: Positive (Correct)
It is a great size and amazing  windows 8  included !  
windows 8 --> Positive  Real: Positive (Correct)
 I do not like too much  Windows 8  .  
Windows 8 --> Negative  Real: Negative (Correct)
Took a long time trying to decide between one with  retina display  and one without .  
retina display --> Neutral  Real: Neutral (Correct)
 I was also informed that the  components  of the Mac Book were dirty .  
components --> Negative  Real: Negative (Correct)
 the  hardware  problems have been so bad , i ca n't wait till it completely dies in 3 years , TOPS !  
hardware --> Negative  Real: Negative (Correct)
 It 's so nice that the  battery  last so long and that this machine has the snow lion !  
battery --> Positive  Real: Positive (Correct)
 It 's so nice that the battery last so long and that this machine has the  snow lion  !  
snow lion --> Positive  Real: Positive (Correct)
```

## Training on our checkpoint
### 1. Import necessary entries
```
from pyabsa.functional import APCCheckpointManager
from pyabsa.functional import Trainer
from pyabsa.functional import APCConfigManager
from pyabsa.functional import ABSADatasetList
from pyabsa.functional import APCModelList
```
### 2. Choose a base param_dict
```
apc_config_english = APCConfigManager.get_apc_config_english()
```
### 3. Specify an APC model and alter some hyper-parameters
```
apc_config_english.model = APCModelList.DLCF_DCA_BERT
apc_config_english.lcf = "cdw" # or "cdm"
apc_config_english.dlcf_a = 2
apc_config_english.dca_p = 1
apc_config_english.dca_layer = 3
apc_config_english.max_seq_len = 80
apc_config_english.dropout = 0.5
apc_config_english.num_epoch = 10
apc_config_english.l2reg = 0.00001
apc_config_english.seed = {0, 1, 2, 3}
apc_config_english.evaluate_begin = 0
```
### 4. Assume the sent_classifier and checkpoint
```
checkpoint_path = APCCheckpointManager.get_checkpoint('dlcf-dca-bert1')
Laptop14 = ABSADatasetList.Laptop14
sent_classifier = Trainer(config=apc_config_english,
                          dataset=Laptop14,
                          from_checkpoint=checkpoint_path,
                          checkpoint_save_mode=1,
                          auto_device=True
                          )
```

## Acknowledgement
We have based our model development on [PyABSA](https://github.com/yangheng95/PyABSA). Thanks for their contribution.
