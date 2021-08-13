# PyABSA
We exploit a efficient and easy-to-use aspect-based sentiment analysis framework PyABSA. Futhermore, we integrate the optimized DLCF-DCA model into this framework.

You can easily train our DLCF-DCA models and design your models based on PyABSA.

To use PyABSA, install the latest version from pip or source code:
```
pip install -U pyabsa
```
More detail is shown in [PyABSA](https://github.com/yangheng95/PyABSA).


我们开发了一个高效易用的方面级情感分析框架PyABSA，并将优化后的DLCF-DCA模型整合到这个框架之中。

您可以基于PyABSA快速地开始训练DLCF-DCA模型并设计您自己的模型。

您可以通过以下代码来安装PyABSA ：
```
pip install -U pyabsa
```
更多的细节可以参考[PyABSA](https://github.com/yangheng95/PyABSA) 。

## Quick Start
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
### 3. Specify an APC model and alter some hyper-parameters (if necessary)
```
apc_config_english.model = APCModelList.DLCF_DCA_BERT
apc_config_english.similarity_threshold = 1
apc_config_english.max_seq_len = 80
apc_config_english.dropout = 0.5
apc_config_english.log_step = 5
apc_config_english.num_epoch = 10
apc_config_english.evaluate_begin = 0
apc_config_english.l2reg = 0.00001
apc_config_english.seed = {1, 2, 3}
apc_config_english.cross_validate_fold = -1  # disable cross_validate
apc_config_english.use_syntax_based_SRD = True
```
### 4. Configure runtime setting and running training
```
Laptop14 = ABSADatasetList.Laptop14
sent_classifier = Trainer(config=apc_config_english,
                          dataset=Laptop14,  # train set and test set will be automatically detected
                          checkpoint_save_mode=1,  # =None to avoid save model
                          auto_device=True  # automatic choose CUDA or CPU
                          )
```

# DLCF-DCA
 Codes for paper Combining Dynamic Local Context Focus and Dependency Cluster Attention for Aspect-level sentiment classification. submitted to 《Neurocomputing》.

## Requirement
* Python >= 3.6 <br> 
* PyTorch >= 1.0 <br> 
* pytorch-transformers == 1.2.0 <br> 
* SpaCy >= 2.2

To use our models, you need download `en_core_web_sm` by
`python -m spacy download en_core_web_sm`

## Training
```
python train.py --model dlcf_dca
```
##  Model Architecture
![dlcf_dca](pic/dlcf_dca.png)

## Note
Some important scripts to note:
* datasets/semeval14/*.seg: Preprocessed training and testing sentences in SemEval2014.
* datasets/semeval15/*.seg: Preprocessed training and testing sentences in SemEval2015.
* datasets/semeval16/*.seg: Preprocessed training and testing sentences in SemEval2016.
* models/dlcf_dca.py: the source code of DLCF_DCA model.
* data_utils.py/ABSADataSet class: preprocess the tokens and calculates the shortest distance to target words and cluster via the Dependency Syntax Parsing Tree.

## Out of Memory
Since BERT models require a lot of memory. If the out-of-memory problem while training the model, here are the ways to mitigate the problem:
1. Reduce the training batch size ( batch_size = 4 or 8 )
2. Reduce the longest input sequence ( max_seq_len = 40 or 60 )
3. Set `use_single_bert = true` to use a unique BERT layer to model for both local and global contexts

## Acknowledgement
We have based our model development on https://github.com/songyouwei/ABSA-PyTorch. Thanks for their contribution.

