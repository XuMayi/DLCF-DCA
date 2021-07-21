# DLCF-DCA
 Codes for paper Combining Dynamic Local Context Focus and Dependency Cluster Attention for Aspect-level sentiment classification. submitted to 《Neurocomputing》.

We integrate and optimize the DLCF_DCA model in [PyABSA](https://github.com/yangheng95/PyABSA).

To quickly train the DCLF_DCA, you can install the [PyABSA](https://github.com/yangheng95/PyABSA) by:
```
pip install -U pyabsa
```
More detail is shown in [PyABSA](https://github.com/yangheng95/PyABSA).

# Requirement
* Python >= 3.6 <br> 
* PyTorch >= 1.0 <br> 
* pytorch-transformers == 1.2.0 <br> 
* SpaCy >= 2.2

To use our models, you need download `en_core_web_sm` by
`python -m spacy download en_core_web_sm`

# Training
```
python train.py --model dlcf_dca
```
#  Model Architecture
![dlcf_dca](pic/dlcf_dca.png)

# Note
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

# Acknowledgement
We have based our model development on https://github.com/songyouwei/ABSA-PyTorch. Thanks for their contribution.

