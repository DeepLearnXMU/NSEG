Graph based Neural Sentence Ordering
=====================================================================

### Installation

The following packages are needed:

- Python == 3.6
- Pytorch >= 1.0
- torchtext
- Stanford POS tagger


### Dataset Format
Each line in *.lower is a document: sentence_0 <eos> sentence_1 <eos> sentence_2

*.eg:
entity1:i-r means entity1 is in the sentence_i and its role is r.


### Training and Evaluation
bash run.sh


