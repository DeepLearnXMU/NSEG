Graph based Neural Sentence Ordering
=====================================================================

### Installation

The following packages are needed:

- Python == 3.6
- Pytorch == 0.4
- torchtext


### Dataset Format
Each line in *.lower is a document: sentence_0 <eos> sentence_1 <eos> sentence_2

*.eg:
entity1:i-r means entity1 is in the sentence_i and its role is r.


### Training and Evaluation

CUDA_VISIBLE_DEVICES=0 python -u main.py --model modelname --vocab vocab.100d.pt \
    --corpus train.txt train.eg --valid val.txt val.eg \
    --test test.txt test.eg --loss 0 \
    --writetrans decoding/eval.order --ehid 150 --entityemb glove \
    --gnnl 2 --labeldim 64 --agg gate --gnndp 0.5 \
    --batch_size 16 --beam_size 64 --senenc rnn --lr 1.0 --optimizer Adadelta \
    --d_emb 100 --d_rnn 500 --d_mlp 500 --input_drop_ratio 0.5 --drop_ratio 0.5 \
    --save_every 2 --maximum_steps 100 --early_stop 3 >train.log 2>&1

### Evaluation
CUDA_VISIBLE_DEVICES=0 python main.py --mode test --load_from models/${modelname} \
    --test test.txt test.eg --writetrans decoding/test.order --beam_size 64 >${modelname}.tranlog 2>&1

