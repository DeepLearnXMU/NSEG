#!/bin/bash
datapath=nips

batch=16
lr=1.0

h=300
eh=150
labeldim=50

dp=0.5
mdp=0.5

agg=gate

gnnl=2
gnndp=0.5

modelname=gdp${gnndp}_gl${gnnl}
prefix=lower
CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --model ${modelname} --vocab $datapath/vocab.new.100d.lower.pt \
    --corpus $datapath/train.lower $datapath/train.eg --valid $datapath/val.lower $datapath/val.eg \
    --test $datapath/test.lower $datapath/test.eg --loss 0 \
    --writetrans decoding/${modelname}.devorder --ehid ${eh} --entityemb glove \
    --gnnl ${gnnl} --labeldim ${labeldim} --agg ${agg} --gnndp ${gnndp} \
    --batch_size ${batch} --beam_size 64 --lr ${lr} --seed 1234 \
    --d_emb 100 --d_rnn ${h} --d_mlp ${h} --input_drop_ratio ${dp} --drop_ratio ${mdp} \
    --save_every 50 --maximum_steps 100 --early_stop 5 >${modelname}.train 2>&1 &

