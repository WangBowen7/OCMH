#!/bin/bash
set -e
for mdoel in 'dgcpn'
do
    for dataname in 'MIRFlickr' 'NUSWIDE' 'MSCOCO'
    do
        for bit in 16 32 64 128
        do
            CUDA_VISIBLE_DEVICES=0 python -u main_lr.py --bits $i --dataname $dataname  --model $model --LR_IMG 0.001 --LR_TXT 0.001 --LR_ORIGIN 0.1  --LR_EXPAND 1  --LAMBDA0 1 --LAMBDA1 0.1 --LAMBDA2 0.1 --log 'main_lr_0.1_0331' --log_best 'best_main_lr_0.1_0331'
        done
    done
done