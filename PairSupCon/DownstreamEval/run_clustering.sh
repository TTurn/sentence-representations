python eval_cluster.py \
    --eval_instance local \
    --pretrained_dir path-to-your-ckpt-for-evaluation \
    --pdataname nli_train_posneg \
    --mode pairsupcon \
    --bert bertbase \
    --contrast_type HardNeg \
    --temperature 0.05 \
    --beta 1 \
    --lr 5e-06 \
    --lr_scale 100 \
    --p_batchsize 1024 \
    --pretrain_epoch 3 \
    --pseed 0 \
    --seed 0 \
    --eval_epoch 3 \
    --device_id 0 
