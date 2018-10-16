BATCH_SIZE=1
NUM_WORKER=0
LEARNING_RATE='0.001'
DECAY_STEP=10
python trainval_net.py \
    --dataset vg-sgg \
    --net res101 \
    --bs $BATCH_SIZE \
    --nw $NUM_WORKER \
    --lr $LEARNING_RATE \
    --lr_decay_epoch $DECAY_STEP \
    --cuda \
    --use_tfb


