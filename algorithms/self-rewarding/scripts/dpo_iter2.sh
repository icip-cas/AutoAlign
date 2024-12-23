export GA=4
export LR=3e-8
export TRAIN_BATCH_SIZE=2
export WD=0.1
export REPORT_TO=wandb
export ATA_ROOT="../../"
export DS_CONFIG=${ATA_ROOT}/configs/zero2.json
bash ${ATA_ROOT}/scripts/train_dpo.sh
