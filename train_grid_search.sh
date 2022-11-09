#!/bin/bash -x

echo "Starting CycleNER Grid Search"
source "../cycle_ner_venv/bin/activate"

OUTPUT_TXT="train_grid_search.txt"

S_SAMPLES=(10000)
E_SAMPLES=(10000)
S2E_LR=(1e-4 3e-4 1e-3)
E2S_LR=(1e-4 3e-4 1e-3)

EPOCHS=10
BATCH_SIZE=64

touch $OUTPUT_TXT

for S in ${S_SAMPLES[*]}
do
    for E in ${E_SAMPLES[*]}
    do
        # Prepare data for pretraining
        DATA_DIR="./data/${S}s_${E}e"
        TRAIN_DIR="${DATA_DIR}/train"
        EVAL_DIR="${DATA_DIR}/eval"
        TEST_DIR="${DATA_DIR}/test"

        python format_conll_data.py --train_dir $TRAIN_DIR --eval_dir $EVAL_DIR --test_dir $TEST_DIR --s_samples $S --e_samples $E

        for S2E_LR in ${S2E_LR[*]}
        do
            for E2S_LR in ${E2S_LR[*]}
            do
                MODEL_NAME="${S}s_${E}e_${S2E_LR}s2e_${E2S_LR}e2s"
                NOW=`date +"%Y-%m-%d %T"`

                echo "${NOW}: Model Name: ${MODEL_NAME}, S Samples: ${S}, E Samples: ${E}, S2E Learning Rate: ${S2E_LR}, E2S Learning Rate: ${E2S_LR}, Num Epochs: ${EPOCHS}" >> $OUTPUT_TXT

                # Prepare the model and 
                python prepare.py --model_name $MODEL_NAME --model_dir ./models/train_search --train_data_dir $TRAIN_DIR --eval_data_dir $EVAL_DIR --test_data_dir $TEST_DIR

                # Pretrain the model
                python train.py --no-save --model_dir "./models/train_search/${MODEL_NAME}" --summary_writer_dir "./train_search_runs/${MODEL_NAME}" --batch_size $BATCH_SIZE --s2e_lr $S2E_LR --es2_lr $E2S_LR --epochs $EPOCHS 
            done
        done
    done
done
