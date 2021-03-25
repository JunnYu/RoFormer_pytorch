CURRENT_DIR=`pwd`
export MODEL_DIR=$CURRENT_DIR/pretrained_models/chinese_roformer_base
export DATA_DIR=$CURRENT_DIR/dataset
export OUTPUR_DIR=$CURRENT_DIR/outputs
export TASK_NAME=cail2019_scm

#-----------training-----------------
python task_text_classification_cail2019_scm.py \
  --model_type=roformer \
  --model_path=$MODEL_DIR \
  --task_name=$TASK_NAME \
  --do_train \
  --do_eval \
  --eval_all_checkpoints \
  --gpu=0 \
  --monitor=eval_acc \
  --data_dir=$DATA_DIR/${TASK_NAME}/ \
  --train_max_seq_length=1024 \
  --eval_max_seq_length=1024 \
  --gradient_accumulation_steps 4 \
  --per_gpu_train_batch_size=2 \
  --per_gpu_eval_batch_size=4 \
  --learning_rate=6e-6 \
  --num_train_epochs=20.0 \
  --logging_steps=-1 \
  --save_steps=-1 \
  --output_dir=$OUTPUR_DIR/${TASK_NAME}_output/ \
  --overwrite_output_dir \
  --seed=42
