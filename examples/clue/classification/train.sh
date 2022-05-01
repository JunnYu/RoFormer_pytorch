#export TRANSFORMERS_CACHE=/mnt/f/hf/models

ALL_TASKS="iflytek tnews afqmc ocnli cluewsc2020 csl cmnli"
for TASK_NAME in $ALL_TASKS
do
if [ $TASK_NAME == "cluewsc2020" ] ;then
  EPOCHS=30
else
  EPOCHS=10
fi
python run_clue_no_trainer.py \
  --model_name_or_path "junnyu/roformer_v2_chinese_char_base" \
  --task_name $TASK_NAME \
  --max_length 128 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 64 \
  --learning_rate 2e-5 \
  --num_train_epochs $EPOCHS \
  --weight_decay 0.01 \
  --num_warmup_steps_or_radios 0.1 \
  --seed 42 \
  --with_tracking \
  --output_dir ./outputs/$TASK_NAME/
done