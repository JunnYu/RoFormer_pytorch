ALL_TASKS="cmnli iflytek tnews afqmc ocnli cluewsc2020 csl"

for TASK_NAME in $ALL_TASKS
do
python run_clue_predict_no_trainer.py \
  --model_name_or_path "outputs/$TASK_NAME/epoch_best" \
  --task_name $TASK_NAME
done