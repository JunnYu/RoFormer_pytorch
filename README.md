# huggingface roformer
## 转换权重
```bash
python convert_roformer_original_tf_checkpoint_to_pytorch.py \
    --tf_checkpoint_path=xxxxxx/chinese_roformer_L-12_H-768_A-12/bert_model.ckpt \
    --roformer_config_file=converted/config.json \
    --pytorch_dump_path=converted/pytorch_model.bin
```
## 比较
```python
python compare.py
mean diff : tensor(4.3925e-07)
max diff : tensor(7.6294e-06)
```