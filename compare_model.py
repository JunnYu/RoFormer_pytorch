import torch
import jieba
import tensorflow as tf
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from model import RoFormerModel, RoFormerTokenizer

jieba.initialize()
config_path = 'E:/BaiduNetdiskDownload/chinese_roformer_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'E:/BaiduNetdiskDownload/chinese_roformer_L-12_H-768_A-12/bert_model.ckpt'
dict_path = 'E:/BaiduNetdiskDownload/chinese_roformer_L-12_H-768_A-12/vocab.txt'
converted_ckpt_path = "pretrained_models/chinese_roformer_base"
tokenizer = Tokenizer(dict_path,
                      do_lower_case=True,
                      pre_tokenize=lambda s: jieba.cut(s, HMM=False))
text = "这里基本保留了唐宋遗留下来的坊巷格局和大量明清古建筑，其中各级文保单位29处，被誉为“里坊制度的活化石”“明清建筑博物馆”！"

#bert4keras
inputs = tokenizer.encode(text)
tf_inputs = [
    tf.convert_to_tensor(inputs[0])[None],
    tf.convert_to_tensor(inputs[1])[None]
]
model = build_transformer_model(config_path=config_path,
                                checkpoint_path=checkpoint_path,
                                model='roformer')
bert4keras_outputs = torch.tensor(model(tf_inputs).numpy())
# pt
roformer_tokenizer = RoFormerTokenizer.from_pretrained(converted_ckpt_path)
pt_model = RoFormerModel.from_pretrained(converted_ckpt_path)
pt_inputs = roformer_tokenizer(text, return_tensors="pt")
with torch.no_grad():
    pt_outputs = pt_model(**pt_inputs)[0]

print("mean diff :", (bert4keras_outputs - pt_outputs).abs().mean())
print("max diff :", (bert4keras_outputs - pt_outputs).abs().max())
