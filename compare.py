import tensorflow as tf
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from model.modeling_roformer import RoFormerModel
import torch
import jieba
jieba.initialize()
config_path = 'E:/BaiduNetdiskDownload/chinese_roformer_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'E:/BaiduNetdiskDownload/chinese_roformer_L-12_H-768_A-12/bert_model.ckpt'
dict_path = 'E:/BaiduNetdiskDownload/chinese_roformer_L-12_H-768_A-12/vocab.txt'
converted_ckpt_path = "outputs/"
tokenizer = Tokenizer(dict_path,
                      do_lower_case=True,
                      pre_tokenize=lambda s: jieba.cut(s, HMM=False))
text = "这里基本保留了唐宋遗留下来的坊巷格局和大量明清古建筑，其中各级文保单位29处，被誉为“里坊制度的活化石”“明清建筑博物馆”！"
inputs = tokenizer.encode(text)
pt_model = RoFormerModel.from_pretrained(converted_ckpt_path)
pt_inputs = {
    "input_ids": torch.tensor(inputs[0]).long()[None],
    "token_type_ids": torch.tensor(inputs[1]).long()[None]
}
with torch.no_grad():
    o1 = pt_model(**pt_inputs)[0]

model = build_transformer_model(config_path=config_path,
                                checkpoint_path=checkpoint_path,
                                model='roformer')
tf_inputs = [
    tf.convert_to_tensor(inputs[0])[None],
    tf.convert_to_tensor(inputs[1])[None]
]
o2 = torch.tensor(model(tf_inputs).numpy())

print((o1 - o2).abs().mean())
print((o1 - o2).abs().max())