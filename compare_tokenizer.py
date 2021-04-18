import jieba
from roformer import RoFormerTokenizer
from bert4keras.tokenizers import Tokenizer

dict_path = 'pretrained_models/chinese_roformer_base'
text = "12312格ab局A B cdA,.567 861351 684！今天萨达天 气非常好王企。文保鹅按时发放了的撒这些seqetvgsa国内拉手的喀什。、]P[,./()*7656&【；，‘"
#text = "这里基本保留了唐宋遗留下来的坊巷格局和大量明清古建筑，其中各级文保单位29处，被誉为“里坊制度的活化石”“明清建筑博物馆”！"
bert4keras_tokenizer = Tokenizer(
    dict_path + "/vocab.txt",
    do_lower_case=True,
    pre_tokenize=lambda s: jieba.cut(s, HMM=False))
roformer_tokenizer = RoFormerTokenizer.from_pretrained(dict_path)

bert4keras_tokenizer_input_ids = bert4keras_tokenizer.encode(text)[0]
roformer_tokenizer_input_ids = roformer_tokenizer.encode(text)

print(bert4keras_tokenizer_input_ids == roformer_tokenizer_input_ids)
