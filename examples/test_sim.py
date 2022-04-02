import torch
import numpy as np
from roformer import RoFormerForCausalLM, RoFormerConfig
from transformers import BertTokenizer

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
pretrained_model = "junnyu/roformer_chinese_sim_char_base"
tokenizer = BertTokenizer.from_pretrained(pretrained_model)
config = RoFormerConfig.from_pretrained(pretrained_model)
config.is_decoder = True
config.eos_token_id = tokenizer.sep_token_id
config.pooler_activation = "linear"
model = RoFormerForCausalLM.from_pretrained(pretrained_model, config=config)
model.to(device)
model.eval()

def gen_synonyms(text, n=100, k=20):
    ''''含义： 产生sent的n个相似句，然后返回最相似的k个。
    做法：用seq2seq生成，并用encoder算相似度并排序。
    '''
    # 寻找所有相似的句子
    r = []
    inputs1 = tokenizer(text, return_tensors="pt")
    for _ in range(n):
        inputs1.to(device)
        output = tokenizer.batch_decode(model.generate(**inputs1, top_p=0.95, do_sample=True, max_length=128), skip_special_tokens=True)[0].replace(" ","").replace(text, "") # 去除空格，去除原始text文本。
        r.append(output)
    
    # 对相似的句子进行排序
    r = [i for i in set(r) if i != text and len(i) > 0]
    r = [text] + r
    inputs2 = tokenizer(r, padding=True, return_tensors="pt")
    with torch.no_grad():
        inputs2.to(device)
        outputs = model(**inputs2)
        Z = outputs.pooler_output.cpu().numpy()
    Z /= (Z**2).sum(axis=1, keepdims=True)**0.5
    argsort = np.dot(Z[1:], -Z[0]).argsort()
    
    return [r[i + 1] for i in argsort[:k]]

out = gen_synonyms("广州和深圳哪个好？")
print(out)
# ['深圳和广州哪个好？',
#  '广州和深圳哪个好',
#  '深圳和广州哪个好',
#  '深圳和广州哪个比较好。',
#  '深圳和广州哪个最好？',
#  '深圳和广州哪个比较好',
#  '广州和深圳那个比较好',
#  '深圳和广州哪个更好？',
#  '深圳与广州哪个好',
#  '深圳和广州，哪个比较好',
#  '广州与深圳比较哪个好',
#  '深圳和广州哪里比较好',
#  '深圳还是广州比较好？',
#  '广州和深圳哪个地方好一些？',
#  '广州好还是深圳好？',
#  '广州好还是深圳好呢？',
#  '广州与深圳哪个地方好点？',
#  '深圳好还是广州好',
#  '广州好还是深圳好',
#  '广州和深圳哪个城市好？']
