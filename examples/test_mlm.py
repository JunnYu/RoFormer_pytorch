import torch
from roformer import RoFormerForMaskedLM, RoFormerTokenizer

text = "今天[MASK]很好，我[MASK]去公园玩。"
tokenizer = RoFormerTokenizer.from_pretrained("junnyu/roformer_chinese_base")
model = RoFormerForMaskedLM.from_pretrained("junnyu/roformer_chinese_base")

inputs = tokenizer(text, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs).logits[0]

outputs_sentence = ""
for i, id in enumerate(tokenizer.encode(text)):
    if id == tokenizer.mask_token_id:
        tokens = tokenizer.convert_ids_to_tokens(outputs[i].topk(k=5)[1])
        outputs_sentence += "[" + "||".join(tokens) + "]"
    else:
        outputs_sentence += "".join(
            tokenizer.convert_ids_to_tokens([id], skip_special_tokens=True))

print(outputs_sentence)