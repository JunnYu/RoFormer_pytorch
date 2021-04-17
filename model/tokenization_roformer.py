import jieba
from transformers import BertTokenizer
from transformers import BasicTokenizer


class CustomBasicTokenizer(BasicTokenizer):
    def __init__(self,
                 vocab,
                 do_lower_case=True,
                 never_split=None,
                 tokenize_chinese_chars=True,
                 strip_accents=None):
        super().__init__(do_lower_case=do_lower_case,
                         never_split=never_split,
                         tokenize_chinese_chars=tokenize_chinese_chars,
                         strip_accents=strip_accents)

        self.vocab = vocab

    def _tokenize_chinese_chars(self, text):
        output = []
        '''
        1、输入一个句子s，用pre_tokenize先分一次词，得到[w1,w2,…,wl]；
        2、遍历各个wi，如果wi在词表中则保留，否则将wi用BERT自带的tokenize函数再分一次；
        3、将每个wi的tokenize结果有序拼接起来，作为最后的tokenize结果。
        '''
        for wholeword in jieba.cut(text, HMM=False):
            if wholeword in self.vocab:
                output.append(" ")
                output.append(wholeword)
                output.append(" ")
            else:
                for char in wholeword:
                    cp = ord(char)
                    if self._is_chinese_char(cp):
                        output.append(" ")
                        output.append(char)
                        output.append(" ")
                    else:
                        output.append(char)
        return "".join(output)


class RoFormerTokenizer(BertTokenizer):
    def __init__(self,
                 vocab_file,
                 do_lower_case=True,
                 do_basic_tokenize=True,
                 never_split=None,
                 unk_token="[UNK]",
                 sep_token="[SEP]",
                 pad_token="[PAD]",
                 cls_token="[CLS]",
                 mask_token="[MASK]",
                 tokenize_chinese_chars=True,
                 strip_accents=None,
                 **kwargs):
        super().__init__(vocab_file,
                         do_lower_case=do_lower_case,
                         do_basic_tokenize=do_basic_tokenize,
                         never_split=never_split,
                         unk_token=unk_token,
                         sep_token=sep_token,
                         pad_token=pad_token,
                         cls_token=cls_token,
                         mask_token=mask_token,
                         tokenize_chinese_chars=tokenize_chinese_chars,
                         strip_accents=strip_accents,
                         **kwargs)
        if self.do_basic_tokenize:
            self.basic_tokenizer = CustomBasicTokenizer(
                vocab=self.vocab,
                do_lower_case=do_lower_case,
                never_split=never_split,
                tokenize_chinese_chars=tokenize_chinese_chars,
                strip_accents=strip_accents,
            )
