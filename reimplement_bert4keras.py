# import bert4keras
import bert4keras.models as M
import numpy as np
import unicodedata
import re

def load_vocab(dict_path,
               encoding='utf-8',
               simplified=False,
               startwith=None):
    """从bert的词典文件中读取词典
    """
    token_dict = {}
    with open(dict_path, encoding=encoding) as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)

    return token_dict

# 这个encode(), 是BasicTokenizer写的, 说实话, 想重写不直接
class BasicTokenizer(object):
    """分词器基类
    """

    def __init__(self,
                 token_start='[CLS]',
                 token_end='[SEP]',
                 do_lower_case=False):
        """初始化
        """
        self._token_pad = '[PAD]'
        self._token_unk = '[UNK]'
        self._tokne_mask = '[MASK]'
        self._token_start = token_start
        self._token_end = token_end
        self._do_lower_case = do_lower_case

    def _tokenize(self, text):
        """基本分类器函数,需要在继承的类中自己实现
        """
        # return []
        raise NotImplementedError

    def token_to_id(self, token):
        """token转换为相对应的id序列
        """
        return NotImplementedError


class Tokenizer(BasicTokenizer):
    """实现自苏建林的Bert4keras
    https://github.com/bojone/bert4keras
    """

    def __init__(self,
                 token_dict,
                 token_start='[CLS]',
                 token_end='[SEP]',
                 do_lower_case=False,
                 *args, **kwargs):
        """初始化
        """
        super().__init__(*args, **kwargs)

        self._token_pad = '[PAD]'
        self._token_unk = '[UNK]'
        self._tokne_mask = '[MASK]'
        self._token_start = token_start
        self._token_end = token_end
        self._do_lower_case = do_lower_case

        token_dict = load_vocab(token_dict)  # 返回一个字典d, d[token] = id

        self._token_dict = token_dict
        self._vocab_size = len(token_dict)

        for token in ['pad', 'unk', 'mask', 'start', 'end']:
            try:
                # getattr(object, name[, default]) -> value, get attribute
                # getattr(x, 'y') is equivalent to x.y
                _token_id = token_dict[getattr(self, '_token_%s' % token)]
                # setattr(x, 'y', v) is equivalent to 'x.y = v', set attribute
                setattr(self, '_token_%s_id' % token, _token_id)
            except:
                pass

    def encode(self,
               first_text,
               second_text=None,
               max_length=None,
               first_length=None,
               second_length=None):
        """输出文本对应的token, id和segment id
        如果传入first_length, 则强行padding第一个句子到指定长度
        second_length同理
        """
        if isinstance(first_text, str):
            first_tokens = self.tokenize(first_text)  # ok tokenize写完了
        else:
            first_tokens = first_text

        if second_text is None:
            second_tokens = None
        elif isinstance(second_text, str):
            # 如果_token_start不是空串 idx就是1 否则0
            idx = int(bool(self._token_start))
            # 如果_token_start不是空串, 经过tokenize(second_text)就以'[CLS]'开头
            # 需要去掉
            second_tokens = self.tokenize(second_text)[idx:]
        else:
            second_tokens = second_text

        if max_length is not None:
            self.truncate_sequence(max_length, first_tokens, second_tokens, -2)

        first_token_ids = self.tokens_to_ids(first_tokens)
        if first_length is not None:
            first_token_ids = first_token_ids[:first_length]
            first_token_ids.extend([self._token_pad_id] *
                                   (first_length - len(first_token_ids)))
        first_segment_ids = [0] * len(first_token_ids)  # 这里的first_token_ids已经被padding过了, 因此定长

        if second_text is not None:
            second_token_ids = self.tokens_to_ids(second_tokens)
            if second_length is not None:
                second_token_ids = second_token_ids[:second_length]
                second_token_ids.extend([self._token_pad_id] * (second_length - len(second_token_ids)))
            second_segment_ids = [1] * len(second_token_ids)

            first_token_ids.extend(second_token_ids)
            first_segment_ids.extend(second_segment_ids)

        return first_token_ids, first_segment_ids

    def tokens_to_ids(self, tokens):
        """tokens列表变为ids列表
        """
        return [self._token_dict.get(token, self._token_unk_id) for token in tokens]

    def tokenize(self, text, max_length=None):
        """分词函数
        1. 将unicode转换为ASCII, 然后转小写(如果lower case)
        2. 分词[过程: 1. 标点, 中文字符: 前后加空格,
                     2. 空格: 用统一的空格' '代替,
                     3. 控制字符: 跳过
                     4. 拆分, 经过以上四步得到的新的字符串, 得到列表
                     5. 字内切分, 对列表内每个元素进行, 得到最终的列表]
        3. 开头加入_token_start(如果有)
        4. 结尾加入_token_end(如果有)
        5. 截断到max_length长度(如果有)
        """
        if self._do_lower_case:
            text = unicodedata.normalize('NFD', text)
            text = ''.join([ch for ch in text if unicodedata.category(ch) != 'Mn'])
            text = text.lower()

        tokens = self._tokenize(text)
        if self._token_start is not None:
            """list_name.insert(index, element)
            在list的给定index位置插入element
            >>> list1 = [ 1, 2, 3, 4, 5, 6, 7 ]  
            >>> # insert 10 at 4th index  
            >>> list1.insert(4, 10)  
            >>> print(list1) 
            [1, 2, 3, 4, 10, 5, 6, 7]
            在最前面插入[CLS]
            """
            tokens.insert(0, self._token_start)
        if self._token_end is not None:
            # 在最后插入[SEP]
            tokens.append(self._token_end)

        if max_length is not None:
            # 这里只需要first_sequence是因为没有second_sequence
            # -2是因为最后是[SEP], 去掉的是[SEP]前面的一个字符
            self.truncate_sequence(max_length, tokens, None, -2)

        return tokens

    def _tokenize(self, text):
        """基本分词函数, 被BasicTokenizer类的tokenize函数调用
        分词过程:
        1. 是标点, 中文字符: 用空格分开, 空格重复没关系
        2. 是泛泛的空格: 用统一的空格' '代替
        3. 控制字符: 跳过
        4. 拆分, 经过以上四步得到的新的字符串, 得到列表
        5. 对列表内每个元素进行字内切分, 得到最终的列表
        """
        spaced = ''
        for ch in text:
            if self._is_punctuation(ch) or self._is_cjk_character(ch):
                # 这样不会出现两个空格连在一起吗?
                # 我不懂
                spaced += ' ' + ch + ' '
            elif self._is_space(ch):
                spaced += ' '
            elif ord(ch) == 0 or ord(ch) == 0xfffd or self._is_control(ch):
                continue
            else:
                spaced += ch

        tokens = []
        # str.strip(): 删除字符串的前面的空格和后面的空格, 返回副本
        # S.split(sep=None, maxsplit=-1) -> list of strings
        # 如果没有指定sep, 那么就是所有的空格
        for word in spaced.strip().split():
            # L.extend(iterable) -> None
            # extend list by appending elements from the iterable
            tokens.extend(self._word_piece_tokenize(word))

        return tokens

    def _word_piece_tokenize(self, word):
        """把word细分成subword
        """
        if word in self._token_dict:
            return [word]

        tokens = []
        start, stop = 0, 0
        while start < len(word):
            stop = len(word)
            while stop > start:
                sub = word[start: stop]
                if start > 0:
                    sub = '##' + sub
                    # 这里的目的就是找到"##阿"的样子
                    # 会重复找, 第一次"##阿比词低"
                    # 第二次"##阿比词"
                    # 第三次"##阿比"
                    # 第四次"##阿", 在字典中, break
                    if sub in self._token_dict: break
                    stop -= 1
            if start == stop:
                # 如果连最小的"##阿"也不在字典里, 就不管字典了,
                # 首先"##阿"加入tokens, 然后start加一, 跳过这个字搜索后面的字
                stop += 1
            # 最终目的就是在这里把sub给append进tokens列表
            # 从后往前搜索, 而不是从前往后搜索, 原因主要是希望找到最大匹配的字
            tokens.append(sub)
            start = stop

        return tokens

    def truncate_sequence(self,
                          max_length,
                          first_sequence,
                          second_sequence=None,
                          pop_index=-1):
        """截断总长度
        # 如果两个串的总长度超过了max_length, 那么
        # 长的串从后pop, 直到长度相等短的串, 如果仍超过
        # 然后两者轮流pop, 直到两者长度之和小于max_length
        """
        if second_sequence is None:
            second_sequence = []

        while True:
            total_length = len(first_sequence) + len(second_sequence)
            if total_length <= max_length:
                break
            elif len(first_sequence) > len(second_sequence):
                first_sequence.pop(pop_index)
            else:
                second_sequence.pop(pop_index)

    @staticmethod
    def _is_space(ch):
        """空格类字符判断
        """
        # 'Zs' 里面有各种空白符, 粘贴展示效果不好
        # 但是没有TAB LF CR 因为它们在Cc(Other Control里)
        # TAB: \t
        # https://www.compart.com/en/unicode/category/Zs
        # LF: \n (Line Feed) used to signify the end of a line of text and the start of a new one
        # CR: \r (Carriage Return) used to reset a device's position to the beginning of a line of text
        return ch == ' ' or \
               ch == '\n' or \
               ch == '\r' or \
               ch == '\t' or \
               unicodedata.category(ch) == 'Zs'

    @staticmethod
    def _is_control(ch):
        """控制类字符判断
        Cc: Other, control
        Cf: Other, format
        """
        return unicodedata.category(ch) in ['Cc', 'Cf']

    def ids_to_tokens(self, ids):
        """类似id_to_token, 但是ids是iterable
        """
        return list(self.id_to_token(id) for id in ids)

    @staticmethod
    def _is_special(ch):
        """判断是否有特殊含义的符号
        # https://www.geeksforgeeks.org/bool-in-python/
        # Returns True as x is a non empty string
        Example:
        >>> x = 'GeeksforGeeks'
        >>> print(bool(x))
        True
        """
        # 这句话的意思是, 字符串不为空并且开头结尾是[]
        return bool(ch) and (ch[0] == "[") and (ch[-1] == "]")

    @staticmethod
    def _is_cjk_character(ch):
        """cjk类字符(包括中文字符也在此列)
        参考: https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        C: chinese, J: japanese, K: korean
        0x4E00 <= code <= 0x9FFF, CJK Unified Ideographs, https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        0x3400 <= code <= 0x4DBF, CJK Unified Ideographs Extension A, https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_Extension_A
        0x20000 <= code <= 0x2A6DF, CJK Unified Ideographs Extension B, https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_Extension_B
        0x2A700 <= code <= 0x2B73F, CJK Unified Ideographs Extension C, https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_Extension_C
        0x2B740 <= code <= 0x2B81F, CJK Unified Ideographs Extension D, https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_Extension_D
        0x2B820 <= code <= 0x2CEAF, CJK Unified Ideographs Extension E, https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_Extension_E
        0x2CEB0 <= code <= 0x2EBEF, CJK Unified Ideographs Extension F, https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_Extension_F
        """

        # The ord() function returns an integer representing the Unicode character.
        # by the way, the ord function is the inverse of chr()
        code = ord(ch)
        return 0x4E00 <= code <= 0x9FFF or \
               0x3400 <= code <= 0x4DBF or \
               0x20000 <= code <= 0x2A6DF or \
               0x2A700 <= code <= 0x2B73F or \
               0x2B740 <= code <= 0x2B81F or \
               0x2B820 <= code <= 0x2CEAF or \
               0x2CEB0 <= code <= 0x2EBEF

    @staticmethod
    def _is_punctuation(ch):
        """标点符号类判断，全、半角通用
        33-47: !"#$%&'()*+,-./
        58-64: :;<=>?@
        91-96: [\]^_`
        123-126: {|}~
        Unicode character property
            -> General Category
                -> Punctuation (P),
        https://en.wikipedia.org/wiki/Unicode_character_property
        """
        code = ord(ch)
        return 33 <= code <= 47 or \
               58 <= code <= 64 or \
               91 <= code <= 96 or \
               123 <= code <= 126 or \
               unicodedata.category(ch).startswith('P')

roberta_dir = "."
config_path = f"{roberta_dir}/bert_config.json"
ckpt_path = f"{roberta_dir}/bert_model.ckpt"
dict_path = f"{roberta_dir}/vocab.txt"

tokenizer = Tokenizer(dict_path, do_lower_case=True)
model = M.build_transformer_model(config_path=config_path, checkpoint_path=ckpt_path)

token_ids, segment_ids = tokenizer.encode('语言模型')

print('\n ===== predicting ===== \n')
print(model.predict([np.array([token_ids]), np.array([segment_ids])]))
