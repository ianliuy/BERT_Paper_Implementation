# import bert4keras
import bert4keras.tokenizers as T
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

    if simplified: # 过滤冗余部分token
        new_token_dict, keep_tokens = {}, []
        # 这里是什么意思 没看懂
        # 如果有startwith 那就继续 没有 那就是空[]??
        startwith = startwith or []
        for t in startwith:
            new_token_dict[t] = len(new_token_dict)
            # 这一步是什么意思? 不应该是new_token_dict吗?
            # 或者说 keep_tokens是干什么的list?
            # keep_token就是为了去掉cjk和标点，那startwith又是干什么的?
            keep_tokens.append(token_dict[t])

        # Return a new list containing all items from the iterable in ascending order.
        # A custom key function can be supplied to customize the sort order, and the
        # reverse flag can be set to request the result in descending order.

        # Dictionary in Python is an unordered collection of data values

        # dict.item()返回一个dict_items对象，是一个list，里面有很多key value的pair
        # 这里的lambda s: s[1] 就是把id(value)取出
        # t就是排序id(value)后的token(key)
        for t, _ in sorted(token_dict.items(), key=lambda s: s[1]):
            if t not in new_token_dict:
                keep = True
                if len(t) > 1:
                    for c in (t[2:] if t[:2] == '##' else t):
                        if (Tokenizer._is_cjk_character(c) or Tokenizer._is_punctuation(c)):
                            keep = False
                            break
                # 经过上面的判断条件，keep下来的符号有
                # [CLS]等特殊token、英文token
                if keep:
                    new_token_dict[t] = len(new_token_dict)
                    keep_tokens.append(token_dict[t])

        return new_token_dict, keep_tokens
    else:
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

    def token_to_id(self, token):
        """token转换为相对应的id序列
        """
        return NotImplementedError

    def tokens_to_ids(self, tokens):
        """tokens列表变为ids列表
        """
        return [self.token_to_id(token) for token in tokens]

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


class Tokenizer(BasicTokenizer):
    """实现自苏建林的Bert4keras
    https://github.com/bojone/bert4keras
    """

    def __init__(self, token_dict, *args, **kwargs):
        """初始化
        """
        super().__init__(*args, **kwargs)
        if isinstance(token_dict, str):
            token_dict = T.load_vocab(token_dict)  # 返回一个字典d, d[token] = id

        self._token_dict = token_dict
        self._token_dict_inv = {v: k for k, v in token_dict.items()}  # inverse的字典,原来是key: value, 变成value: key
        self._vocab_size = len(token_dict)

        for token in ['pad', 'unk', 'mask', 'start', 'end']:
            try:
                # getattr(object, name[, default]) -> value, get attribute
                # getattr(x, 'y') is equivalent to x.y
                _token_id = token_dict[getattr(self, '_token_%s' % token)]
                # setattr(x, 'y', v) is equivalent to 'x.y = v', set attribute
                setattr(self, '_token_%s_id' % token, _token_id)
                # 这一步我没看懂,
                # 我猜是如果某人重写了这个__init__方法, 或者未来作者制定了_token_pad的id
                # 然后根据上面inverse的逻辑, 也重新token和id换一下位置
                # inverse又为了什么?
            except:
                pass

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

    def token_to_id(self, token):
        """BasicTokenizer里没实现
        在Tokenizer类里实现
        # the reason why using dict.get(key) instead of dict[key]
        # is that it allows me to provide a default value if the key is missing
        # https://stackoverflow.com/questions/11041405/why-dict-getkey-instead-of-dictkey
        """
        return self._token_dict.get(token, self._token_unk_id)

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

    def id_to_token(self, id):
        """id 转换为相应的token
        """
        return self._token_dict_inv[id]

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

    @staticmethod
    def _cjk_punctuation():
        # ＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠
        # ［＼］＾＿｀｛｜｝～｟｠｢｣､　、〃〈〉
        # 《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞
        # 〟〰〾〿–—‘’‛“”„‟…‧﹏﹑﹔\xb7！？｡。
        # \xb7是什么我没搞懂
        # \xb7是"middle dot"的Python Escape, \u00b7是"middle dot"的Java Escape, 两者打印出的结果是一样的
        # https://charbase.com/00B7
        return u'\uff02\uff03\uff04\uff05\uff06\uff07\uff08\uff09\uff0a\uff0b\uff0c\uff0d\uff0f\uff1a\uff1b\uff1c\uff1d\uff1e\uff20\uff3b\uff3c\uff3d\uff3e\uff3f\uff40\uff5b\uff5c\uff5d\uff5e\uff5f\uff60\uff62\uff63\uff64\u3000\u3001\u3003\u3008\u3009\u300a\u300b\u300c\u300d\u300e\u300f\u3010\u3011\u3014\u3015\u3016\u3017\u3018\u3019\u301a\u301b\u301c\u301d\u301e\u301f\u3030\u303e\u303f\u2013\u2014\u2018\u2019\u201b\u201c\u201d\u201e\u201f\u2026\u2027\ufe4f\ufe51\ufe54\u00b7\uff01\uff1f\uff61\u3002'

    def ids_to_token(self, ids):
        return [self.id_to_token(id) for id in ids]

    def decode(self, ids, tokens=None):
        """ids是一个iterable object, 返回可读文本
        """
        tokens = tokens or self.ids_to_token(ids)
        # 去掉[CLS] [SEP]等特殊的token
        tokens = [token for token in tokens if not self._is_special(token)]

        text, flag = '', False
        for i, token in enumerate(tokens):
            if token[:2] == "##":
                text += token[2:]  # 如果token类似于"##阿" text后接"阿"
            elif len(token) == 1 and self._is_cjk_character(token):
                text += token  # 如果token是中文单字, 直接加入text
            elif len(token) == 1 and self._is_punctuation(token):
                text += token
                text += ' '
            # 如果不是中英文的"##阿"、"##aa"，也不是中文单子，也不是标点符号，
            # 还有可能是数字(??有可能不是，因为有[num]token)
            # 还有可能是英文单词 a the 之类(也有可能不是a the，停用词去掉了，反正就是某个英文单词)
            # 末尾是标点符号就直接加，不是标点符号就空一格再加
            elif i > 0 and self._is_cjk_character(text[-1]):
                text += token
            else:
                text += ' '
                text += token
        # 使用空格' '替换所有' +'??? 没看懂
        # Python re.sub Examples
        # https://lzone.de/examples/Python%20re.sub
        text = re.sub(' +', ' ', text)
        # 被替换的字符串: '(re|m|s|t|ve|d|ll)
        # 要替换的字符串: '\1
        text = re.sub('\'(re|m|s|t|ve|d|ll)', '\'\\1', text)
        punctuation = self._cjk_punctuation() + '+-/={(<['
        # Escape all the characters in pattern except ASCII letters, numbers and '_'.
        # 每个汉字前面都添加'\'
        punctuation_regex = '|'.join(re.escape(p) for p in punctuation)
        # 前后加括号
        punctuation_regex = '(%s)' % punctuation_regex
        # 应该是re sub里的引用, 但是我没太看懂
        # 是引用自己吗?
        text = re.sub(punctuation_regex, '\\1', text)
        # 这句我猜是删除数字之间的空格? \1引用第一个数字 \2引用第二个数字
        text = re.sub('(\d\.) (\d)', '\\1\\2', text)
        return text.strip()


def _cjk_punctuation():
    # ＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠
    # ［＼］＾＿｀｛｜｝～｟｠｢｣､　、〃〈〉
    # 《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞
    # 〟〰〾〿–—‘’‛“”„‟…‧﹏﹑﹔\xb7！？｡。
    # \xb7是什么我没搞懂
    # \xb7是"middle dot"的Python Escape, \u00b7是"middle dot"的Java Escape
    # https://charbase.com/00B7
    return u'\uff02\uff03\uff04\uff05\uff06\uff07\uff08\uff09\uff0a\uff0b\uff0c\uff0d\uff0f\uff1a\uff1b\uff1c\uff1d\uff1e\uff20\uff3b\uff3c\uff3d\uff3e\uff3f\uff40\uff5b\uff5c\uff5d\uff5e\uff5f\uff60\uff62\uff63\uff64\u3000\u3001\u3003\u3008\u3009\u300a\u300b\u300c\u300d\u300e\u300f\u3010\u3011\u3014\u3015\u3016\u3017\u3018\u3019\u301a\u301b\u301c\u301d\u301e\u301f\u3030\u303e\u303f\u2013\u2014\u2018\u2019\u201b\u201c\u201d\u201e\u201f\u2026\u2027\ufe4f\ufe51\ufe54\xb7\uff01\uff1f\uff61\u3002'

roberta_dir = "."
config_path = f"{roberta_dir}/bert_config.json"
ckpt_path = f"{roberta_dir}/bert_model.ckpt"
dict_path = f"{roberta_dir}/vocab.txt"

tokenizer = Tokenizer(dict_path, do_lower_case=True)
model = M.build_transformer_model(config_path=config_path, checkpoint_path=ckpt_path)

token_ids, segment_ids = tokenizer.encode('语言模型')

print('\n ===== predicting ===== \n')
print(model.predict([np.array([token_ids]), np.array([segment_ids])]))
