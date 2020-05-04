# import bert4keras
import bert4keras.models as M
import bert4keras.layers as L
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

class Tokenizer(object):
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

class Transformer(object):
    """模型基类
    """
    def __init__(self,
                 vocab_size, # 词表大小
                 hidden_size, # 编码维度
                 num_hidden_layers, # Transformer总层数
                 num_attention_heads, # Attention的头数
                 intermediate_size, # FeedForward的隐层维度
                 hidden_act, # FeedForward隐层的激活函数
                 dropout_rate, # Dropout比例
                 embedding_size=None, # 是否指定embedding_size, 如果不指定默认为hidden_size
                 keep_tokens=None, # 要保留的词id列表
                 layers=None, # 外部传入的keras层, dictionary, 不指定默认={}
                 name=None, # 模型名称
                 **kwargs
                 ):
        if keep_tokens is None:
            self.vocab_size = vocab_size
        else:
            # 不懂, 如果有要保留的词id列表, 那么词表大小就是保留词数量吗?
            self.vocab_size = len(keep_tokens)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_lyers = num_attention_heads
        # 每个注意力头的长度 = 隐层长度 整除 注意力头个数
        self.attention_head_size = hidden_size // num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.dropout_rate = dropout_rate
        self.embedding_size = embedding_size or hidden_size
        self.keep_tokens = keep_tokens
        self.attention_mask = None
        self.position_bias = None
        self.layers = {} if layers is None else layers
        self.name = name

    def set_inputs(self,
                   inputs,
                   additional_input_layers=None):
        """设置input和inputs属性
        """
        if inputs is None:
            inputs = []
        elif not isinstance(inputs, list):
            inputs = [inputs]

        inputs = inputs[:] # 没看懂
        if additional_input_layers is not None:
            if not isinstance(additional_input_layers, list):
                additional_input_layers = [additional_input_layers]
            inputs.extend(additional_input_layers)

        self.inputs = inputs
        # 如果inputs多于1个,就整个传给input
        # 如果inputs等于一个, input就取inputs[0] 唯一的一个
        if len(inputs) > 1:
            self.input = inputs
        else:
            self.input = inputs[0]

    def build(self,
              layer_norm_cond=None,
              layer_norm_cond_hidden_size=None,
              layer_norm_cond_hidden_act=None,
              additional_input_layers=None,
              **kwargs):
        """模型构建函数
        layer_norm_*系列参数为实现Conditional layer normalization时使用,
        用来实现以"固定长度向量"为条件的条件bert
        # 什么叫"条件Bert"? 这段描述看不懂
        """
        inputs = self.prepare_inputs()
        self.set_inputs(inputs, additional_input_layers)
        outputs = inputs

        # Other
        self.layer_norm_conds = [
            layer_norm_cond,
            layer_norm_cond_hidden_size,
            layer_norm_cond_hidden_act or 'linear',
        ]

        # Embedding
        # 这一步没看懂, 主要就是层之间的关系不懂
        outputs = self.prepare_embeddings(outputs)

        # Main
        for i in range (self.num_hidden_layers):
            outputs = self.prepare_main_layers(outputs, i)

        # Final
        outputs = self.prepare_final_layers(outputs)

    def prepare_inputs(self):
        raise NotImplementedError

    def prepare_embeddings(self, inputs):
        raise NotImplementedError

    def prepare_main_layers(self, inputs, index):
        raise NotImplementedError

    def prepare_final_layers(self, inputs):
        raise NotImplementedError



class BERT(M.Transformer):
    """构建bert模型
    """
    def __init__(self,
                 max_position, # 序列最大长度
                 with_pool=False, # 是否包含Pool部分
                 with_nsp=False, # 是否包含NSP部分
                 with_mlm=False, # 是否包含MLM部分
                 ** kwargs # 其余参数
                 ):
        super(BERT, self).__init__(**kwargs)
        # 这一步真的需要吗? 感觉没有用到MRO
        # 感觉还是需要的, 比如例子
        # 例子删了, 是csdn的, 太长了影响appearance
        # python的super()的作用和原理 https://blog.csdn.net/liweiblog/article/details/54425572
        # Supercharge Your Classes With Python super() https://realpython.com/python-super/
        self.max_position = max_position
        self.with_pool = with_pool
        self.with_nsp = with_nsp
        self.with_mlm = with_mlm

    def prepare_inputs(self):
        """bert的输入是token_ids和segment_ids
        """
        # 1. tensor of ids, 2. tensor of segment_ids
        # 这里的Input()就是keras.layers.Input()(我初步推断的)
        # Input(shape=None, batch_shape=None,
        #       name=None, dtype=None, sparse=False,
        #       tensor=None):
        # name: An optional name string for the layer.
        #     Should be unique in a model (do not reuse the same name twice).
        #     It will be autogenerated if it isn't provided.
        # shape: A shape tuple (integer), not including the batch size.
        #     For instance, `shape=(32,)` indicates that the expected input
        #     will be batches of 32-dimensional vectors.
        x_in = L.Input(shape=(None,), name='Input-Token')
        s_in = L.Input(shape=(None,), name='Input-Segment')
        return [x_in, s_in]

    def prepare_embeddings(self, inputs):
        """BERT的embedding是token, position, segment三者的embedding之和
        """
        x, s = inputs
        z = self.layer_norm_conds[0]

        # def call(self, inputs, layer=None, arguments=None, **kwargs):
        # 通过call调用层会自动重用同名层
        # inputs: 上一层的输出；
        # layer: 要调用的层类名；
        # arguments: 传递给layer.call的参数；
        # kwargs: 传递给层初始化的参数。

        # 这一层就是建立一个embedding层,
        # 这一层shape=(vocab_size, embedding_size)
        x = self.call(inputs=x,
                      layer=L.Embedding,
                      # self.vocab_size = vocab_size 词表大小
                      input_dim=self.vocab_size,
                      # self.embedding_size = embedding_size or hidden_size
                      output_dim=self.embedding_size,
                      embeddings_initializer=self.initializer,
                      mask_zero=True,
                      name='Embedding-Token')
        # 这一层就是建立一个embedding层
        # shape=(2, embedding_size) 我看不懂 真的想不出来
        # self.initializer 竟然没定义, 我更不懂是什么意思了
        s = self.call(inputs=s,
                      layer=L.Embedding,
                      input_dim=2,
                      output_dim=self.embedding_size,
                      embeddings_initializer=self.initializer,
                      name='Embedding-Segment')

        # 看不懂, segment想不出来, 别的也想不出来
        x = self.call(inputs=[x, s], layer=L.Add, name='Embedding-Token-Segment')

        # 这一层就是Position Embedding
        # 我知道high-level怎么做的,
        # 但是对现在的这些代码具体实现细节不懂
        x = self.call(inputs=x,
                      layer=L.PositionEmbedding,
                      input_dim=self.max_position,
                      output_dim=self.embedding_size,
                      merge_mode='add',
                      embedding_initializer=self.initializer,
                      mask_zero=True,
                      name='Embedding-Token')
        # (z is not None) 返回的结果是一个bool, 而不是一个bool的tuple
        # tuple: 类似list, 但是ordered and unchanged
        # self.layer_norm_conds = [
        #     layer_norm_cond,
        #     layer_norm_cond_hidden_size,
        #     layer_norm_cond_hidden_act or 'linear',
        # ]
        x = self.call(inputs=self.simplify([x, z]),
                      layer=L.LayerNormalization,
                      conditional=(z is not None),
                      hidden_unites=self.layer_norm_conds[1], #layer_norm_cond_hidden_size
                      hidden_actibation=self.layer_norm_conds[2], # layer_norm_cond_hidden_act or 'linear'
                      hidden_initializer=self.initializer,
                      name='Embedding-Norm')
        # 我还是不懂, 什么是layer, 什么是inputs 什么是outputs
        x = self.call(inputs=x,
                      layer=L.Dropout,
                      rate=self.dropout_rate,
                      name='Embedding-Dropout')
        if self.embedding_size != self.hidden_size:
            # 意思就是说 将embedding map到一个另外的空间
            # 说白了 就是用矩阵乘一下, 因为矩阵意味着变换
            # What exactly do we mean by embeddings “mapping words to a high-dimensional semantic space”?
            # https://towardsdatascience.com/why-do-we-use-embeddings-in-nlp-2f20e1b632d2
            #
            # Word embedding is the collective name for a set of
            # language modeling and feature learning techniques in
            # natural language processing (NLP) where words or phrases
            # from the vocabulary are mapped to vectors of real numbers.
            # Conceptually it involves a mathematical embedding from a space
            # with many dimensions per word to a continuous vector space
            # with a much lower dimension.
            # https://en.wikipedia.org/wiki/Word_embedding
            x = self.call(inputs=x,
                          layer=L.Dense,
                          units=self.hidden_size,
                          kernel_initializer=self.initializer,
                          name="Embedding-Mapping")

        return x

    def compute_attention_mask(self, inputs=None):
        return self.attention_mask

    def prepare_main_layers(self, inputs, index):
        """Bert的主体是基于Multi-Head Self Attention多头自注意力机制
        顺序: Att --> Add --> LN --> FFN --> Add --> LN
        名词解释:
        Att: Multi-Head Self Attention, 多头自注意力机制,
            "Attention Is All You Need" https://arxiv.org/abs/1706.03762
        Add: Att之前和之后的结果相加, ResNet的思路, 可以有效防止梯度消失
            "Deep Residual Learning for Image Recognition" https://arxiv.org/abs/1512.03385
        LN: Layer Normalization, 为了解决机器学习中的ICS问题,
            "Layer Normalization", https://arxiv.org/abs/1607.06450
            # 但是我不懂其中的原理. 没看懂
            # "Layer Normalization Explained" https://leimao.github.io/blog/Layer-Normalization/
        FFN: Feed Forward Neural Network/Multi-Layered Neural N(MLN), 前向神经网络, 最简单的神经网络
            "Simulation Neuronaler Netze", https://scholar.google.com/scholar?cluster=3666297338783020225
            "Feedforward Neural Networks Explained" https://hackernoon.com/deep-learning-feedforward-neural-networks-explained-c34ae3f084f1
        Add: 同上
        LN: 同上
        """
        x = inputs
        # self.layer_norm_conds = [
        #     layer_norm_cond,
        #     layer_norm_cond_hidden_size,
        #     layer_norm_cond_hidden_act or 'linear',
        # ]
        z = self.layer_norm_conds[0] # layer_norm_cond

        attention_name = 'Transformer-%d-MultiHeadSelfAttention' % index
        feed_forward_name = 'Tansformer-%d-FeedForward' % index
        # 好像在transformer模型里并没有实现过程,
        # 仅仅返回了self.attention_mask
        # 而 self.attention_mask = None

        # 按理说, 正常的attention_mask应该
        # 开始: 是0或1的mask, 然后被计算后: 变成0或-10000(adder)
        # 实现: attention_score = (1.0 - attention_mask) * -10000.0
        # https://github.com/huggingface/transformers/blob/master/src/transformers/modeling_tf_openai.py#L282
        # https://github.com/google-research/bert/blob/master/modeling.py#L709
        attention_mask = self.compute_attention_mask()

        # Multi-Head Self Attention
        x1, x, arguments = x, [x, x, x], {'a_mask': None}
        if attention_mask is not None:
            arguments['a_mask'] = True
            x.append(attention_mask)
        x = self.call(inputs=x,
                      layer=L.MultiHeadAttention,
                      arguments=arguments,
                      heads=self.num_attention_heads,
                      kernel_initializer=self.initializer,
                      name=attention_name) # attention_name = 'Transformer-%d-MultiHeadSelfAttention' % index
        x = self.call(inputs=x,
                      layer=L.Dropout,
                      rate=self.dropout_rate,
                      name='%s-Dropout' % attention_name)
        # Add
        x = self.call(inputs=[x1, x],
                      layer=L.Add,
                      name='%s-Add' % attention_name)
        # LN
        # z = self.layer_norm_conds[0] # layer_norm_cond
        x = self.call(inputs=self.simplify([x, z]),
                      layer=L.LayerNormalization,
                      conditional=(z is not None),
                      hidden_units=self.layer_norm_conds[1],
                      hidden_activation=self.layer_norm_conds[2],
                      hidden_initializer=self.initializer,
                      name='%s-Norm' % attention_name)

        # FFN
        xi = x
        x = self.call(inputs=x,
                      layer=L.FeedForward,
                      units=self.intermediate_size, # intermediate_size, # FeedForward的隐层维度
                      activation=self.hidden_act,
                      kernel_initializer=self.initializer,
                      name=feed_forward_name) # feed_forward_name = 'Tansformer-%d-FeedForward' % index
        x = self.call(inputs=x,
                      layer=L.Dropout,
                      rate=self.dropout_rate,
                      name='%s-Dropout' % feed_forward_name)
        # Add
        x = self.call(inputs=[x1, x],
                      layer=L.Add,
                      name='%s-Add' % feed_forward_name)
        # LN
        x = self.call(inputs=self.simplify([x, z]),
                      layer=L.LayerNormalization,
                      conditional=(z is not None),
                      hidden_units=self.layer_norm_conds[1],
                      hidden_activation=self.layer_norm_conds[2],
                      hidden_initializer=self.initializer,
                      name='%s-Norm' % feed_forward_name)

        return x

def prepare_final_layers(self, inputs):
    """根据剩余参数决定输出
    """




import json
def build_transformer_model(config_path=None,
                            checkpoint_path=None,
                            model='bert',
                            application='encoder',
                            return_keras_model=True,
                            **kwargs):
    """根据配置文件构建模型, 可选加载checkpoint权重
    "bert_config.json"
    {
      "attention_probs_dropout_prob": 0.1,
      "directionality": "bidi",
      "hidden_act": "gelu",
      "hidden_dropout_prob": 0.1,
      "hidden_size": 1024,
      "initializer_range": 0.02,
      "intermediate_size": 4096,
      "max_position_embeddings": 512,
      "num_attention_heads": 16,
      "num_hidden_layers": 24,
      "pooler_fc_size": 768,
      "pooler_num_attention_heads": 12,
      "pooler_num_fc_layers": 3,
      "pooler_size_per_head": 128,
      "pooler_type": "first_token_transform",
      "type_vocab_size": 2,
      "vocab_size": 21128
    }
    """
    config = kwargs
    if config_path is not None:
        config.update(json.load(open(config_path)))
    if 'max_position' not in config:
        # "max_position_embeddings": 512,
        config['max_position'] = config.get('max_position_embeddings')
    if 'drop_out_rate' not in config:
        # "hidden_dropout_prob": 0.1,
        config['dropout_rate'] = config.get('hidden_dropout_prob')

    model, application = model.lower(), application.lower()

    models = {
        # class BERT() def __init__():
        'bert': BERT,
        # 'albert': M.ALBERT,
        # 'albert_unshare': M.ALBERT_Unshared,
        # 'nezha': M.NEZHA,
        # 'gpt2_ml': M.GPT2_ML,
        # # 末尾也有一个逗号的原因: 改变时, 不改变原有行, 在源码检查, 管理时有用
        # # 有没有逗号的字典的行为完全相同
        # # https://stackoverflow.com/questions/29063868/does-the-extra-comma-at-the-end-of-a-dict-list-or-set-has-any-special-meaning
        # 't5': M.T5,
    }
    # python对变量是区分大小写的
    # Python 区分大小写吗？ https://blog.csdn.net/PoGeN1/article/details/82317956
    MODEL = models[model]

    # 我还没有用到lm任务和unilm(这是什么?), 不实现
    # if model != 'T5':
    #     # application的意思我还没搞懂
    #     # 但我猜是指语言模型的应用是什么
    #     # 比如'encoder'就是作为编码器, 输入sentence 输出编码
    #     if application == 'lm':
    #         MODEL = M.extend_with_language_model(MODEL)
    #     elif application == 'unilm':
    #         MODEL = M.extend_with_unified_language_model(MODEL)

    # 这里的kwargs的意思就是保持一致性
    # 因为GPT2_ML的__init__里有**kwargs,
    # Transformers的__init__里有**kwargs,
    # 但是BERT里的super()是不是可以去掉
    # 因为没有用到MRO
    transformer = MODEL(**kwargs)
    # 实现build函数
    # class BERT() def build()
    transformer.build(**kwargs)

    if checkpoint_path is not None:
        transformer.load_weights_from_checkpoint(checkpoint_path)

    # 这一步没看懂, 为什么return transformer model就返回了.model
    # 这是keras的机制吗?
    # 或者是... 变量transformer实例化的类是什么?
    if return_keras_model:
        return transformer.model
    else:
        return transformer



roberta_dir = "C:/JupyterWorkspace/sentiment-keras4bert/roberta"
config_path = f"{roberta_dir}/bert_config.json"
ckpt_path = f"{roberta_dir}/bert_model.ckpt"
dict_path = f"{roberta_dir}/vocab.txt"

tokenizer = Tokenizer(dict_path, do_lower_case=True)
model = M.build_transformer_model(config_path=config_path, checkpoint_path=ckpt_path)

token_ids, segment_ids = tokenizer.encode('语言模型')

print('\n ===== predicting ===== \n')
print(model.predict([np.array([token_ids]), np.array([segment_ids])]))
