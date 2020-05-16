# import bert4keras
import keras
import tensorflow as tf
from keras.models import Model
import numpy as np
import unicodedata
import bert4keras.backend as B


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
        self._token_mask = '[MASK]'
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

    def tokens_to_ids(self, tokens):
        """tokens列表变为ids列表
        """
        return [self._token_dict.get(token, self._token_unk_id) for token in tokens]

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

    def encode(self,
               first_text,
               second_text=None,
               max_length=None,
               first_length=None,
               second_length=None):
        """输出文本对应的token id和segment id
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

    def ids_to_tokens(self, ids):
        """类似id_to_token, 但是ids是iterable
        """
        return list(self.id_to_token(id) for id in ids)

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
        0xF900 <= code <= 0xFADF, 兼容汉字
        0x2F800 <= code <= 0x2FA1F, 兼容扩展
        reference: https://www.cnblogs.com/straybirds/p/6392306.html
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
               0x2CEB0 <= code <= 0x2EBEF or \
               0xF900 <= code <= 0xFADF or \
               0x2F800 <= code <= 0x2FA1F


    @staticmethod
    def _is_control(ch):
        """控制类字符判断
        Cc: Other, control
        Cf: Other, format
        """
        return unicodedata.category(ch) in ('Cc', 'Cf')

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
        return bool(ch) and (ch[0] == '[') and (ch[-1] == ']')

if keras.__version__[-2:] != 'tf' and keras.__version__ < '2.3':

    class Layer(keras.layers.Layer):
        """重新定义Layer，赋予“层中层”功能
        （仅keras 2.3以下版本需要）
        """
        def __init__(self, **kwargs):
            super(Layer, self).__init__(**kwargs)
            self.supports_masking = True  # 本项目的自定义层均可mask

        def __setattr__(self, name, value):
            if isinstance(value, keras.layers.Layer):
                if not hasattr(self, '_layers'):
                    self._layers = []
                if value not in self._layers:
                    self._layers.append(value)
            super(Layer, self).__setattr__(name, value)

        @property
        def trainable_weights(self):
            trainable = getattr(self, 'trainable', True)
            if trainable:
                trainable_weights = super(Layer, self).trainable_weights[:]
                for l in getattr(self, '_layers', []):
                    trainable_weights += l.trainable_weights
                return trainable_weights
            else:
                return []

        @property
        def non_trainable_weights(self):
            trainable = getattr(self, 'trainable', True)
            non_trainable_weights = super(Layer, self).non_trainable_weights[:]
            for l in getattr(self, '_layers', []):
                if trainable:
                    non_trainable_weights += l.non_trainable_weights
                else:
                    non_trainable_weights += l.weights
            return non_trainable_weights

else:

    class Layer(keras.layers.Layer):
        def __init__(self, **kwargs):
            super(Layer, self).__init__(**kwargs)
            self.supports_masking = True  # 本项目的自定义层均可mask

class PositionEmbedding(Layer):
    """定义位置Embedding，这里的Embedding是可训练的
    """
    def __init__(self,
                 input_dim,
                 output_dim,
                 merge_mode='add',
                 embeddings_initializer='zeros',
                 **kwargs):
        super(PositionEmbedding, self).__init__(**kwargs)
        # input_dim=self.max_position, = max_position_embeddings = 512
        # output_dim=self.embedding_size, = embedding_size or hidden_size = 1024
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.merge_mode = merge_mode
        # 这里的get是什么意思，我不懂
        # 大意就是用全0去初始化位置
        # 原版中是用的截断初始化正态分布
        # 产生一个自己定义mean和standard deviation的normal distribution
        # 然后产生随机数
        # https://github.com/google-research/bert/blob/master/modeling.py#L495
        self.embeddings_initializer = keras.initializers.get(embeddings_initializer)

    def build(self, input_shape):
        super(PositionEmbedding, self).build(input_shape)
        # 这里添加的
        self.embeddings = self.add_weight(
            name='embeddings',
            shape=(self.input_dim, self.output_dim),
            initializer=self.embeddings_initializer,
        )

    def call(self, inputs, **kwargs):
        input_shape = keras.backend.shape(inputs)
        batch_size, seq_len = input_shape[0], input_shape[1]
        pos_embeddings = self.embeddings[:seq_len]
        # Adds a 1-sized dimension at index "axis
        pos_embeddings = keras.backend.expand_dims(pos_embeddings, 0)

        if self.merge_mode == 'add':
            return inputs + pos_embeddings
        else:
            # Creates a tensor by tiling `x` by `n`
            pos_embeddings = keras.backend.tile(pos_embeddings, [batch_size, 1, 1])
            return keras.backend.concatenate([inputs, pos_embeddings])

    def compute_output_shape(self, input_shape):
        if self.merge_mode == 'add':
            return input_shape
        else:
            return input_shape[:2] + (input_shape[2] + self.output_dim, )

    def get_config(self):
        config = {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'merge_mode': self.merge_mode,
            'embeddings_initializer': keras.initializers.serialize(self.embeddings_initializer),
        }
        base_config = super(PositionEmbedding, self).get_config()
        # 这是什么意思？什么叫items(), list(dict.items())之后又是什么？
        return dict(list(base_config.items()) + list(config.items()))

class LayerNormalization(Layer):
    """(conditional) Layer Normalization
    hidden_* 系列参数为有条件输入时(conditional=True)使用
    """
    def __init__(self,
                 center=True,
                 scale=True,
                 epsilon=None,
                 conditional=False,
                 hidden_units=None,
                 hidden_activation='linear',
                 hidden_initializer='glorot_uniform',
                 **kwargs):
        super(LayerNormalization, self).__init__(**kwargs)
        self.center = center
        self.scale = scale
        self.conditional = conditional
        self.hidden_units = hidden_units
        self.hidden_activation = keras.activations.get(hidden_activation)
        self.hidden_initializer = keras.initializers.get(hidden_initializer)
        self.epsilon = epsilon or 1e-12

    def build(self, input_shape):
        super(LayerNormalization, self).build(input_shape)
        if self.conditional:
            # 因为如果使用conditional，inputs = [inputs, condition]
            # 所以shape
            shape = (input_shape[0][-1], )
        else:
            shape = (input_shape[-1], )
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        initializer='zeros',
                                        name='beta')
        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         initializer='ones',
                                         name='gamma')
        if self.conditional:
            if self.hidden_units is not None:
                self.hidden_dense = keras.layers.Dense(
                    units=self.hidden_units,
                    activation=self.hidden_activation,
                    use_bias=False,
                    kernel_initializer=self.hidden_initializer)
            # beta是加上的东西，决定中心
            # gamma是乘上的东西，决定缩放大小
            if self.center:
                self.beta_dense = keras.layers.Dense(
                    units=shape[0],
                    use_bias=False,
                    kernel_initializer='zeros')
            if self.scale:
                self.gamma_dense = keras.layers.Dense(
                    units=shape[0],
                    use_bias=False,
                    kernel_initializer='zeros')

    def call(self, inputs, **kwargs):
        """如果是条件Layer Norm，则默认以list为输入，第二个是condition
        """
        if self.conditional:
            inputs, cond = inputs
            if self.hidden_units is not None:
                cond = self.hidden_dense(cond)
            for _ in range(keras.backend.ndim(inputs) - keras.backend.ndim(cond)):
                cond = keras.backend.expand_dims(cond, 1)
            if self.center:
                beta = self.beta_dense(cond) + self.beta
            if self.scale:
                gamma = self.gamma_dense(cond) + self.gamma
        else:
            if self.center:
                beta = self.beta
            if self.scale:
                gamma = self.gamma
        outputs = inputs
        if self.center:
            mean = keras.backend.mean(outputs, axis=-1, keepdims=True)
            outputs = outputs - mean
        if self.scale:
            variance = keras.backend.mean(keras.backend.square(outputs), axis=-1, keepdims=True)
            std = keras.backend.sqrt(variance + self.epsilon)
            outputs = outputs / std
            outputs = outputs * gamma
        if self.center:
            outputs = outputs + beta
        return outputs

    def get_config(self):
        config = {
            'center': self.center,
            'scale': self.scale,
            'epsilon': self.epsilon,
            'conditional': self.conditional,
            'hidden_units': self.hidden_units,
            'hidden_activation': keras.activations.serialize(self.hidden_activation),
            'hidden_initializer': keras.initializers.serialize(self.hidden_initializer),}
        base_config = super(LayerNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class MultiHeadAttention(Layer):
    """多头注意力机制
    """
    def __init__(self,
                 heads,
                 head_size,
                 key_size=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.heads = heads
        self.head_size = head_size
        self.out_dim = heads * head_size
        self.key_size = key_size or head_size
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)

    def build(self, input_shape):
        super(MultiHeadAttention, self).build(input_shape)
        self.q_dense = keras.layers.Dense(units=self.key_size * self.heads,
                             use_bias=self.use_bias,
                             kernel_initializer=self.kernel_initializer)
        self.k_dense = keras.layers.Dense(units=self.key_size * self.heads,
                             use_bias=self.use_bias,
                             kernel_initializer=self.kernel_initializer)
        self.v_dense = keras.layers.Dense(units=self.out_dim,
                             use_bias=self.use_bias,
                             kernel_initializer=self.kernel_initializer)
        self.o_dense = keras.layers.Dense(units=self.out_dim,
                             use_bias=self.use_bias,
                             kernel_initializer=self.kernel_initializer)

    def call(self, inputs, mask=None, a_mask=None, p_bias=None):
        """实现多头注意力机制
        q_mask: 对输入的query序列进行mask。
                主要是将输出结果的padding部分置0
        v_mask: 对输入的value序列的mask。
                主要是防止attention读取到padding信息
        a_mask: 对attention矩阵的mask
                不同的attention mask对应不同的应用
        p_bias: 在attention里的位置偏置
                一般用来指定相对位置编码的种类
        """
        q, k, v = inputs[:3]
        q_mask, v_mask, n = None, None, 3
        if mask is not None:
            if mask[0] is not None:
                q_mask = keras.backend.cast(mask[0], keras.backend.floatx())
            if mask[2] is not None:
                v_mask = keras.backend.cast(mask[2], keras.backend.floatx())
        if a_mask:
            a_mask = inputs[n]
            n += 1
        # 线性变换
        qw = self.q_dense(q)
        kw = self.k_dense(k)
        vw = self.v_dense(v)
        # 形状变换
        qw = keras.backend.reshape(qw, (-1, keras.backend.shape(q)[1], self.heads, self.key_size))
        kw = keras.backend.reshape(kw, (-1, keras.backend.shape(k)[1], self.heads, self.key_size))
        vw = keras.backend.reshape(vw, (-1, keras.backend.shape(v)[1], self.heads, self.head_size))
        # Attention
        a = tf.einsum('bjhd,bkhd->bhjk', qw, kw)
        a = a / self.key_size**0.5
        a = B.sequence_masking(a, v_mask, 1, -1)
        if a_mask is not None:
            a = a - (1 - a_mask) * 1e12
        a = keras.backend.softmax(a)
        # 完成输出
        o = tf.einsum('bhjk,bkhd->bjhd', a, vw)
        o = keras.backend.reshape(o, (-1, keras.backend.shape(o)[1], self.out_dim))
        o = self.o_dense(o)
        # 返回结果
        o = B.sequence_masking(o, q_mask, 0)
        return o
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.out_dim)
    def compute_mask(self, inputs, mask=None):
        return mask[0]
    def get_config(self):
        config = {
            'heads': self.heads,
            'head_size': self.head_size,
            'key_size': self.key_size,
            'use_bias': self.use_bias,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
        }
        base_config = super(MultiHeadAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class FeedForward(Layer):
    """FeedForwrd层，其实就是两个Dense层的叠加
    """
    def __init__(self,
                 units,
                 activation='relu',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 **kwargs):
        super(FeedForward, self).__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
    def build(self, input_shape):
        super(FeedForward, self).build(input_shape)
        output_dim = input_shape[-1]
        if not isinstance(output_dim, int):
            output_dim = output_dim.value
        self.dense_1 = keras.layers.Dense(units=self.units,
                                          activation=self.activation,
                                          use_bias=self.use_bias,
                                          kernel_initializer=self.kernel_initializer)
        self.dense_2 = keras.layers.Dense(units=output_dim,
                                          use_bias=self.use_bias,
                                          kernel_initializer=self.kernel_initializer)
    def call(self, inputs, **kwargs):
        x = inputs
        x = self.dense_1(x)
        x = self.dense_2(x)
        return x
    def get_config(self):
        config = {
            'units': self.units,
            'activation': keras.activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
        }
        base_config = super(FeedForward, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class EmbeddingDense(Layer):
    """运算与Dense一致，但是kernel用的Embedding层的embeddings矩阵。
    根据Embedding层的名字搜索定位Embedding层
    """
    def __init__(self,
                 embedding_name,
                 activation='softmax',
                 use_bias=True,
                 **kwargs):
        super(EmbeddingDense, self).__init__(**kwargs)
        self.embedding_name = embedding_name
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias
    def call(self, inputs, **kwargs):
        if not hasattr(self, 'kernel'):
            embedding_layer = B.search_layer(inputs, self.embedding_name)
            if embedding_layer is None:
                raise Exception('Embedding layer not found')

            self.kernel = keras.backend.transpose(embedding_layer.embeddings)
            self.units = keras.backend.int_shape(self.kernel)[1]
            if self.use_bias:
                self.bias = self.add_weight(name='bias',
                                            shape=(self.units, ),
                                            initializer='zeros')

        outputs = keras.backend.dot(inputs, self.kernel)
        if self.use_bias:
            outputs = keras.backend.bias_add(outputs, self.bias)
        outputs = self.activation(outputs)
        return outputs
    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.units, )
    def get_config(self):
        config = {
            'embedding_name': self.embedding_name,
            'activation': keras.activations.serialize(self.activation),
            'use_bias': self.use_bias,
        }
        base_config = super(EmbeddingDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class BERT(object):
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
                 max_position, # 最大句子长度
                 embedding_size=None, # 是否指定embedding_size, 如果不指定默认为hidden_size
                 keep_tokens=None, # 要保留的词id列表
                 layers=None, # 外部传入的keras层, dictionary, 不指定默认={}
                 name=None, # 模型名称
                 with_pool=False,  # 是否包含Pool部分
                 with_nsp=False,  # 是否包含NSP部分
                 with_mlm=False,  # 是否包含MLM部分
                 **kwargs
                 ):
        if keep_tokens is None:
            self.vocab_size = vocab_size
        else:
            # 不懂, 如果有要保留的词id列表, 那么词表大小就是保留词数量吗?
            self.vocab_size = len(keep_tokens)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        # 每个注意力头的长度 = 隐层长度 整除 注意力头个数
        self.attention_head_size = hidden_size // num_attention_heads
        self.intermediate_size = intermediate_size
        self.dropout_rate = dropout_rate
        self.hidden_act = hidden_act
        self.embedding_size = embedding_size or hidden_size
        self.keep_tokens = keep_tokens
        self.attention_mask = None
        self.position_bias = None
        self.layers = {} if layers is None else layers
        self.name = name
        self.max_position = max_position
        self.with_pool = with_pool
        self.with_nsp = with_nsp
        self.with_mlm = with_mlm

    def build(self,
              layer_norm_cond=None,
              layer_norm_cond_hidden_size=None,
              layer_norm_cond_hidden_act=None,
              additional_input_layers=None,
              **kwargs):
        """模型构建函数
        layer_norm_*系列参数为实现Conditional layer normalization时使用,
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
        for i in range(self.num_hidden_layers):
            outputs = self.prepare_main_layers(outputs, i)

        # Final
        outputs = self.prepare_final_layers(outputs)
        self.set_outputs(outputs)

        # Model
        self.model = Model(self.inputs, self.outputs, name=self.name)

    def call(self, inputs, layer=None, arguments=None, **kwargs):
        """通过call调用层会自动重命名层
        inputs: 上一层的输出；
        layer: 要调用的层类名
        arguments: 传递给layer.call的参数
        kwargs: 传递给层初始化的参数
        """
        if layer is keras.layers.Dropout and self.dropout_rate == 0:
            # 如果dropout_rate属性为0
            # 就原地TP
            return inputs

        arguments = arguments or {}
        name = kwargs.get('name')
        if name not in self.layers:
            layer = layer(**kwargs)
            name = layer.name
            self.layers[name] = layer
        return self.layers[name](inputs, **arguments)

    def prepare_inputs(self):
        x_in = keras.layers.Input(shape=(None, ), name='Input-Token')
        s_in = keras.layers.Input(shape=(None, ), name='Input-Segment')
        return [x_in, s_in]

    def prepare_embeddings(self, inputs):
        """BERT的embedding是token, position, segment三者的embedding之和
        """
        x, s = inputs
        z = self.layer_norm_conds[0]
        x = self.call(inputs=x,
                      layer=keras.layers.Embedding,
                      input_dim=self.vocab_size,
                      output_dim=self.embedding_size,
                      embeddings_initializer=self.initializer,
                      mask_zero=True,
                      name='Embedding-Token')
        s = self.call(inputs=s,
                      layer=keras.layers.Embedding,
                      input_dim=2,
                      output_dim=self.embedding_size,
                      embeddings_initializer=self.initializer,
                      name='Embedding-Segment')
        x = self.call(inputs=[x, s], layer=keras.layers.Add, name='Embedding-Token-Segment')
        x = self.call(inputs=x,
                      layer=PositionEmbedding,
                      input_dim=self.max_position,
                      output_dim=self.embedding_size,
                      merge_mode='add',
                      embeddings_initializer=self.initializer,
                      name='Embedding-Position')
        x = self.call(inputs=self.simplify([x, z]),
                      layer=LayerNormalization,
                      conditional=(z is not None),
                      hidden_units=self.layer_norm_conds[1], #layer_norm_cond_hidden_size
                      hidden_activation=self.layer_norm_conds[2], # layer_norm_cond_hidden_act or 'linear'
                      hidden_initializer=self.initializer,
                      name='Embedding-Norm')
        x = self.call(inputs=x,
                      layer=keras.layers.Dropout,
                      rate=self.dropout_rate,
                      name='Embedding-Dropout')
        if self.embedding_size != self.hidden_size:
            x = self.call(inputs=x,
                          layer=keras.layers.Dense,
                          units=self.hidden_size,
                          kernel_initializer=self.initializer,
                          name='Embedding-Mapping')

        return x

    def prepare_main_layers(self, inputs, index):
        """Bert的主体是基于Multi-Head Self Attention多头自注意力机制
        顺序: Att --> Add --> LN --> FFN --> Add --> LN
        """
        x = inputs
        z = self.layer_norm_conds[0] # layer_norm_cond
        attention_name = 'Transformer-%d-MultiHeadSelfAttention' % index
        feed_forward_name = 'Transformer-%d-FeedForward' % index
        attention_mask = self.compute_attention_mask()

        # Multi-Head Self Attention
        xi, x, arguments = x, [x, x, x], {'a_mask': None}
        if attention_mask is not None:
            arguments['a_mask'] = True
            x.append(attention_mask)
        x = self.call(inputs=x,
                      layer=MultiHeadAttention,
                      arguments=arguments,
                      heads=self.num_attention_heads,
                      head_size=self.attention_head_size,
                      kernel_initializer=self.initializer,
                      name=attention_name) # attention_name = 'Transformer-%d-MultiHeadSelfAttention' % index
        x = self.call(inputs=x,
                      layer=keras.layers.Dropout,
                      rate=self.dropout_rate,
                      name='%s-Dropout' % attention_name)
        # Add
        x = self.call(inputs=[xi, x],
                      layer=keras.layers.Add,
                      name='%s-Add' % attention_name)
        # LN
        # z = self.layer_norm_conds[0] # layer_norm_cond
        x = self.call(inputs=self.simplify([x, z]),
                      layer=LayerNormalization,
                      conditional=(z is not None),
                      hidden_units=self.layer_norm_conds[1],
                      hidden_activation=self.layer_norm_conds[2],
                      hidden_initializer=self.initializer,
                      name='%s-Norm' % attention_name)

        # FFN
        xi = x
        x = self.call(inputs=x,
                      layer=FeedForward,
                      units=self.intermediate_size, # intermediate_size, # FeedForward的隐层维度
                      activation=self.hidden_act,
                      kernel_initializer=self.initializer,
                      name=feed_forward_name) # feed_forward_name = 'Tansformer-%d-FeedForward' % index
        x = self.call(inputs=x,
                      layer=keras.layers.Dropout,
                      rate=self.dropout_rate,
                      name='%s-Dropout' % feed_forward_name)
        # Add
        x = self.call(inputs=[xi, x],
                      layer=keras.layers.Add,
                      name='%s-Add' % feed_forward_name)
        # LN
        x = self.call(inputs=self.simplify([x, z]),
                      layer=LayerNormalization,
                      conditional=(z is not None),
                      hidden_units=self.layer_norm_conds[1],
                      hidden_activation=self.layer_norm_conds[2],
                      hidden_initializer=self.initializer,
                      name='%s-Norm' % feed_forward_name)

        return x

    def prepare_final_layers(self, inputs):
        """根据剩余参数决定输出
        """
        x = inputs
        z = self.layer_norm_conds[0]
        outputs = [x]

        if self.with_pool or self.with_nsp:
            x = outputs[0]
            x = self.call(inputs=x,
                          layer=keras.layers.Lambda,
                          function=lambda x: x[:, 0],
                          name='Pooler')
            pool_activation = 'tanh' if self.with_pool is True else self.with_pool

            x = self.call(inputs=x,
                          layer=keras.layers.Dense,
                          units=self.hidden_size,
                          activation=pool_activation,
                          kernel_initializer=self.initializer,
                          name='Pooler-Dense')
            if self.with_nsp:
                # Next Sentence Prediction
                x = self.call(inputs=x,
                              layer=keras.layers.Dense,
                              units=2,
                              activation='softmax',
                              kernel_initializer=self.initializer,
                              name='NSP-Proba')
            outputs.append(x)

        if self.with_mlm:
            # Masked Language Model部分
            x = outputs[0]
            x = self.call(inputs=x,
                          layer=keras.layers.Dense,
                          units=self.embedding_size,
                          activation=self.hidden_act,
                          kernel_initializer=self.initializer,
                          name='MLM-Dense')
            x = self.call(inputs=self.simplify([x, z]),
                          layer=LayerNormalization,
                          conditional=(z is not None),
                          hidden_units=self.layer_norm_conds[1],
                          hidden_activation=self.layer_norm_conds[2],
                          hidden_initializer=self.initializer,
                          name='MLM-Norm')
            mlm_activation = 'softmax' if self.with_mlm is True else self.with_mlm
            x = self.call(inputs=x,
                          layer=EmbeddingDense,
                          embedding_name='Embedding-Token',
                          activation=mlm_activation,
                          name='MLM-Proba')
            outputs.append(x)
        if len(outputs) == 1:
            outputs = outputs[0]
        elif len(outputs) == 2:
            outputs = outputs[1]
        else:
            outputs = outputs[1:]

        return outputs

    def compute_attention_mask(self, inputs=None):
        """定义每一层的Attention Mask
        """
        return self.attention_mask

    def set_inputs(self, inputs, additional_input_layers=None):
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
        if len(inputs) > 1:
            self.input = inputs
        else:
            self.input = inputs[0]

    def set_outputs(self, outputs):
        """设置output和outputs属性
        """
        if not isinstance(outputs, list):
            outputs = [outputs]
        outputs = outputs[:]
        self.outputs = outputs
        if len(outputs) > 1:
            self.output = outputs
        else:
            self.output = outputs[0]

    @property
    def initializer(self):
        """默认使用截断正态分布初始化
        """
        return keras.initializers.TruncatedNormal(stddev=0.02)

    def simplify(self, inputs):
        """将list中的None过滤掉
        """
        inputs = [i for i in inputs if i is not None]
        if len(inputs) == 1:
            inputs = inputs[0]
        return inputs

    def load_variable(self, checkpoint, name):
        """加载单个变量的函数
        """
        return tf.train.load_variable(checkpoint, name)

    def variable_mapping(self):
        """映射到官方BERT权重格式
        """
        mapping = {
            'Embedding-Token': ['bert/embeddings/word_embeddings'],
            'Embedding-Segment': ['bert/embeddings/token_type_embeddings'],
            'Embedding-Position': ['bert/embeddings/position_embeddings'],
            'Embedding-Norm': [
                'bert/embeddings/LayerNorm/beta',
                'bert/embeddings/LayerNorm/gamma',
            ],
            'Embedding-Mapping': [
                'bert/encoder/embedding_hidden_mapping_in/kernel',
                'bert/encoder/embedding_hidden_mapping_in/bias',
            ],
            'Pooler-Dense': [
                'bert/pooler/dense/kernel',
                'bert/pooler/dense/bias',
            ],
            'NSP-Proba': [
                'cls/seq_relationship/output_weights',
                'cls/seq_relationship/output_bias',
            ],
            'MLM-Dense': [
                'cls/predictions/transform/dense/kernel',
                'cls/predictions/transform/dense/bias',
            ],
            'MLM-Norm': [
                'cls/predictions/transform/LayerNorm/beta',
                'cls/predictions/transform/LayerNorm/gamma',
            ],
            'MLM-Proba': ['cls/predictions/output_bias'],
        }

        for i in range(self.num_hidden_layers):
            prefix = 'bert/encoder/layer_%d/' % i
            mapping.update({
                'Transformer-%d-MultiHeadSelfAttention' % i: [
                    prefix + 'attention/self/query/kernel',
                    prefix + 'attention/self/query/bias',
                    prefix + 'attention/self/key/kernel',
                    prefix + 'attention/self/key/bias',
                    prefix + 'attention/self/value/kernel',
                    prefix + 'attention/self/value/bias',
                    prefix + 'attention/output/dense/kernel',
                    prefix + 'attention/output/dense/bias',
                ],
                'Transformer-%d-MultiHeadSelfAttention-Norm' % i: [
                    prefix + 'attention/output/LayerNorm/beta',
                    prefix + 'attention/output/LayerNorm/gamma',
                ],
                'Transformer-%d-FeedForward' % i: [
                    prefix + 'intermediate/dense/kernel',
                    prefix + 'intermediate/dense/bias',
                    prefix + 'output/dense/kernel',
                    prefix + 'output/dense/bias',
                ],
                'Transformer-%d-FeedForward-Norm' % i: [
                    prefix + 'output/LayerNorm/beta',
                    prefix + 'output/LayerNorm/gamma',
                ],
            })

        mapping = {k: v for k, v in mapping.items() if k in self.layers}
        # mapping就是一个字典
        # 但是其中key是string，也就是各个层的名字
        # value是list of string，其中只有一个string，就是官方的名字
        return mapping

    def load_weights_from_checkpoint(self, checkpoint, mapping=None):
        """根据mapping从checkpoint加载权重
        """
        mapping = mapping or self.variable_mapping()

        weight_value_pairs = []
        for layer, variables in mapping.items():
            # 外部传入的keras层, dictionary, 不指定默认={}
            layer = self.layers[layer]
            weights = layer.trainable_weights
            values = [self.load_variable(checkpoint, v) for v in variables]
            weight_value_pairs.extend(zip(weights, values))
        keras.backend.batch_set_value(weight_value_pairs)

import json
def build_transformer_model(config_path=None,
                            checkpoint_path=None,
                            model='bert',
                            application='encoder',
                            return_keras_model=True,
                            **kwargs):
    """根据配置文件构建模型，可选加载checkpoint权重
    """
    config = kwargs
    if config_path is not None:
        config.update(json.load(open(config_path)))
    if 'max_position' not in config:
        # max_position_embeddings: 512
        config['max_position'] = config.get('max_position_embeddings')
    if 'dropout_rate' not in config:
        # "hidden_dropout_prob": 0.1,
        config['dropout_rate'] = config.get('hidden_dropout_prob')
    model, application = model.lower(), application.lower()
    models = {
        'bert': BERT,
    }
    MODEL = models[model]
    transformer = MODEL(**kwargs)
    transformer.build(**kwargs)
    if checkpoint_path is not None:
        transformer.load_weights_from_checkpoint(checkpoint_path)
    if return_keras_model:
        return transformer.model
    else:
        return transformer

roberta_dir = "C:/JupyterWorkspace/sentiment-keras4bert/roberta"
config_path = f"{roberta_dir}/bert_config.json"
ckpt_path = f"{roberta_dir}/bert_model.ckpt"
dict_path = f"{roberta_dir}/vocab.txt"

tokenizer = Tokenizer(dict_path, do_lower_case=True)
model = build_transformer_model(config_path=config_path, checkpoint_path=ckpt_path, return_keras_model=True)

model.summary()

token_ids, segment_ids = tokenizer.encode('语言模型')

print('\n ===== predicting ===== \n')
print(model.predict([np.array([token_ids]), np.array([segment_ids])]))
