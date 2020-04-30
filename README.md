# Reimplement_bert4keras
自己实现的bert4keras，在一个文件内写出所有功能，并且尽量逻辑清晰
## 哲学

尽量只使用keras、tensorflow这种库，逐步不使用bert4keras库

用到的功能一定**自己**实现，没用到的功能尽量不实现

```
比如bert4keras里的Tokenizer类继承了BasicTokenizer类，但是两个类有高度相似性。
我就尽量把它合成一个类
原作者分离出基类的原因是另一个类SpTokenizer继承了BasicTokenizer，
但是这个类我没有用到
因此无需设计如此复杂
```
并且原设计增加了逻辑复杂性
```
以基础的encode方法为例
encode方法在BasicTokenizer中实现，其中一个很重要的过程是tokens_to_ids
tokens_to_ids方法在BasicTokenizer中实现，其实是逐个token使用token_to_id方法
token_to_id方法在BasicTokenizer中没实现，只是raise一个NotImplementedError
token_to_id方法的真正实现在Tokenizer类中
如果以上三个方法encode、tokens_to_ids、token_to_id都在一个类里
那么逻辑就相对清晰一些
```
（虽然我承认换个角度说这是OOP的优点之一）

## Reimplement 方法论

**逐步**、**渐进**实现功能，最终目的是用自己写的代码完全代替keras4bert框架

比如说官方编码句子(例子)[https://bert4keras.spaces.ac.cn/#_1]

```python
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
import numpy as np

config_path = '/root/kg/bert/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/root/kg/bert/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/root/kg/bert/chinese_L-12_H-768_A-12/vocab.txt'

tokenizer = Tokenizer(dict_path, do_lower_case=True)  # 建立分词器
model = build_transformer_model(config_path, checkpoint_path)  # 建立模型，加载权重

# 编码测试
token_ids, segment_ids = tokenizer.encode(u'语言模型')

print('\n ===== predicting =====\n')
print(model.predict([np.array([token_ids]), np.array([segment_ids])]))
```

代码在第一步实例化了一个Tokenizer()类

那么，首先实现出Tokenizer()类，这样就能把

```python
from bert4keras.tokenizers import Tokenizer
```

这一行删掉

最终复现的目的是开头的导入包只剩下：

```python
import tensorflow as tf
import keras
import numpy as np
```

同时，复现过程中，也遵循**逐步**、**渐进**的原则

接上面，复现Tokenizer类时，Tokenizer类继承了BasicTokenizer类

即

```python
class Tokenizer(BasicTokenizer):
    """具体代码
    """
```

那么在我的代码中就写成

```python
import bert4keras.tokenizers as T
class Tokenizer(T.BasicTokenizer):
    """具体代码
    """
```

并且把其中所有的bert4keras.tokenizers不在Tokenizer中实现的方法都先引用而不实现

因为现在主要目的是将Tokenizer类实现

比如Tokenizer类中用了load_vocab方法读取字典，而这个方法在Tokenizer类之外：

https://github.com/bojone/bert4keras/blob/master/bert4keras/tokenizers.py#L9

那么就把

```python
token_dict = load_vocab(token_dict)
```

改成

```python
import bert4keras.tokenizers as T
token_dict = T.load_vocab(token_dict)
```

load_vocab不急于实现，因为还没有直接用到它（但它也很重要，相当于直接用到了它，因为它在初始化函数__init__()中执行）

首先实现的是encode函数，因为我在上面的示例代码里使用到了。（同时还有__init__函数）

```python
import bert4keras.tokenizers as T

class Tokenizer(T.BasicTokenizer):
    """具体代码
    """
    def __init__(self):
        super().__init__(*args, **kwargs) # 这一步好像不需要，因为BasicTokenizer继承的object类
        """具体代码
        """
    
    def encode(self):
        """具体代码
        """

token_ids, segment_ids = tokenizer.encode(u'语言模型')
```

encode实现完之后，再实现BasicTokenizer类，之后逐步就可以把

```python
import bert4keras.tokenizers as T
```
这句话删除

这就是一个illustration，如何**逐步**，**渐进**学习、修改一个框架