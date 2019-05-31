# gensim-fast2vec


相关博客：[gensim-fast2vec改造、灵活使用大规模外部词向量（具备OOV查询能力）](https://blog.csdn.net/sinat_26917383/article/details/90713664)


本篇是继 [极简使用︱Gemsim-FastText 词向量训练以及OOV（out-of-word）问题有效解决](https://blog.csdn.net/sinat_26917383/article/details/83041424) 之后，让之前的一些旧的"word2vec"具备一定的词表外查询功能。

还有一个使用场景是很多开源出来的词向量很好用，但是很大，用gensim虽然可以直接用，如果能尽量节省一些内存且比较集中会更好，同时如果有一些OOV的功能就更好了，于是笔者就简单抛砖引玉的简单写了该模块。


**譬如以下这些大规模词向量：**

## 1  Embedding/Chinese-Word-Vectors

地址：https://github.com/Embedding/Chinese-Word-Vectors

![在这里插入图片描述](https://img-blog.csdnimg.cn/20181031210647496.?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzI2OTE3Mzgz,size_16,color_FFFFFF,t_70)

## 2 腾讯AI Lab开源大规模高质量中文词向量数据
地址：https://ai.tencent.com/ailab/nlp/embedding.html





----------


# 应用一：gensim - fast2vec 简单查询功能


笔者自己使用的是腾讯的词向量，自己清洗过之后使用，还是很不错的。

```
    # 初始化
    fv = fast2vec()
    
    # 加载模型
    fv.load_word2vec_format(word2vec_path = 'Tencent_AILab_ChineseEmbedding_refine.txt')  # 加载.txt文件
    fv.model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path,binary=True) # 加载.bin文件
    
    # 求相似词 - 不带oov
    fv.model.most_similar(positive=[ '香味'], negative=['香'], topn=10)  # 带否定词,不带OOV
    fv.model.most_similar(positive='香味', topn=10)  # 词条求相似,不带OOV
    
    # 求相似词 - 带oov
    fv.most_similar('苹果', topn=10,min_n = 1, max_n = 3)# 词条求相似,带OOV
    fv.most_similar('苹果',negative = ['手机','水果'], topn=10,min_n = 1, max_n = 3)# 带否定词,带OOV
    
    # 词条之间求相似
    fv.similarity('香味','香')
    
    # 词条拆分,用在截取词向量环节
    fv.compute_ngrams('香')
    
    # 得到词向量
    fv.wordVec('香味')
    
    # 其他word2vec用法
    fv.model
```

本模块是基于gensim-word2vec使用的，那么之前的所有功能都是可以继续使用的


----------

# 应用三：拓词

通过一些种子词进行相近词查找。

```
vocab_dict_word2vec = fv.EmbeddingFilter('香味',topn = 100,min_n = 1, max_n = 3,GotWord2vec = False)
vocab_dict_word2vec = fv.EmbeddingFilter(['香味','香气'],topn = 100,min_n = 1, max_n = 3,GotWord2vec = False)
```

其中主函数中`TencentEmbeddingFilter`中参数分别代表：

 - topn ，每个单词查找相近词的范围，一般为topn = 50;
- min_n = 1，OOV功能中，拆分词，最小n-grams
- max_n = 3，OOV功能中，拆分词，最大n-grams
- GotWord2vec ,GotWord2vec = True为可获得拓展词的词向量，可以保存；GotWord2vec = False的时候，只能返回附近的词群
该函数可以输入单词条，可以输入词语List。


其中，OOV问题如何解决的思路可参考： [极简使用︱Gemsim-FastText 词向量训练以及OOV（out-of-word）问题有效解决](https://blog.csdn.net/sinat_26917383/article/details/83041424) 


----------


# 应用二：gensim - fast2vec 抽取部分词向量

大规模的词向量想截取其中一部分使用，一种方式就是查询之后保存。首先就需要准备一些种子词，然后通过托词，找到相关联的一批词，然后进行保存。

```
seed_words = ['牛仔骨','泡椒牛蛙']
# 查询
vocab_dict_word2vec = fv.EmbeddingFilter(seed_words,topn = 100,min_n = 1, max_n = 3,GotWord2vec = True)
# 保存
fv.wordvec_save2txt(vocab_dict_word2vec,save_path = 'test_word2vec_1.txt',encoding = 'utf-8-sig')  
```


抽取部分词向量的前提是，提供一些要截取的这个行业的种子词，然后查找这些词附近所有的词群（most_similar），同时每个词拆分开来的词条也要记录（compute_ngrams，用于OOV）。

这些词，导出，保存在.txt之中。


----------


# 应用三：gensim - entity2vec 实体词抽取与查询（类item2vec用法）

这个是建立在有庞大的词向量训练基础，譬如腾讯的大规模词向量，里面有非常多的词，这些词一部分也是可以用来当作item2vec的用法。

譬如，一个简单案例，我要做针对菜谱的查询，那么我这边准备好了一些菜式名称，然后截取一部分出来，供以后不断使用。

```
items = ['牛仔骨','泡椒牛蛙','农家小炒肉','目鱼大烤','龙虾意面','榴莲酥','越式牛肉粒']

# 拓词 +  保存
vocab_dict_word2vec = fv.EmbeddingFilter(items,GotWord2vec = True)
fv.wordvec_save2txt(vocab_dict_word2vec,save_path = 'food2vec.txt',encoding = 'utf-8-sig')

# 加载新模型
fv2 = fast2vec()
fv2.load_word2vec_format(word2vec_path = 'food2vec.txt')

# 查询
 fv2.entity_similar('牛仔骨',items,topn=500)

>>> [('牛仔骨', 1.0),
>>>  ('农家小炒肉', 0.7015136480331421),
>>>  ('榴莲酥', 0.6885859966278076),
>>>  ('泡椒牛蛙', 0.6880079507827759),
>>>  ('龙虾意面', 0.6354280710220337),
>>>  ('越式牛肉粒', 0.6056148409843445),
>>>  ('目鱼大烤', 0.6046081185340881)]

```

其中，`entity_similar`，就是查询的时候只能显示`items`，提供的词群里面的内容，还是一个比较简单的应用。

本模块是非常有意思的一个模块，当然，虽然有OOV的功能，一些生僻菜名还是很难根据线索找到他们的词向量，如何解决这个问题，是个可以后续研究的地方。


