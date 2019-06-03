from tqdm import tqdm
import numpy as np
import gensim

class fast2vec():
    '''
    # 初始化
    fv = fast2vec()
    
    # 加载模型
    fv.load_word2vec_format(word2vec_path = 'Tencent_AILab_ChineseEmbedding_refine.txt')  # 加载.txt文件
    fv.model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path,binary = True) # 加载.bin文件
    
    # 求相似词
    fv.model.most_similar(positive=[ '香味'], negative=['香'], topn=10)  # 带否定词,不带OOV
    fv.model.most_similar(positive='香味', topn=10)  # 词条求相似,不带OOV
    
    fv.most_similar('苹果', topn=10,min_n = 1, max_n = 3)# 词条求相似,带OOV
    fv.most_similar(['苹果'], topn=10,min_n = 1, max_n = 3)# 词条求相似,带OOV,支持list
    fv.most_similar('苹果',negative = ['手机','水果'], topn=10,min_n = 1, max_n = 3)# 带否定词,带OOV
    
    # 词条之间求相似
    fv.similarity('香味','香')
    
    # 词群之间求相似
    w1 = ['香味','苹果']
    w2 = ['香味很好闻','苹果']
    fv.n_similarity(w1,w2)    # 两个词列表的相似性
    
    # 词条拆分,用在截取词向量环节
    fv.compute_ngrams('香')
    
    # 截取词向量
        # 从大的词向量截取一部分内容,并保存
    vocab_dict_word2vec = fv.EmbeddingFilter('香味',topn = 100,min_n = 1, max_n = 3,GotWord2vec = True)
    fv.wordvec_save2txt(vocab_dict_word2vec,save_path = 'test_word2vec_1.txt',encoding = 'utf-8-sig')  
    
    # 得到词向量
    fv.wordVec('香味',min_n = 1, max_n = 3)
    
    # 其他word2vec用法
    fv.model
    '''
    def __init__(self): # 
        pass
    
    def load_word2vec_format(self,word2vec_path = 'Tencent_AILab_ChineseEmbedding_refine.txt'):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path,binary=False)
        self.model.init_sims(replace=True)  # 神奇，很省内存，可以运算most_similar
        
    def compute_ngrams(self,word, min_n = 1, max_n = 3):
        '''
        词条拆解
        '''
        #BOW, EOW = ('<', '>')  # Used by FastText to attach to all words as prefix and suffix
        extended_word =  word
        ngrams = []
        for ngram_length in range(min_n, min(len(extended_word), max_n) + 1):
            for i in range(0, len(extended_word) - ngram_length + 1):
                ngrams.append(extended_word[i:i + ngram_length])
        return list(set(ngrams))


    def wordVec(self,word,min_n = 1, max_n = 3):
        '''词向量函数(带OOV)
            ngrams_single/ngrams_more,主要是为了当出现oov的情况下,最好先不考虑单字词向量
        '''
        # 确认词向量维度
        word_size = self.model.wv.syn0[0].shape[0]   
        # 计算word的ngrams词组
        ngrams = self.compute_ngrams(word,min_n = min_n, max_n = max_n)
        # 如果在词典之中，直接返回词向量
        if word in self.model.wv.vocab.keys():
            return self.model[word]
        else:  
            # 不在词典的情况下
            word_vec = np.zeros(word_size, dtype=np.float32)
            ngrams_found = 0
            ngrams_single = [ng for ng in ngrams if len(ng) == 1]
            ngrams_more = [ng for ng in ngrams if len(ng) > 1]
            # 先只接受2个单词长度以上的词向量
            for ngram in ngrams_more:
                if ngram in self.model.wv.vocab.keys():
                    word_vec += self.model[ngram]
                    ngrams_found += 1
                    #print(ngram)
            # 如果，没有匹配到，那么最后是考虑单个词向量
            if ngrams_found == 0:
                for ngram in ngrams_single:
                    word_vec += self.model[ngram]
                    ngrams_found += 1
            if word_vec.any():
                return word_vec / max(1, ngrams_found)
            else:
                raise KeyError('all ngrams for word %s absent from model' % word)

    def most_similar(self,word,negative = None,topn=50,min_n = 1, max_n = 3):
        '''
            具有OOV功能
            根据word,托词并给出相似词列表
            word,支持str,list
            negative，支持str,list
        '''
        if isinstance(word,str):
            vec = self.wordVec(word,min_n = min_n, max_n = max_n)
            vec = [vec]
        if isinstance(word,list):
            vec = [self.wordVec(w,min_n = min_n, max_n = max_n) for w in word ]
        
        if negative == None:
            return self.model.most_similar(positive = vec, topn=topn)# ,negative=['水果']
        else:
            if isinstance(negative,str):
                negative = [negative]
            return self.model.most_similar(positive = vec ,negative = negative, topn=topn)# ,negative=['水果']


    def EmbeddingFilter(self,select_dict_list,topn = 100,min_n = 1, max_n = 3,GotWord2vec = False):
        '''词向量截取
        输入：
            - select_dict_list,需要拓词的种子词包
            - self.model,fasttext模型
            - topn = 100,取前100个单词
            - GotWord2vec = False,是否只是返回单词，而不返回对应的词向量
            - min_n/max_n,分别代表使用fasttext时候应该n-grams多少
        输出：
            - GotWord2vec = True,输出拓词后的词向量，dict形态
            - GotWord2vec = False,输出扩词的词条list，错词list
        功能：
            这样的输出才有之后可以使用OOV的可能性。    
        '''
        word2vec_total_words = list(self.model.wv.vocab.keys())
        vocab_dict = {}
        error_words = []
        if isinstance(select_dict_list,str):
            select_dict_list = [select_dict_list]
        
        # 得到单词 + 相似性 列表
        for word in select_dict_list:
            if word in word2vec_total_words:
                sim_words = self.model.most_similar(word,topn = topn)
                if GotWord2vec:
                    # 在GotWord2vec开启的时候，就会把查询词条，一系列的下位词都给加到词向量之中
                    expand_words = self.compute_ngrams(word,min_n = min_n, max_n = max_n)
                    vocab_dict[word] = [s[0] for s in sim_words] + expand_words
                else:
                    vocab_dict[word] = [s[0] for s in sim_words]
            else:
                error_words.append(word)
        
        if GotWord2vec:
            # 得到词向量表
            vocabs = []
            for k,v in vocab_dict.items():
                vocabs.extend([k] + v)
            vocab_list = list(set(  vocabs  ))
            #print(vocab_list)
            vocab_dict_word2vec = {}
            for wo in vocab_list:
                if wo in word2vec_total_words:
                    vocab_dict_word2vec[wo] = self.model[wo]
            return vocab_dict_word2vec
        print('fail load words has : %s'%error_words)
        return vocab_dict

    def wordvec_save2txt(self,vocab_dict_word2vec,save_path = 'Tencent_to_cec.txt',encoding = 'utf-8-sig'):
        # 保存下来的词向量(字典型)，存储成为.txt格式
        '''
        input:
            dict,{word:vector}
        '''
        length = len(vocab_dict_word2vec)
        size = len(list(vocab_dict_word2vec.values())[0])
        f2 = open(save_path,'a')
        f2.write('%s %s\n'%(length,size))
        f2.close()

        for keys_,values_ in vocab_dict_word2vec.items():
            f2 = open(save_path,'a',encoding =encoding)
            f2.write(keys_ + ' ' + ' '.join([str(i) for i in list(values_)]) + '\n')
            f2.close()
        pass

    def similarity(self,word1,word2):
        # 词 - 词 求相似
        if (isinstance(word1,list)) or (isinstance(word2,list)):
            raise KeyError('all words need str.' )
        return self.model.similarity(word1,word2)    # 两个词的相似性距离
    
    def n_similarity(self,w1,w2):
        # 词条相似
        _w1 = [w for w in w1 if w in self.model.wv.vocab.keys() ]
        _w2 = [w for w in w2 if w in self.model.wv.vocab.keys() ]

        if len(_w1) == 0:
            raise KeyError ("word '{}' not in vocabulary".format( '_'.join(w1)   ) )
        if len(_w2) == 0:
            raise KeyError ("word '{}' not in vocabulary".format( '_'.join(w2)   ) )
        return self.model.n_similarity(_w1,_w2)    # 两个词列表的相似性距

    def entity_similar(self,word,entitys,negative = None,min_n = 1, max_n = 3,topn=200):
        '''
        main_words,主要items有哪些,做一个简单筛选
        '''
        searched_words = self.most_similar(word, topn=topn,min_n = min_n, max_n = max_n)
        output = []
        for word in searched_words:
            if word[0] in entitys:
                output.append(word)
        return output