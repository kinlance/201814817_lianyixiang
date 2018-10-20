import nltk
import os
import chardet
import string
from textblob import TextBlob
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from tkinter import _flatten#用于拉直文档列表
import collections
import numpy
import json

######################################数据集读入函数###################################################3
def readfile(path):#读取文件函数，返回string
    # 文件读取，由于文件编码方式未知，所以要首先预测文件的编码方式，然后进行解码
    file = open(path, "rb")  # 打开文件，以二进制只读打开防止编码报错
    f_read = file.read()
    f_charinfo = chardet.detect(f_read)  # f_charinfo 存储文档编码格式
    # print(f_charinfo)  #查看文件编码格式
    f_read_decode = f_read.decode('ISO-8859-1')  # 解码文档
    file.close()
    return f_read_decode
def savefile(path,result):#存储文件
    with open(path,'w',errors='ignore') as file:
        file.write(result)
def catalogueread(inputpath):#将目录中所有文档读入到一个list当中（此方法对存储空间要求较大）
    """

    :param inputpath:
    :return:
    """
    fatherLists = os.listdir(inputpath)#获取父目录下所有子文件夹的名称
    corpus = []#用于存储所有文档集合的列表
    for eachDir in fatherLists:
        eachpath = inputpath + '\\' + eachDir + '\\'#填充子文件夹路径
        childLists = os.listdir(eachpath)#获取子文件夹中所有文件的名称
        for eachFile in childLists:
            eachpathFile = eachpath + eachFile#生成每个文件的路径
            print('\033[0;31;40m%s\033[0m'%eachpathFile)   #查看文件路径
            corpus.append(readfile(eachpathFile))#将文件添加到列表中
        #print(corpus)
    return corpus
def CreateLabelList(inputpath):
    """

    :param inputpath:数据集文件夹目录
    :return: 文档分类标签
    """
    fatherLists = os.listdir(inputpath)  # 获取父目录下所有子文件夹的名称
    label = []  # 用于生成label
    label_class = 0  # 类别起始计数
    for eachDir in fatherLists:
        eachpath = inputpath + '\\' + eachDir + '\\'  # 填充子文件夹路径
        label_class = label_class + 1  # 按文件夹进行标签分类
        childLists = os.listdir(eachpath)  # 获取子文件夹中所有文件的名称
        for eachFile in childLists:
            eachpathFile = eachpath + eachFile  # 生成每个文件的路径
            print('\033[0;31;40m%s\033[0m' % eachpathFile)  # 查看文件路径
            label.append(label_class)
    con = json.dumps(label,ensure_ascii=False)
    with open(r'E:\data mining\label.txt','wb') as f:
        f.write(con.encode('utf-8'))
    return label
#####################################分词#########################################
def CleanLines(Documents):#输入文档（string），输出去除数字的文档（string）
    outtable = " "*len(string.digits+string.punctuation)
    intable = string.digits+string.punctuation #去除符号
    maketrans = str.maketrans(intable,outtable)
    cleanline = Documents.translate(maketrans)
    return cleanline
def tokenization(Documents):#输入文档（string），输出分词列表（list）
    article = TextBlob(Documents)
    words = article.words
    return words
##########################################normalization and steming# ################################################3
def steming(words):#输入词列表，输出还原原型后的词列表
    stemmer = SnowballStemmer("english")#选择一种语言
    stem_words = []
    for word in words:
        stem_words.append(stemmer.stem(word))
    return stem_words
def lowlitter(ini_words):#输入词列表，输出小写化的词列表
    lines = []
    for word in ini_words:
        lines.append(str.lower(word))
    return lines
def drop_stopwords(segment):#输入词列表，输出去掉停用词的词列表
    filtered = [w for w in segment if not w in stopwords.words('english')]
    return filtered
######################################语料库预处理##################################################
def preprocessing(corpus):#输入文档列表
    normalList = []
    print("-------------------------预处理----------------------------------")
    for Document in corpus:
        Document_no_digit = CleanLines(Document)
        Document_cut = tokenization(Document_no_digit)
        Document_steming = steming(Document_cut)
        Document_low = lowlitter(Document_steming)
        Document_drop = drop_stopwords(Document_low)
        normalList.append(Document_drop)
        #print(Document_drop)
    print("-------------------------------预处理结束---------------------------------")
    return normalList
##############################################计算词频###############################################
def CountWords(Document):
    """
    计算文档词频
    :param Document:list 文档分词后的列表
    :return: collection.counter()类
    """
    Frequency = collections.Counter(Document)
    d = dict(Frequency.most_common())
    json_d = json.dumps(d)
    f = open(r"E:\data mining\词典统计.txt",'wb')
    f.write(json_d.encode('utf-8'))
    f.close()
    #Frequency = nltk.FreqDist(Document) #此方法不好用，暂时搁置
    return (Frequency)
def visual_Frequency(DocumentList):
    """
    打印数据集中的词频分布
    :param DocumentList: 二维列表，每一行存储一篇文章的分词结果
    :return: none，打印词频分布图
    """
    database = list(_flatten(DocumentList))
    vocabulary = nltk.FreqDist(database)
    vocabulary.plot()
def DocumentFrequency(DocumentList):
    """
    计算每个文档的单独词频
    :param DocumentList:list 文档列表
    :return: list of collection.counter() 文档词频列表
    """
    Document_Frequency = []
    for Documnet in DocumentList:
        Document_Frequency.append(collections.Counter(Documnet))
    return Document_Frequency
####################################生成词典##################################################
def Create_Vocabulary(vocabulary_list,slide_st,slide_sp):
    """
    根据输入的词频统计生成词典
    :param vocabulary_list: counter()类
    :param slide_st: 词典起始词频序列
    :param slide_sp: 词典结束词频序列
    :return:
        生成词典的词频统计
    """
    vocabulary_dict = dict(vocabulary_list.most_common())
    vocabulary_name = list(vocabulary_dict.keys())
    vocabulary_count = list(vocabulary_dict.values())
    vocabulary = {}
    vocabulary_name = vocabulary_name[slide_st:slide_sp:1]
    vocabulary_count = vocabulary_count[slide_st:slide_sp:1]


    for i in range(len(vocabulary_name)):
        vocabulary[vocabulary_name[i]] = vocabulary_count[i]
    json_vocabulary = json.dumps(vocabulary,ensure_ascii=False)
    f = open(r'E:\data mining\vocabulary.txt','wb')
    f.write(json_vocabulary.encode('utf-8'))
    f.close()
    return vocabulary
###################################生成VSM##################################################
def IDF(vocabulary):
    """
    生成整个数据集的IDF
    :param vocabulary:dict of ‘词典’ = 词频
    :return: array of IDF
    """
    vector = list(vocabulary.values())
    normalization= numpy.array(vector)
    N = normalization.sum()
    IDF_v = numpy.log(N/normalization)
    IDF_v.tofile(r"E:\data mining\IDF.txt")
    return IDF_v

def TF(Document, Vocabulary):
    """
    计算一篇文档的词频
    :param Document:collections.counter()
    :param Vocabulary: dict (words,Frequency)
    :return: array of TF
    """
    Document_dict = dict(Document.most_common())#先将collection.counter格式转换为字典
    keys = list(Vocabulary.keys())#取词典的单词
    Document_count = {}
    for key in keys:
        if Document_dict.get(key):
            Document_count[key] = Document_dict[key]
        else:
            Document_count[key] = 0
    #print(Document_count) #打印词典统计
    vector = numpy.array(list(Document_count.values()))
    arlfa = 0.1
    max_element = vector.max()
    TF_vector = arlfa + (1-arlfa)*vector/(max_element+1)
    TF_vector.tofile(r"E:\data mining\TF.txt")
    return TF_vector

def CreateVSM(DocumentList):
    """
    生成向量空间模型
    :param DocumentList:分词结果列表
    :return: list 每篇文档的词向量组成的列表
    """
    database = list(_flatten(DocumentList))#将所有文档整合成语料库
    Vocabulary = Create_Vocabulary(CountWords(database),10000,23000)#创建词典，词典大小为b-a
    Document_Frequency =  DocumentFrequency(DocumentList)
    IDF_v = IDF(Vocabulary)
    #print(type(IDF_v))
    TF_list = []
    VSM = []
    for Document in Document_Frequency:
        TF_v = TF(Document, Vocabulary)
        TF_list.append(TF_v)
        VSM_v = TF_v*IDF_v
        print(IDF_v,TF_v,VSM_v)
        #print(VSM_v)
        VSM.append(VSM_v)

    #保存所有文档的TF
    TF_array = numpy.array(TF_list)
    print("the TF_array is ",TF_array.shape)
    TF_array.tofile(r"E:\data mining\TF.txt")
    #保存所有文档的向量列表
    VSM_array = numpy.array(VSM)
    print(VSM_array.shape)
    VSM_array.tofile(r'E:\data mining\VSM.txt')

    return VSM_array
#####################################评估词向量##################################################


#main()
path = r'E:\data mining\20news-18828'
corpus = catalogueread(path)
label = CreateLabelList(path)
#print(label)
#print(len(label))
DocumentList = preprocessing(corpus)#所有文档的最终分词集合
#visual_Frequency(DocumentList)
VSM = CreateVSM(DocumentList)
#print(VSM)
#print(VSM.shape)






# for i in range(len(DocumentList)):
#     print(DocumentList[i])
# print(len(DocumentList))


#end

