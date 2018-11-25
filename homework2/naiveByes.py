import os
import chardet
import string
from textblob import TextBlob
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from tkinter import _flatten#用于拉直文档列表
import json
from nltk.corpus import wordnet
import collections
import math
import random
###################################################读取######################################################
def CreateCatelog(fatherpath):
    """

    :param fatherpath:20news-18828目录路径
    :return: 返回dict ,key = 文件路径,value = 类别
    """
    childlists = os.listdir(fatherpath)
    catelog = []
    for eafile in childlists:  #子文件夹中每个文件的循环
        eachfilepath = fatherpath +'\\'+ eafile
        print('\033[0;31;40m%s\033[0m' % eachfilepath)  # 查看文件路径
        catelog.append(eachfilepath)

    return catelog
def ReadDoc(catelog):
    classlog = {}
    for i in range(len(catelog)):
        f = open(catelog[i],"r")
        json_f = json.loads(f.read())
        f.close()
        #print(json_f, '\n') #每篇文章
        classlog[str(i + 1)] = json_f
    return classlog
def prePossib(Docunment):
    """

    :param Docunment:输入dict{class 1 : list of Document}
    :return:返回一篇文档的先验概率
    """
    classnumber = []
    for key in Document.keys():
        classnumber.append(len(Document[key]))
    N = sum(classnumber)
    print(N)
    classpossible = []
    for i in range(len(classnumber)):
        classpossible.append(math.log(classnumber[i]/N))

    f = open(r'E:\data mining\homework2\类概率.json','w')
    json_classpossible = json.dumps(classpossible)
    f.write(json_classpossible)
    f.close()

    return classpossible

def WordPossible(Document):
    classcount = {}
    for key in Document.keys():
        wordcount = {}
        Documentflat = _flatten(Document[key])
        a = collections.Counter(Documentflat)
        N = sum(a.values())
        count = dict(a.most_common())
        for key1 in count.keys():
            wordcount[key1] = math.log((count[key1]+1)/(N+2))
        classcount[key] = wordcount

    f = open(r'E:\data mining\homework2\词概率.json','w')
    json_classpossible = json.dumps(classcount)
    f.write(json_classpossible)
    f.close()
    return classcount
################################################贝叶斯分类器#########################################################
def backprediction(WordList,Word_possible):
    """

    :param WordList:所要判断文档的词列表
    :param Word_possible: 单词后验概率矩阵
    :return: 该篇文章的后验概率
    """
    back_prediction = []
    for label in Word_possible:
        count = 0
        for word in WordList:
            if word in word_possible[label]:
                count += word_possible[label][word]
        back_prediction.append(count)


    return back_prediction

def Byes(prepossible,wordpossible,test_data):
    test_lable = []
    test_possible = []
    for key in test_data.keys():
        for i in range(len(test_data[key])):
            document_possible = []
            back_possible = backprediction(test_data[key][i],wordpossible)
            for j in range(len(back_possible)):
                document_possible.append(back_possible[j]+prepossible[j])
            test_lable.append(key)
            test_possible.append(document_possible)

    f = open(r'E:\data mining\homework2\文档概率.json','w')
    for i in range(len(test_possible)):
        f.write(json.dumps(test_possible[i]))
        f.write("\n")
    f.close()

    f = open(r'E:\data mining\homework2\test_label.json','w')
    for i in range(len(test_lable)):
        f.write(json.dumps(test_lable[i]))
        f.write('\n')
    f.close()

    return test_possible,test_lable

def evalue(test_possible,test_label):
    right = 0
    total = len(test_possible)
    for i in range(total):
        print(int(test_label[i]),test_possible[i].index(max(test_possible[i]))+1)
        if int(test_label[i]) == test_possible[i].index(max(test_possible[i]))+1:
            right += 1
    print(total,right)
    input()
    return right/total


# def bonuli(test_data,train_data):
#     classposible = ClassPossib(train_data)#每个类的概率p(ci)
#     wordpossible = WordPossible(train_data)
#     test_possible = {}
#     for key in test_data.keys():
#         test_documentlist = test_data[key] #每一类的文档列表
#         documentlist = []
#         for i in range(len(test_documentlist)):
#             test_eache_document = test_documentlist[i]
#             doc_class_possible = []
#             for key1 in classposible.keys():
#                 class_pos = classposible[key1] #该文章属于第I类的概率
#                 for word in test_eache_document:#计算每个单词的在key1类中出现的概率
#                     if word in wordpossible[key1].keys():
#                         class_pos += wordpossible[key1][word]
#                 doc_class_possible.append(class_pos)#一共20维，每一维是该文章属于key1类的概率
#             documentlist.append(doc_class_possible)
#
#         #############################################存储###############################
#         test_possible[key] = documentlist
#         json_predict = json.dumps(test_possible)
#         f = open(r'E:\data mining\homework2\伯努利分布.json','w')
#         f.write(json_predict)
#         f.close()
#         ################################################################################
#     return test_possible
# def accrency(bonuli):
#     right = 0
#     total = 0
#     for key in bonuli.keys():
#         for i in range(len(bonuli[key])):
#             total += 1
#
#             print(int(key),bonuli[key][i].index(max(bonuli[key][i]))+1)
#             if int(key) == bonuli[key][i].index(max(bonuli[key][i]))+1:
#                 right += 1
#     print(total,right)
#     return right/total








######################################分数据######################################################
def CreatDataset(Document):
    train_data = {}
    test_data = {}
    sum = 0
    for key in Document.keys():
        test = random.sample(Document[key],int(len(Document[key])*0.2))
        train = [x for x in Document[key] if x not in test ]
        test_data[key] = test
        train_data[key] = train
        sum += len(test)
        sum += len(train)


    json_testdata = json.dumps(test_data)
    json_traindata = json.dumps(train_data)
    f = open(r'E:\data mining\homework2\test_data.json','w')
    d = open(r'E:\data mining\homework2\train_data.json','w')
    f.write(json_testdata)
    d.write(json_traindata)
    f.close()
    d.close()


    return train_data,test_data

######################################nain()######################################################

if __name__ == "__main__":
    path = r'E:\data mining\homework2\document'
    catelog = CreateCatelog(path)
    Document = ReadDoc(catelog)
    train_data, test_data = CreatDataset(Document)
    class_possible = prePossib(Document)
    word_possible = WordPossible(Document)
    test_possible,test_label = Byes(class_possible,word_possible,test_data)
    print(evalue(test_possible,test_label))




    #print(Document['1'])
    #train_data,test_data = CreatDataset(Document)
    #onuli_possible = bonuli(test_data,Document)
    #print(bonuli_possible)
    #print(accrency(bonuli_possible))



