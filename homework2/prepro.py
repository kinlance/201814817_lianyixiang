import os
import chardet
import string
from textblob import TextBlob
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from tkinter import _flatten#用于拉直文档列表
import json
from nltk.corpus import wordnet
########################################文档读取#######################################################
def readfile(path):
    """

    :param path:读取文件的路径
    :return: 返回string
    """
    file = open(path,"rb")
    f_read = file.read()
    f_charinfo = chardet.detect(f_read)
    if f_charinfo["encoding"] != None:
        code = f_charinfo["encoding"]
    else :
        code = "ISO-8859-1"
    f_read_decoder = f_read.decode(code,errors='ignore')
    file.close()
    return f_read_decoder
def CreateCatelog(fatherpath):
    """

    :param fatherpath:20news-18828目录路径
    :return: 返回dict ,key = 文件路径,value = 类别
    """
    fatherLists = os.listdir(fatherpath)

    classlog = []
    for eachdir in fatherLists:
        catelog = []  # key = 文件名，value = 类别
        eachpath = fatherpath+ '\\' + eachdir +'\\'
        childlists = os.listdir(eachpath)

        for eafile in childlists:  #子文件夹中每个文件的循环
            eachfilepath = eachpath + eafile
            print('\033[0;31;40m%s\033[0m' % eachfilepath)  # 查看文件路径
            catelog.append(eachfilepath)
        classlog.append(catelog)

    return classlog
############################################预处理################################################
def CleanLines(Documents):
    """

    :param Documents:输入string
    :return: 返回去掉符号的string
    """
    #输入文档（string），输出去除数字的文档（string）
    outtable = " "*len(string.digits+string.punctuation)
    intable = string.digits+string.punctuation #去除符号
    maketrans = str.maketrans(intable,outtable)
    cleanline = Documents.translate(maketrans)
    return cleanline
def tokenization(Documents):
    """

    :param Documents:输入string
    :return: 输出分词列表
    """
    #输入文档（string），输出分词列表（list）
    article = TextBlob(Documents)
    words = article.words
    return words

def steming(words):
    """

    :param words:输入分词列表
    :return: 输出词形还原词列表
    """
    #输入词列表，输出还原原型后的词列表
    stemmer = WordNetLemmatizer()#选择一种语言
    stem_words = []
    for word in words:
        noun = stemmer.lemmatize(word,'v')
        verb = stemmer.lemmatize(word,'n')
        stem_words.append(verb)
    return stem_words
def lowlitter(ini_words):#输入词列表，输出小写化的词列表
    lines = []
    for word in ini_words:
        lines.append(str.lower(word))
    return lines
def drop_stopwords(segment):#输入词列表，输出去掉停用词的词列表
    filtered = [w for w in segment if not w in stopwords.words('english')]
    return filtered
def judgeword(segment):
    """

    :param segment:list of word
    :return: 去除非英文单词的list
    """
    lines = []
    for word in segment:
        if wordnet.synsets(word) :
            lines.append(word)
    return lines
def ReadDoc(catelog):
    classlog = {}
    for i in range(len(catelog)):
        a = []
        for j in range(len(catelog[i])):
            c = readfile(catelog[i][j])
            #print(c, '\n') #每篇文章
            a.append(c)
        print(a)
        classlog[str(i + 1)] = a
    return classlog
def preprocessing(DocumentDict):
    classlog = {}
    print("____________________预处理__________________")
    for key in DocumentDict.keys():
        Documentlist = DocumentDict[key]
        classDoc = []
        for i in range(len(Documentlist)):
            Document_no_digit = CleanLines(Documentlist[i])
            Document_cut = tokenization(Document_no_digit)
            Document_steming = steming(Document_cut)
            Document_low = lowlitter(Document_steming)
            Document_drop = drop_stopwords(Document_low)
            Document_Judge = judgeword(Document_drop)
            classDoc.append(Document_Judge)
        classlog[key] = classDoc
        f = open("E:\data mining\homework2\document"+"\\"+key,"w")
        json_Document = json.dumps(classDoc)
        f.write(json_Document)
        f.close()

    print("____________________预处理结束______________")
    return classlog

#############################################分数据###################################################




#################################main（）##################################
if __name__ == "__main__":
    path = r'E:\data mining\20news-18828'
    catelog = CreateCatelog(path)
    Document = ReadDoc(catelog)
    print(len(Document["1"]))
    Document = preprocessing(Document)
    print(Document)

