import numpy
import json
import collections
#########################################数据读取###############################################
def GetData(path):
    VSM = numpy.fromfile(path)
    VSM = VSM.reshape(18828,len(VSM)//18828)
    print(VSM)
    return VSM
def GetLabel(path):
    file = open(path,'rb')
    label = json.load(file)
    return numpy.array(label)
########################################划分数据集与测试集##############################################33
def DataDivide(label,VSM):
    """
    将数据集分为测试数据集和训练数据集
    :param label: ndarray of label
    :param VSM: ndarray of VSM
    :return:
    """
    data = numpy.c_[VSM,Label]
    label_list = list(label)
    con = collections.Counter(label_list)
    pro = list(dict(sorted(dict(con.most_common()).items(),key = lambda k:k[0])).values())
    start = [0]#每个类别的起始地址
    cal = 0
    for i in range(len(pro)):
        cal += pro[i]
        start.append(cal)
    print(start)
    test_pos = list(numpy.trunc(numpy.array(pro) * 0.2))
    test_range = []
    train_range = []
    for i in range(20):
        a = []
        a.append(start[i])
        a.append(int(test_pos[i]+start[i])+1)
        test_range.append(a)
        b = []
        b.append(int(test_pos[i]+start[i])+1)
        b.append(start[i+1])
        train_range.append(b)
    test_data = data[test_range[0][0]:test_range[0][1],:]
    print(len(test_data))
    train_data = data[train_range[0][0]:train_range[0][1],:]

    for i in range(19):
        c = data[test_range[i+1][0]:test_range[i+1][1],:]
        d = data[train_range[i+1][0]:train_range[i+1][1],:]
        test_data = numpy.r_[test_data,c]
        train_data = numpy.r_[train_data,d]
    train_data.tofile(r"E:\data mining\train_data.txt")
    test_data.tofile(r"E:\data mining\test_data.txt")
    return train_data,test_data
def Dividexy(data,pos):
    """

    :param data:ndarray型变量数据集，前pos列为输入，后一列为label
    :pos:表示标签和数据的分界线
    :return: 返回数据和标签的矩阵
    """
    x_data = data[:,0:pos]
    y_data = data[:,pos:pos+1]
    return x_data,y_data
##########################################KNN###################################################
def Disdence(x_train,x_test):
    """

    :param x_train:训练数据集
    :param x_test: 测试数据集
    :return: 返回M * N维的矩阵，表示第M个训练集数据到第N个测试集数据的cosine距离。
    """
    print("----------------------计算距离------------------")
    inner_product = numpy.dot(x_train, x_test.T)
    print(inner_product.shape)
    norm_train = numpy.linalg.norm(x_train, ord=2, axis=1, keepdims=False)
    norm_test = numpy.linalg.norm(x_test, ord=2, axis=1, keepdims=False)
    norm_nd = numpy.dot(numpy.array([norm_train]).T, numpy.array([norm_test]))
    distance = inner_product / norm_nd
    print(distance.shape)
    print("-----------------距离计算结束--------------------")
    distance.tofile('E:\data mining\distance.txt')
    return numpy.nan_to_num(distance)

def KNN(distance,y_train,K):
    """

    :param distance: 距离矩阵，第I列表示第I个测试数据到各个训练数据的cos距离
    :param y_train: 训练数据标签
    :param K: KNN的K值
    :return: 返回测试数据集上各个文档的类别列表，ndarray型
    """
    print("--------------KNN分类----------------")
    num_test = len(distance[0])#获得测试数据数量
    y_predict = []#预测类别列表
    for i in range(num_test):
        a = y_train[distance[:,i].argsort()]
        a = a.reshape(len(a), )
        a = a.tolist()
        a = list(reversed(a))
        NN = a[0:K:]
        class_bag = collections.Counter(NN)
        predict_label = class_bag.most_common(1)
        y_predict.append(list(dict(predict_label).keys())[0])
    print('----------------KNN分类结束-----------------------')
    return numpy.array(y_predict).reshape(len(y_predict),1)
def evaule(y_predict,y_test):
    """
    评估模型，计算预测正确率
    :param y_predict:
    :param y_test:
    :return:
    """
    M = len(y_predict)
    a = y_predict - y_test
    a = a.reshape(len(a), )
    b = collections.Counter(a.tolist())
    return b[0.0]/M


##############################################main()############################################
path_VSM = r'E:\data mining\VSM.txt'
path_Label = r'E:\data mining\label.txt'
VSM = GetData(path_VSM)
Label = GetLabel(path_Label)
data = numpy.c_[VSM,Label] #数据集
train_data,test_data = DataDivide(Label,VSM)

vec_length = len(VSM[0])
x_train,y_train = Dividexy(train_data,vec_length)
x_test,y_test = Dividexy(test_data,vec_length)

distance = Disdence(x_train,x_test)#距离矩阵，第I列表示第I个测试数据到各个训练数据的cos距离
#print(distance)

y_predict = KNN(distance,y_train,11)
y_predict.tofile(r"E:\data mining\y_predict.txt")
print('the accurency is %f'%evaule(y_predict,y_test))






