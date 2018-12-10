import tensorflow as tf
import numpy as np
import pandas as pd
names = locals()


def creatData_train1(n,m,train_dataset,test_dataset): #创建训练数据集
    # num=index(max(value.(index(0-10)))
    ExtDataset=np.mat([None]*m)
    ExtDataset1 =np.mat([None]*m)
    dataset1 = train_dataset[:,1:n]
    maxIndex = np.argmax(dataset1,axis = 1)
    dataset2 =test_dataset[:,1:n]
    maxIndex2 =np.argmax(dataset2,axis =1)
    for i in range(train_dataset.shape[0]):
        subIndex = maxIndex[i,0]                       #求前n个值中最大值的索引
        subDataset = train_dataset[i,subIndex:subIndex+m]
        ExtDataset =np.vstack((ExtDataset,subDataset))
    for i in range(test_dataset.shape[0]):
        subIndex1 = maxIndex2[i, 0]  # 求前3000个值中最大值的索引
        subDataset1 = test_dataset[i, subIndex1:subIndex1 + m]
        ExtDataset1 = np.vstack((ExtDataset1, subDataset1))
    train_dataset = ExtDataset[1:,:]
    test_dataset =ExtDataset1[1:]
    #print(train_dataset.shape)
    return train_dataset,test_dataset


def GetData(class0, class1, class2, class3, class4, class5, class6, class7, class8, class9):
    a0,b0 = class0[:8,:],class0[8:,:]
    a1,b1 = class1[:8,:],class1[8:,:]
    a2,b2 = class2[:8,:],class2[8:,:]
    a3,b3 = class3[:8,:],class3[8:,:]
    a4,b4 = class4[:8,:],class4[8:,:]
    a5,b5 = class5[:8,:],class5[8:,:]
    a6,b6 = class6[:8,:],class6[8:,:]
    a7,b7 = class7[:8,:],class7[8:,:]
    a8,b8 = class8[:8,:],class8[8:,:]
    a9,b9 = class9[:8,:],class9[8:,:]
    test_dataset =np.vstack((a0,a1,a2,a3,a4,a5,a6,a7,a8,a9))
    test_labels = test_dataset[:,-1]
    train_dataset = np.vstack((b0,b1,b2,b3,b4,b5,b6,b7,b8,b9))
    train_labels =train_dataset[:,-1]
    return train_dataset,train_labels,test_dataset,test_labels


def classify(file_name):
    Filename = file_name
    # dataset = xlrd.open_workbook("C:\Users\hasee\Desktop\InputData.xlsx")
    data_set = pd.read_csv(Filename)
    data_set = np.mat(data_set)
    for i in range(10):
        names['class%s'%i] = np.mat([None]*data_set.shape[1])   #获取矩阵的列
        for j in range(data_set.shape[0]):  #遍历每一行
            if data_set[j, -1] == i:
                names['class%s' % i] = np.vstack((names['class%s'%i],data_set[j, :]))
        names['class%s' % i] = names['class%s'%i][1:,:]
    return class0, class1, class2, class3, class4, class5, class6, class7, class8, class9

#将标签转化为01数据
def translate_labels(labels):
    translate_labels = np.mat(np.zeros((labels.shape[0],10)))
    for i in range(labels.shape[0]):
        numIndex = labels[i,0]
        numIndex = int(numIndex)
        translate_labels[i,numIndex] = 1
    return translate_labels


filename = 'E:\dataset\Train.csv'
class0, class1, class2, class3, class4, class5, class6, class7, class8, class9 = classify(filename)
train_dataset,labels_train,test_dataset,labels_test = GetData(class0, class1, class2, class3, class4, class5, class6, class7, class8, class9)
dataset_train,dataset_test = creatData_train1(2400,3600,train_dataset,test_dataset)
labels_train = translate_labels(labels_train)
labels_test = translate_labels(labels_test)

"""
def add_layer(inputs, in_size, out_size,n_layer, activationFuntion=None):
    layer_name= "layer%s"%n_layer
    with tf.name_scope("layer_name"):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]),name='weights')  # 一般定义矩阵，首字母大写
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1,name='biases')  # biase是类似列表的形式
        with tf.name_scope('wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs, Weights) + biases
        Wx_plus_b = tf.nn.dropout(Wx_plus_b,keep_prob)
        with tf.name_scope("outputs"):
            if activationFuntion==None:
                outputs = Wx_plus_b
            else:
                 outputs = activationFuntion(Wx_plus_b)
        tf.summary.histogram(layer_name+"/outputs",outputs)
"""


def computer_accuracy(v_xs,v_ys):
    global prediction
    y_pre = sess.run(prediction,feed_dict={xs:v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))#cast将bool类型转换为0，1
        tf.summary.scalar(accuracy)
    result = sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys})
    return result


def weight_variable(shape):
    with tf.name_scope('weight'):
        initial = tf.truncated_normal(shape,mean=0 ,stddev=0.1)  #生成张量的维度，mean是均值，stddev是方差
    return tf.Variable(initial)


def bias_variable(shape):
    with tf.name_scope('biases'):
        initial = tf.constant(0.1,shape = shape)   #初始化bias，为0.1的常量，维度为shape
    return tf.Variable(initial)


# 所需传参：图片数据和过滤器矩阵
def conv2d(x,w):
    # input是指做卷积的输入图像，是一个张量，具有[batch, in_height, in_width, in_channels]
    # 训练时一个batch的图片数量，图片高度，图片宽度，图片通道、
    # filter相当于CNN中的卷积核，要求是一个张量，具有filter_height, filter_width, in_channels, out_channels]
    # 含义是[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]，要求类型与参数input相同
    # strider:卷积时图像每一维的步长。一维的向量，长度为4
    # padding:string类型向量，只能时"SAME"和"VELID"中的一个
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')  #tf.nn.conv2d(input,fitle,strid,padding)

def max_pool_2x2(x):
    # x是传入量，也就是卷积神经网络第一步卷积的输出量，x的shape必须为[batch, height, width, channels]
    # batch:卷积时可能传入一张图片，也可能是n张，参数同上
    # ksize：池化窗口大小[batch, height, width, channels]
    # stride：步长，一般也是[1,stride,stride,1],参数和ksize差不多，功能不同
    # padding同上
    # padding='SAME'输出矩阵的长和宽为W/S，即之前的长和宽除以步长
    # padding = 'VALID'输出矩阵的长和宽为[(W-F+1)/S],F为过滤器的size，W为输入的size，[]向下取整
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')


# 为了神经网络的layer可以使用image数据，我们要将其转化成4d的tensor: (Number, width, height, channels)
# -1表示任意数量的样本数,大小为28x28深度为1的张量
with tf.name_scope('input'):
    xs = tf.placeholder(tf.float32, [None, 3600])  # 定义喂入数据的类型和形状
    ys = tf.placeholder(tf.float32, [None, 10])
    keep_prob = tf.placeholder(tf.float32)  # 防止过拟合
    x_image = tf.reshape(xs,[-1,60,60,1])
# conv1 layer
# 使用32个5x5x1的filter，然后通过maxpooling。
with tf.name_scope('con_layer1'):
    W_conv1 = weight_variable([5, 5, 1, 128])   # patch5*5,in size1,out size32,
    b_conv1 = bias_variable([128])
    h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1) # output size28*28*32
    h_pool1 = max_pool_2x2(h_conv1)  # output

# padding='SAME'【out/strid】向上取整14*14*32
# conv2 layer
# 使用64个5x5的filter。
with tf.name_scope('con_layer2'):
    W_conv2 = weight_variable([5,5,128,32])  # patch5*5 ,in size1,out size64
    b_conv2 = bias_variable([32])
    h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2) # output size14*14*64
    h_pool2 = max_pool_2x2(h_conv2)  # output size out/strid 7*7*64

# func1 layer
# 全连接层，将h_pool2中的所有元素平铺
with tf.name_scope('func_layer1'):
    with tf.name_scope('weights'):
        W_func1 = weight_variable([15*15*32, 128])#输入7*7*64，输出size：1024
    with tf.name_scope('biases'):
        b_func1 = bias_variable([128])
# [n_samples,7,7,64]-->[n_samples,7*7*64]
    h_pool2_flat = tf.reshape(h_pool2, [-1, 15*15*32])  # 平铺成n行，构建全连接层的输入
    h_func1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_func1)+b_func1)
    h_func1_drop = tf.nn.dropout(h_func1, keep_prob)


# func2 layer
with tf.name_scope('func_layer2'):
    W_func2 = weight_variable([128, 10])
    b_func2 = bias_variable([10])

with tf.name_scope('prediction'):
    prediction = tf.nn.softmax(tf.matmul(h_func1_drop,W_func2)+b_func2)

# 训练和评估
with tf.name_scope('loss'):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))
    tf.summary.scalar('loss', cross_entropy)
# cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=ys, logits=prediction))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
with tf.name_scope("accuracy"):
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(ys, 1), tf.argmax(prediction, 1)), tf.float32))
    tf.summary.scalar('accuracy', accuracy)
# 初始化

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    merged = tf.summary.merge_all()  # 将所有的变量放在一起
    writer = tf.summary.FileWriter("E:\Ten-graph", sess.graph)  # 存放文件的指定位置
    for i in range(250):
        # batch = mnist.train.next_batch(50)
        train_step.run(feed_dict={xs: dataset_train, ys: labels_train,keep_prob: 0.5})
        if i % 20 == 0:
            train_accuracy = accuracy.eval(feed_dict={xs: dataset_train, ys: labels_train, keep_prob: 1.})
            print('step{},the train accuracy:{}'.format(i,train_accuracy))
            result = sess.run(merged, feed_dict={xs: dataset_train, ys: labels_train, keep_prob: 1.})
            writer.add_summary(result, i)  # 调用result的add_summary方法将训练过程以及训练步数保存
    test_accuracy = accuracy.eval(feed_dict={xs: dataset_test, ys: labels_test, keep_prob: 1.})
    print('the test accuracy :{}'.format(test_accuracy))
    saver = tf.train.Saver()
    path = saver.save(sess, 'E:/dataset/test.ckpt')










