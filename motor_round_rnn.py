import pandas as pd
import numpy as np
import random
import tensorflow as tf

def get_traindata():
    csv_data = pd.read_csv('train.csv')
    col = csv_data.iloc[:, 6001]
    y_train = col.values
    y_traindata = np.array(y_train,np.float32)
    y_traindata = y_traindata.reshape(-1, 1)

    row = csv_data.iloc[0:792, 1:6001]
    x_train = row.values
    x_train = np.array(x_train, np.float32)

    return x_train, y_traindata

def get_testdata():
    csv_data = pd.read_csv('Test_data.csv')
    row = csv_data.iloc[0:528, 1:6001]
    x_test = row.values
    x_test = np.array(x_test, np.float32)

    return x_test

def data_handle_test(x):
    x = x.tolist()
    x_train = []
    for i in range(len(x)):
        a = x[i].index(max(x[i][0:2399]))
        b = x[i][a:3600+a]
        x_train.append(b)
    x_test_data = np.array(x_train, np.float32)

    return x_test_data

def data_handle(x,y):
    x = x.tolist()
    y = y.tolist()
    x_train = []
    y_train=[]
    for i in range(len(x)):
        a = x[i].index(max(x[i][0:2399]))
        b = x[i][a:3600+a]
        x_train.append(b)
        y_train.append(y[i][0])
    x_train_data = np.array(x_train, np.float32)
    y_train_data = np.array(y_train, np.float32)
    y_train_data = y_train_data.reshape(-1,1)

    return x_train_data,y_train_data

def chose_data(x_train, y_traindata, size):

    resultList = random.sample(range(0, len(x_train)), size)

    x_train_train = np.array([x_train[i] for i in resultList])

    y_traindata_train = np.array([y_traindata[i] for i in resultList])

    return x_train_train, y_traindata_train

def deal_y_data(y_train_data):
    y_train_data = y_train_data.tolist()
    y_train = []
    for i in range(len(y_train_data)):
        a = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        a[int(y_train_data[i][0])] = 1
        y_train.append(a)
    y_traindata = np.array(y_train, np.float32)
    return y_traindata


lr =0.001
training_iters =15000
n_inputs= 60
n_steps = 60 #time steps
n_hidden_units =[128,256,128,64] #
n_class =10
num_layers = 3

weights =  {'in': tf.Variable(tf.random_normal([n_inputs,n_hidden_units[0]])),
            'out': tf.Variable(tf.random_normal([n_hidden_units[3], n_class]))}

biases = {'in': tf.Variable(tf.constant(0.1,shape = [n_hidden_units[0],])),
         'out': tf.Variable(tf.constant(0.1,shape =[n_class,]))}

with tf.variable_scope('input'):
    x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
    y = tf.placeholder(tf.float32, [None, n_class])
    batch_size = tf.placeholder(tf.int32, [])

with tf.variable_scope('layer_1'):
    monolayer_11 = tf.nn.rnn_cell.LSTMCell(num_units=n_hidden_units[0],
                                               forget_bias=1., state_is_tuple=True, activation=tf.tanh)
    monolayer_1 = tf.nn.rnn_cell.DropoutWrapper(cell=monolayer_11, output_keep_prob=0.8)

with tf.variable_scope('layer_2'):
    monolayer_21 = tf.nn.rnn_cell.LSTMCell(num_units=n_hidden_units[1],
                                               forget_bias=1., state_is_tuple=True, activation=tf.tanh)
    monolayer_2 = tf.nn.rnn_cell.DropoutWrapper(cell=monolayer_21, output_keep_prob=0.8)

with tf.variable_scope('layer_3'):
    monolayer_31 = tf.nn.rnn_cell.LSTMCell(num_units=n_hidden_units[2],
                                               forget_bias=1., state_is_tuple=True, activation=tf.tanh)
    monolayer_3 = tf.nn.rnn_cell.DropoutWrapper(cell=monolayer_31, output_keep_prob=0.8)

with tf.variable_scope('layer_Final'):
    monolayer_final = tf.nn.rnn_cell.LSTMCell(num_units=n_hidden_units[3],
                                                   forget_bias=1., state_is_tuple=True, activation=tf.tanh)

with tf.variable_scope('Layer_Structure_Combination'):
    Layers = tf.nn.rnn_cell.MultiRNNCell(cells=[monolayer_1, monolayer_2, monolayer_3, monolayer_final],
                                         state_is_tuple=True)
    init_state = Layers.zero_state(batch_size, tf.float32)

with tf.variable_scope('outpout'):
    X = tf.reshape(x, [-1, n_inputs])
    X_in = tf.matmul(X, weights['in']) + biases['in']
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units[0]])
    output,states = tf.nn.dynamic_rnn(Layers,X_in,initial_state=init_state,time_major=False)
    result = tf.matmul(output[:, -1, :],weights['out'])+biases['out']

with tf.variable_scope('train'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=result,labels= y))
    train_op = tf.train.AdamOptimizer(lr).minimize(cost)
    tf.summary.scalar('cost', cost)

with tf.variable_scope('correct'):
    correct = tf.equal(tf.argmax(result, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, dtype=tf.float32))
    tf.summary.scalar('accuracy', accuracy)

init = tf.global_variables_initializer()
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("motor_board_rnn03", sess.graph)
    step = 0
    batch_size01 = 32
    x_train, y_train = get_traindata()
    x_train_data, y_train_data = data_handle(x_train, y_train)
    y_train_data = deal_y_data(y_train_data)
    while step*batch_size01 < training_iters:
        x_t, y_t= chose_data(x_train_data, y_train_data,  batch_size01)
        x_t = x_t.reshape([batch_size01, n_steps, n_inputs])
        sess.run(train_op, feed_dict={x: x_t, y: y_t,batch_size: batch_size01})
        if step % 2 == 0:
            rs = sess.run(merged, feed_dict={x: x_t, y: y_t,batch_size: batch_size01})
            writer.add_summary(rs, step)
        step = step + 1

    saver.save(sess, "motor_board_rnn03/motor_board_rnn03.ckpt")


'''
    x_test_data01 = get_testdata()
    x_test_data = data_handle_test(x_test_data01)
    x_testdata = x_test_data.reshape([len(x_test_data), n_steps, n_inputs])
    y_testdata =sess.run(tf.argmax(result,1), feed_dict={x: x_testdata,batch_size: len(x_testdata)})
    x1 = y_testdata.tolist()
    id = np.arange(1, 529, 1)
    id = id.tolist()
    dataframe = pd.DataFrame({'id': id, 'label': x1})  # 创建字典，‘id’和‘label’为第一行的名称，传入的为字典
    dataframe.to_csv("test03.csv", index=False, sep=',')


[9 7 9 0 1 7 4 7 0 3 9 0 7 0 0 7 8 9 3 0 7 2 7 5 8 9 4 0 1 8 3 7 1 0 9 0 1
 4 6 9 9 3 0 7 9 8 5 9 7 8 1 7 0 7 0 2 9 9 2 5 5 7 3 6 8 6 6 9 1 5 9 1 0 3
 7 9 8 0 0 8 7 2 0 0 1 2 7 4 9 0 7 0 6 7 5 8 0 7 9 0 0 9 6 3 9 3 5 0 2 7 8
 7 0 9 4 9 9 1 0 9 9 2 7 7 7 2 2 7 0 4 2 3 9 7 3 8 0 0 3 0 7 5 4 9 5 0 9 7
 2 4 2 9 7 0 7 2 7 0 4 2 1 0 0 3 0 7 7 6 5 1 9 0 1 4 0 7 3 9 4 8 7 9 1 2 7
 9 8 0 9 9 4 7 0 7 7 5 0 7 9 1 9 0 7 6 7 7 5 0 7 8 9 4 0 9 0 9 4 1 0 9 9 4
 0 7 8 9 6 9 1 3 7 3 9 0 3 3 0 7 7 0 0 3 5 8 8 9 0 2 9 2 0 6 0 9 7 6 7 0 7
 7 4 0 7 0 3 9 3 9 1 2 9 7 7 6 0 9 2 5 2 7 0 5 9 7 9 9 9 3 9 7 0 2 6 4 0 0
 9 9 7 7 7 4 1 3 0 8 0 7 7 8 0 7 2 7 5 9 8 7 6 4 2 8 0 6 7 9 6 6 9 0 0 8 0
 7 7 0 0 0 9 0 0 9 2 7 6 3 1 4 1 7 5 0 6 2 1 9 5 6 2 1 2 9 7 7 9 7 7 7 5 9
 2 9 7 9 2 0 5 2 6 3 0 9 4 7 5 3 5 8 9 6 0 8 5 7 0 1 4 5 8 9 9 8 3 5 7 0 0
 3 9 8 0 9 6 0 2 0 7 4 8 0 9 0 0 1 0 5 9 7 3 8 0 9 7 3 3 9 9 7 6 9 9 7 9 7
 5 0 7 6 8 9 5 0 0 8 7 0 1 8 7 2 8 0 0 1 0 9 4 4 6 7 0 1 4 4 0 5 9 3 7 9 9
 7 9 9 0 5 2 7 7 7 0 5 2 7 9 7 5 5 2 7 9 2 9 8 7 1 0 8 7 4 9 6 9 2 0 3 9 1
 4 0 7 1 9 9 3 6 2 0]

'''