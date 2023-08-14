#2021年5月20日14:14:15
#LSTM预测手写数字MNIST数据集
#%tensorflow_version 1.x
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
#tf.reset_default_graph()
"""加载数据"""
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

"""参数设置"""
BATCH_SIZE = 128        # BATCH的大小，相当于一次处理128个image
TIME_STEP = 28         # 一个LSTM中，输入序列的长度，image有28行
INPUT_SIZE = 28         # x_i 的向量长度，image有28列
LR = 0.001           # 学习率
NUM_UNITS = 128         # LSTM的输出维度
ITERATIONS = 10000         # 迭代次数
N_CLASSES = 10            # 输出大小，0-9十个数字的概率
"""定义计算"""
# 定义 placeholders 以便接收x,y
# 维度是[BATCH_SIZE，TIME_STEP * INPUT_SIZE]
train_x = tf.placeholder(tf.float32, [BATCH_SIZE, TIME_STEP * INPUT_SIZE])

# 输入的是二维数据，将其还原为三维，维度是[BATCH_SIZE, TIME_STEP, INPUT_SIZE]
image = tf.reshape(train_x, [BATCH_SIZE, TIME_STEP, INPUT_SIZE])
train_y = tf.placeholder(tf.int32, [BATCH_SIZE, N_CLASSES])

# 定义网络结构两层lSTM
rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=NUM_UNITS*2)
rnn_cell_ = tf.contrib.rnn.BasicLSTMCell(num_units=NUM_UNITS)
multi_cell = tf.contrib.rnn.MultiRNNCell([rnn_cell, rnn_cell_],state_is_tuple=True)
outputs, final_state = tf.nn.dynamic_rnn(
    cell=multi_cell,              # 选择传入的cell
    inputs=image,               # 传入的数据
    initial_state=None,         # 初始状态
    dtype=tf.float32,           # 数据类型
    # False: (batch, time_step, x_input); True: (time_step,batch,x_input)，
    # 这里根据image结构选择False
    # If false, these `Tensors` must be shaped `[batch_size, max_time, depth]`.
    time_major=False,
)

"""获取输出"""
output = tf.layers.dense(
    inputs=outputs[:, -1, :], units=N_CLASSES)  # 取最后一路输出送入全连接层

"""定义损失和优化方法"""
loss = tf.losses.softmax_cross_entropy(
    onehot_labels=train_y,
    logits=output)      # 计算loss

train_op = tf.train.AdamOptimizer(LR).minimize(loss)  # 选择优化方法

correct_prediction = tf.equal(
    tf.argmax(
        train_y, axis=1), tf.argmax(
            output, axis=1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))  # 计算正确率
saver = tf.train.Saver() #保存模型
"""summary"""
tf.summary.scalar('loss_train',loss)
tf.summary.scalar('loss_val',loss)
tf.summary.scalar('acc',accuracy)
merge_summary_train = tf.summary.merge([tf.get_collection(tf.GraphKeys.SUMMARIES,'loss_train')])
merge_summary_val = tf.summary.merge([tf.get_collection(tf.GraphKeys.SUMMARIES,'loss_val'),
                                      tf.get_collection(tf.GraphKeys.SUMMARIES,'acc')])

"""训练"""
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())     # 初始化计算图中的变量
    train_writer = tf.summary.FileWriter('/content/drive/MyDrive/LSTM_MNIST_multi/train',sess.graph)
    val_writer = tf.summary.FileWriter('/content/drive/MyDrive/LSTM_MNIST_multi/val',sess.graph)
    for step in range(1, ITERATIONS):    # 开始训练
        x, y = mnist.train.next_batch(BATCH_SIZE)
        _, loss_ = sess.run([train_op, loss], {train_x: x, train_y: y})
        train_summary = sess.run(merge_summary_train,
                     feed_dict={train_x: x, train_y: y})
        train_writer.add_summary(train_summary,step)
        if step % 100 == 0:      # test（validation）
            saver.save(sess, "Model/LSTM_MNIST_" + str(step))
            test_x, test_y = mnist.test.next_batch(BATCH_SIZE)
            accuracy_ = sess.run(accuracy, {train_x: test_x, train_y: test_y})
            val_summary = sess.run(merge_summary_val,
                        {train_x: x, train_y: y})
            val_writer.add_summary(val_summary,step)
            print(
                'train loss: %f' %
                loss_,
                '| validation accuracy: %f' %
                accuracy_)
    train_writer.close()
    val_writer.close()
