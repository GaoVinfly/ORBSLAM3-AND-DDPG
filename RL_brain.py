# coding=UTF-8
"""
This part of code is the Deep Q Network (DQN) brain.
view the tensorboard picture about this DQN structure on: https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/4-3-DQN3/#modification
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
Using:
r: r1.2
"""
import os
import numpy as np
import tensorflow as tf
from functools import reduce
from operator import mul

np.random.seed(1)  # seed用于指定随机数生成时所用算法开始的整数值
tf.set_random_seed(1)
from utils import conv2d_layer, fully_connected_layer, stateful_lstm
from base import BaseModel
from replay_memory import DRQNReplayMemory

# 通过tf.set_random_seed()函数对该图资源下的全局随机数生成种子进行设置，使得不同Session中的random系列函数表现出相对协同的特征

# Deep Q Network off-policy   1.290000 7 0.850000  1.000000 8 0.790000 1.000000 8 0.790000


class DeepQNetwork(BaseModel):
    def __init__(
            self,
            n_actions,
            n_features,              # n_features接收多少观测，预测action的值
            learning_rate=0.01,  # 学习率用来定义每次参数更新的幅度
            reward_decay=0.9,  # 奖励衰减
            e_greedy=0.9,  # 贪婪度
            replace_target_iter=400,  # 目标网络迭代次数,更新Q现实网络参数的步骤数，隔多少步更新target
            memory_size=2000,  # 内存回放单元大小  源memory_size=500
            batch_size=64,  # 批大小，#每次从记忆库中取的样本数量，神经网络学习时用到
            e_greedy_increment=None,    #不断缩小随机范围，对学习有促进作用
            output_graph=True,
            config=None
    ):
        self.n_actions = n_actions  # 输出多少个action的值
        self.n_features = n_features  ##接受多少个观测值的相关特征
        self.lr = learning_rate
        self.gamma = reward_decay  # 奖励衰减
        self.epsilon_max = e_greedy  # 贪婪度
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max  #epsilon


        # total learning step用于判断是否更换 target_net 参数
        self.learn_step_counter = 0  #记录已经学习了多少步


        self.replay_memory = DRQNReplayMemory(config)
        self.cnn_format = config.cnn_format
        self.num_lstm_layers = config.num_lstm_layers
        self.lstm_size = config.lstm_size
        self.min_history = config.min_history
        self.states_to_update = config.states_to_update

        # initialize zero memory 初始化全0记忆[s, a, r, s_],  高度是memory_size大小，长度是两个观测值，加上动作和奖励
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        self.config = config

        self.states = np.empty((self.config.mem_size, self.config.screen_height, self.config.screen_width), dtype=np.uint8)
        self.actions = np.empty((self.config.mem_size), dtype=np.int32)
        self.rewards = np.empty((self.config.mem_size), dtype=np.int32)
        self.states_ = np.empty((self.config.mem_size, self.config.screen_height, self.config.screen_width), dtype=np.uint8)
        self.count = 0
        self.current = 0
        self.obser = np.empty((self.config.batch_size, self.config.min_history + self.config.states_to_update + 1,
                                self.config.screen_height, self.config.screen_width), dtype=np.uint8)
        self.obser_ = np.empty((self.config.batch_size, self.config.min_history + self.config.states_to_update + 1,
                               self.config.screen_height, self.config.screen_width), dtype=np.uint8)
        self.actions_out = np.empty(
            (self.config.batch_size, self.config.min_history + self.config.states_to_update + 1))
        self.rewards_out = np.empty(
            (self.config.batch_size, self.config.min_history + self.config.states_to_update + 1))
        self.timesteps = np.empty((self.config.mem_size), dtype=np.int32)


        # consist of [target_net, evaluate_net]
        self._build_net()  # 调用_build_net()函数

        # # tf.get_collection() 主要作用：从一个集合中取出变量
        # # 返回target_net名称域中所有放入‘GLOBAL_VARIABLES’变量的列表
        # # t提取 target_net 的参数
        # t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        # # 提取  eval_net 的参数
        # # 返回eval_net名称域中所有放入‘GLOBAL_VARIABLES’变量的列表
        # e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

        #  tf.variable_scope下,对创建的变量均增加作用域
        with tf.variable_scope('hard_replacement'):
            # tf.get_collection() 主要作用：从一个集合中取出变量
            # 返回target_net名称域中所有放入‘GLOBAL_VARIABLES’变量的列表
            # t提取 target_net 的参数
            t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
            # 提取  eval_net 的参数
            # 返回eval_net名称域中所有放入‘GLOBAL_VARIABLES’变量的列表
            e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

            # zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
            self.target_replace_op = [tf.assign(t, e) for t, e in
                                      zip(t_params, e_params)]  # assign把e赋值给t ,更新 target_net 参数

        # 在会话中启动图
        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            tf.summary.FileWriter("logs/", self.sess.graph)  # 指定一个文件和路径用来保存图

        self.cost_his = []  # 记录所有 cost 变化, 用于最后 plot 出来观看
        if bool(1 - output_graph):
            self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.init_update()
        self.get_num_params()
    # ''' 所以placeholder()函数是在神经网络构建graph的时候在模型中的占位，此时并没有把要输入的数据传入模型，
    # 它只会分配必要的内存。等建立session，在会话中，运行模型的时候通过feed_dict()函数向占位符喂入数据'''
    def _build_net(self):  # 函数实现 构建网络

        # ------------------ all inputs ------------------------
        self.w = {}
        self.w_target = {}
        ### need two inputs:
        ####### input iamges :
        self.state = tf.placeholder(tf.float32, [None, 1, 84, 84], name='state')  # input State
        ############### input's state:
        # self.init_state = tf.placeholder(tf.float32, [None, self.lstm_size], name='init_state')

        # tf.placeholder(dtype,shape=None,name=None)
        #dtype：数据类型。常用的是tf.float32,tf.float64等数值类型
        #shape：数据形状。默认是None，就是一维值，也可以是多维（比如[2,3], [None, 3]表示列是3，行不定）
        #name：名称
        self.init_state = tf.placeholder(tf.float32, [None,512], name='init_state')
        self.state_ = tf.placeholder(tf.float32, [None, 1, 84, 84], name='state_')  # input Next State
        self.r = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward
        self.a = tf.placeholder(tf.int32, [None, ], name='a')  # input Action

        # create placeholder to fill in lstm state
        # self.c_state_train = tf.placeholder(tf.float32, [None, self.lstm_size], name="train_c")
        # self.h_state_train = tf.placeholder(tf.float32, [None, self.lstm_size], name="train_h")
        #LSTMStateTuple是一种特殊的 "二元组数据类型" ,它专门用来存储LSTM单元的state_size/zero_state/output_sta
        # self.lstm_state_train = tf.nn.rnn_cell.LSTMStateTuple(self.c_state_train, self.h_state_train)
        #暂时注释
        self.c_state_target = tf.placeholder(tf.float32, [None, self.lstm_size], name="target_c")
        self.h_state_target = tf.placeholder(tf.float32, [None, self.lstm_size], name="target_h")


        self.lstm_state_target = tf.nn.rnn_cell.LSTMStateTuple(self.c_state_target, self.h_state_target)


        # initial zero state to be used when starting episode
        self.initial_zero_state_batch = np.zeros((self.config.batch_size, self.lstm_size))
        self.initial_zero_state_single = np.zeros((1, self.lstm_size))
        self.initial_zero_complete = np.zeros((self.num_lstm_layers, 2, self.config.batch_size, self.lstm_size))

        # random_normal_initializer返回一个生成具有正态分布的张量的初始化器。
        # w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

        # ------------------构建评估网络eval_net ------------------
        # kernel_initialize为卷积核的初始化器,bias_initializer偏置项的初始化器，默认初始化为0,name为层的名字
        # 输出的维度大小为20,输入为当前状态s
        # dense(全连接层)的作用:通常在CNN的尾部进行重新拟合，减少特征信息的损失
        # eval_net用于预测q_eval, 拥有最新的神经网络参数
        with tf.variable_scope('eval_net'):
            self.cnn_format = "NHWC"                                #cnn_format = 'NHWC' 是配置好的，指定了卷积神经网络的一种输入输出格式。跟‘NHWC’只是在输入参数的顺序上有所不同
            if self.cnn_format == "NHWC":
                self.s = tf.transpose(self.state, [0, 2, 3, 1])
            self.image_summary = []
            # 卷积层1
            w, b, out, summary = conv2d_layer(self.s, 32, [8, 8], [4, 4], scope_name="conv1_train",
                                              summary_tag="conv1_out",
                                              activation=tf.nn.relu, data_format=self.cnn_format)   #tf.nn.relu
            self.w["wc1"] = w
            self.w["bc1"] = b
            self.image_summary.append(summary)
            # 卷积层2
            w, b, out, summary = conv2d_layer(out, 64, [4, 4], [2, 2], scope_name="conv2_train",
                                              summary_tag="conv2_out",
                                              activation=tf.nn.relu, data_format=self.cnn_format)
            self.w["wc2"] = w
            self.w["bc2"] = b
            self.image_summary.append(summary)
            # 卷积层3
            w, b, out, summary = conv2d_layer(out, 64, [3, 3], [1, 1], scope_name="conv3_train",
                                              summary_tag="conv3_out",
                                              activation=tf.nn.relu, data_format=self.cnn_format)
            self.w["wc3"] = w
            self.w["bc3"] = b
            self.image_summary.append(summary)

            shape = out.get_shape().as_list()   #get_shape获取out张量返回的是一个元组，as_list():将元组转换为列表，不是元组将会报错
            # 将第三层卷积层的输出转换为
            out_flat = tf.reshape(out, [tf.shape(out)[0], 1, shape[1] * shape[2] * shape[3]])
            # LSTM层
            # print("###############",out_flat.shape[0])
            out, state = stateful_lstm(out_flat, self.num_lstm_layers, self.lstm_size, self.init_state,scope_name="lstm_train")
            self.state_output_c = state[0][0]
            # self.state_output_h = state[0][1]
            shape = out.get_shape().as_list()
            out = tf.reshape(out, [tf.shape(out)[0], shape[2]])
            # 全连接层
            w, b, out = fully_connected_layer(out, self.n_actions, scope_name="out_train", activation=None)
            self.w["wout"] = w
            self.w["bout"] = b
            self.q_eval = out  # shape=10500

            with tf.variable_scope('q_eval'):
                # 计算Q预测
                a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
                self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)  # shape=(None, )

        # ------------------ 构建目标网络target_net ------------------
        # target_net用于预测q_target值, 不会及时更新参数
        # 输入为下一帧状态
        with tf.variable_scope('target_net'):
            if self.cnn_format == "NHWC":
                self.s_ = tf.transpose(self.state_, [0, 2, 3, 1])
            w, b, out, _ = conv2d_layer(self.s_, 32, [8, 8], [4, 4], scope_name="conv1_target", summary_tag=None,
                                        activation=tf.nn.relu, data_format=self.cnn_format)
            self.w_target["wc1"] = w
            self.w_target["bc1"] = b

            w, b, out, _ = conv2d_layer(out, 64, [4, 4], [2, 2], scope_name="conv2_target", summary_tag=None,
                                        activation=tf.nn.relu, data_format=self.cnn_format)
            self.w_target["wc2"] = w
            self.w_target["bc2"] = b

            w, b, out, _ = conv2d_layer(out, 64, [3, 3], [1, 1], scope_name="conv3_target", summary_tag=None,
                                        activation=tf.nn.relu, data_format=self.cnn_format)
            self.w_target["wc3"] = w
            self.w_target["bc3"] = b

            shape = out.get_shape().as_list()
            out_flat = tf.reshape(out, [tf.shape(out)[0], 1, shape[1] * shape[2] * shape[3]])
            out, state = stateful_lstm(out_flat, self.num_lstm_layers, self.lstm_size, self.init_state,scope_name="lstm_target")
            self.state_output_target_c = state[0][0]
            # self.state_output_target_h = state[0][1]
            shape = out.get_shape().as_list()
            out = tf.reshape(out, [tf.shape(out)[0], shape[2]])
            # 全连接层
            w, b, out = fully_connected_layer(out, self.n_actions, scope_name="out_target", activation=None)
            self.w_target["wout"] = w
            self.w_target["bout"] = b
            self.q_next = out

            with tf.variable_scope('q_target'):  # 用于定义创建变量（层）的操作的上下文管理器
                # 计算Q目标
                q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')  # shape=(None, )
                # q_target = self.r
                self.q_target = tf.stop_gradient(q_target)  # 方便求loss，阻止梯度传递

        # reduce_max()计算张量各个维度的最大值，q_next为要减小的张量，
        # stop_gradient来对从loss到targetnet的反传进行截断，换句话说，通过self.q_target = tf.stop_gradient(q_target)
        # ，将原本为TensorFlow计算图中的一个op（节点）转为一个常量self.q_target，这时候对于loss的求导反传就不会传到target net去了

        #转移到‘target_net’最后
        # with tf.variable_scope('q_target'):  # 用于定义创建变量（层）的操作的上下文管理器
        #     #计算Q目标
        #     q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')  # shape=(None, )
        #     # q_target = self.r
        #     self.q_target = tf.stop_gradient(q_target)  # 方便求loss，阻止梯度传递
        # tf.stack()函数主要是用来提升维度
        # 将params中的切片收集到一个由指标indices指定形状的张量中

        #转移到‘eval_net’最后
        # with tf.variable_scope('q_eval'):
        #     #计算Q预测
        #     a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
        #     self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)  # shape=(None, )

            # tf.reduce_mean函数用于计算张量tensor沿着指定的数轴（tensor的某一维度）上的的平均值，主要用作降维或者计算tensor（图像）的平均值.
            # tf.squared_difference 计算张量q_target、q_eval_wrt_a对应元素差平方.
        with tf.variable_scope('loss'):
            #计算损失Loss
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))
            self.merged_image_sum = tf.summary.merge(self.image_summary, "images")

        #  采用RMSPropOptimizer优化器对损失值优化
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)


    #state状态，a动作，r奖励，state_接下来动作，t终端判断是不是最后一个episode(回合)



    def store_transition(self, state, a, r, state_,t):  # 存储状态信息
        if not hasattr(self, 'memory_counter'):  # hasattr() 函数用于判断对象是否包含对应的属性。
            self.memory_counter = 0
        self.states[self.current] = state
        self.actions[self.current] = a
        self.rewards[self.current] = r
        self.states_[self.current] = state_
        self.count = max(self.count, self.current + 1)
        self.timesteps[self.current] = t
        self.current = (self.current + 1) % self.config.mem_size

    def test_action(self, observation):
        image = observation
        self.lstm_state_c, self.lstm_state_h = self.initial_zero_state_single, self.initial_zero_state_single
        if np.random.uniform() < self.epsilon:  # rand()通过本函数可以返回一个或一组服从“0~1”均匀分布的随机样本值。随机样本取值范围是[0,1)，不包括1
            actions_value, self.lstm_state_c, self.lstm_state_h = self.sess.run(
                [self.q_eval, self.state_output_c, self.state_output_h], feed_dict={
                    self.state: [[image]],
                    # self.c_state_train: self.lstm_state_c,
                    # self.h_state_train: self.lstm_state_h
                })
            action = np.argmax(actions_value)
            return action
        else:
            action = np.random.randint(0, self.n_actions)
            self.lstm_state_c, self.lstm_state_h = self.sess.run(
                [self.state_output_c, self.state_output_h],
                {
                    self.state: [[image]],
                    # self.c_state_train: self.lstm_state_c,
                    # self.h_state_train: self.lstm_state_h
                })
            return action


    def choose_action(self, observation):
        # type:(object) -> object
        image = observation
        ######### add single init state :
        self.initial_zero_state_single = np.zeros((1, self.lstm_size))
        if np.random.uniform() < self.epsilon:  # rand()通过本函数可以返回一个或一组服从“0~1”均匀分布的随机样本值。随机样本取值范围是[0,1)，不包括1
            actions_value,self.initial_zero_state_single= self.sess.run([self.q_eval, self.state_output_c],
                                                                        feed_dict={ self.state: [[image]],
                                                                        self.init_state: self.initial_zero_state_single,
                                                                        # self.h_state_train: self.lstm_state_h
                                                                        })
            action = np.argmax(actions_value)
            return action
        else:
            action = np.random.randint(0, self.n_actions)
            self.initial_zero_state_single = self.sess.run(
                [self.state_output_c],
                {
                    self.state: [[image]],
                    ######### add seconde inputs of state:
                    self.init_state: self.initial_zero_state_single
                    # self.c_state_train: self.lstm_state_c,
                    # self.h_state_train: self.lstm_state_h
                })
            return action

    def init_update(self):
        self.target_w_in = {}
        self.target_w_assign = {}
        for name in self.w:
            self.target_w_in[name] = tf.placeholder(tf.float32, self.w_target[name].get_shape().as_list(), name=name)
            self.target_w_assign[name] = self.w_target[name].assign(self.target_w_in[name])
        # self.lstm_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        # lstm_target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self.lstm_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='eval_net/lstm_train')   #4
        lstm_target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_net/lstm_target')

        for i, var in enumerate(self.lstm_vars):
            self.target_w_in[var.name] = tf.placeholder(tf.float32, var.get_shape().as_list())
            self.target_w_assign[var.name] = lstm_target_vars[i].assign(self.target_w_in[var.name])

    def learn(self):  # 強化学习过程
        # check to replace target parameters 检查是否替换target_net参数
        if self.learn_step_counter % self.replace_target_iter == 0:  # replace_target_iter=300
            self.sess.run(self.target_replace_op)
            print('\ntarget_params_replaced\n')

        img, act, re, img_ = self.sample_batch()
        imgg = img / 255.0
        imgg_ = img / 255.0
        imgg = np.transpose(imgg, [1, 0, 2, 3])
        imgg_ = np.transpose(imgg_, [1, 0, 2, 3])
        act = np.transpose(act, [1, 0])
        re = np.transpose(re, [1, 0])
        imgg = np.reshape(imgg, [imgg.shape[0], imgg.shape[1], 1, imgg.shape[2], imgg.shape[3]])
        imgg_ = np.reshape(imgg_, [imgg_.shape[0], imgg_.shape[1], 1, imgg_.shape[2], imgg_.shape[3]])

        ###### add batch init_batch:
        self.initial_zero_state_batch = np.zeros((self.batch_size, self.lstm_size))
        # cell = tf.nn.rnn_cell.GRUCell(self.lstm_size)
        # self.initial_zero_state_batch = cell.zero_state(self.batch_size, dtype=tf.float32)



        # lstm_state_c, lstm_state_h = self.initial_zero_state_batch, self.initial_zero_state_batch
        # lstm_state_target_c, lstm_state_target_h = self.sess.run(
        #     [self.state_output_target_c, self.state_output_target_h], feed_dict={
        #         self.state_: imgg[0],
        #         # self.c_state_target: self.initial_zero_state_batch,
        #         # self.h_state_target: self.initial_zero_state_batch
        #     }
        # )
        for i in range(self.min_history, self.min_history + self.states_to_update):
            _, cost, merged_imgs = self.sess.run(
                [self._train_op, self.loss, self.merged_image_sum],
                feed_dict={
                    self.state: imgg[i],
                    self.a: act[i],
                    self.r: re[i],
                    self.state_: imgg_[i],
                    ##### when learning :
                    self.init_state: self.initial_zero_state_batch
                    # self.c_state_train: lstm_state_c,
                    # self.h_state_train: lstm_state_h,
                    # self.c_state_target: lstm_state_target_c,
                    # self.h_state_target: lstm_state_target_h,
                }
            )
        self.cost_his.append(cost)  # 记录 cost 误差

        # increasing epsilon# 逐渐增加 epsilon, 降低行为的随机性
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def sample_batch(self):
        assert self.count > self.config.min_history + self.config.states_to_update   # min_history=4,states_to_update=4
        indices = []
        while len(indices) < self.config.batch_size:    #64
            while True:
                index = np.random.randint(self.config.min_history, self.count)  #返回随机整数或整型数组，范围区间为(4,16）
                if index >= self.current and index - self.config.min_history < self.current:
                    continue
                if index < self.config.min_history + self.config.states_to_update + 1:
                    continue
                if self.timesteps[index] < self.config.min_history + self.config.states_to_update:
                    continue
                break
            self.obser[len(indices)], self.obser_[len(indices)] = self.getState(index)
            self.actions_out[len(indices)], self.rewards_out[len(indices)] = self.get_scalars(index)
            indices.append(index)
        return self.obser, self.actions_out, self.rewards_out, self.obser_

    def getState(self, index):
        im = self.states[index - (self.config.min_history + self.config.states_to_update + 1): index]
        im_ = self.states_[index - (self.config.min_history + self.config.states_to_update + 1): index]
        return im, im_
    def get_scalars(self, index):
        a = self.actions[index - (self.config.min_history + self.config.states_to_update + 1): index]
        r = self.rewards[index - (self.config.min_history + self.config.states_to_update + 1): index]
        return a, r

    def update_target(self):
        for name in self.w:
            self.target_w_assign[name].eval({self.target_w_in[name]: self.w[name].eval(session=self.sess)},
                                            session=self.sess)
        for var in self.lstm_vars:
            self.target_w_assign[var.name].eval({self.target_w_in[var.name]: var.eval(session=self.sess)},
                                                session=self.sess)

    def get_num_params(self,):
        num_params = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            num_params += reduce(mul, [dim.value for dim in shape], 1)
        print("---------------model_parameters---------------------:")
        print(num_params)

    def plot_cost(self,episodes):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training episodes')
        plt.legend()
        # save the figure
        if episodes == 800:
            plt.savefig("cost1.png")
        elif episodes == 1600:
            plt.savefig("cost2.png")
        elif episodes == 2400:
            plt.savefig("cost3.png")
        elif episodes == 3200:
            plt.savefig("cost4.png")
        # plt.show()


    #保存模型
    def save_model(self, step=1, imgname=None):
        # model_save_path = './Model_m1/'+imgname+'/'
        model_save_path = 'Model/'
        model_name = 'model'
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        savers = tf.train.Saver()
        savers.save(self.sess, os.path.join(model_save_path, model_name))
    #加载模型
    def load_model(self, imgname=None):
        # model_save_path = './Model_m1/'+imgname+'/'
        last_model_path = 'Model/'
        model_name = 'model'
        savers = tf.train.Saver()
        savers.restore(self.sess, os.path.join(last_model_path, model_name))

    # def save_model(self,step=1,imgname=None):
    #     model_save_path = './Model_m1/'
    #     model_name = 'model_self'
    #     if not os.path.exists(model_save_path):
    #         os.makedirs(model_save_path)
    #     savers = tf.train.Saver()
    #     savers.save(self.sess, os.path.join(model_save_path, model_name))
    #
    # def load_model(self,imgname=None):
    #     # last_model_path = './Model_m1/'
    #     # model_name = 'model_self'
    #     # savers = tf.train.Saver()
    #     # savers.restore(self.sess, os.path.join(last_model_path, model_name))
    #
    #     # saver = tf.train.import_meta_graph('./Model_m1/model_self.meta')
    #     # saver.restore(self.sess, tf.train.latest_checkpoint('./Model_m1'))
    #     saver = tf.train.Saver()
    #     saver.restore(self.sess, './Model_m1/model_self')

