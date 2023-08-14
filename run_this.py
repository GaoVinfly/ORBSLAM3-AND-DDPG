# coding=UTF-8

import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from maze_env import Maze
from maze_env import MAZE_H, MAZE_W
from RL_brain11 import DeepQNetwork
import numpy as np
import tensorflow as tf
from PIL import Image
import argparse
from config import GymConfig
import time
from rigid_transform_3D import rigid_transform_3D

logging.basicConfig(filename='./log_run.txt', level=logging.INFO,
                    filemode='w',
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')


#换数据集要修改
#MH01
# ground = np.array([[4.687579000000000384e+00,-1.786059000000000063e+00,8.035400000000000320e-01]])
# estimate = np.array([[-0.013096054,0.148018956,0.051851347]])
#MH02
# ground = np.array([[4.620760999999999896e+00, -1.836673999999999918e+00, 7.462400000000000144e-01]])
# estimate = np.array([[0.079394445, -0.285317361, -0.085732698]])
#MH03
# ground = np.array([[4.637800000000000367e+00, -1.784734000000000043e+00, 5.946029999999999927e-01]])
# estimate = np.array([[0.004397152, -0.022089170, -0.012676725]])

#MH04
# ground = np.array([[4.681549000000000404e+00, -1.751622999999999930e+00, 6.035420000000000229e-01]])
# estimate = np.array([[-0.005734304, -0.027349519, -0.008012845]])
#MH05
# ground = np.array([[4.476938999999999780e+00, -1.649105999999999961e+00, 8.731020000000000447e-01]])
# estimate = np.array([[-0.002883305, -0.031148085, -0.005559548]])

#V101
ground = np.array([[9.528940000000000188e-01, 2.176038000000000139e+00, 1.076596000000000108e+00]])
estimate = np.array([[-0.008066206, -0.070810869, 0.011605747]])

#V102  5.153419999999999668e-01 1.996723000000000026e+00 9.710769999999999680e-01    -0.012740707 -0.042944785 -0.002622362
# ground = np.array([[6.842449999999999921e-01, 2.080815999999999999e+00, 1.268488999999999978e+00]])
# estimate = np.array([[-0.011741405, -0.014601516, 0.006989324]])

#V103
# ground = np.array([[8.840989999999999682e-01, 2.051204999999999945e+00, 1.011306999999999956e+00]])
# estimate = np.array([[-0.011562885, -0.024513992, -0.016537875]])

#V201   -1.067801999999999918e+00 4.953440000000000065e-01 1.372573999999999961e+00      0.000958465 -0.015630638 -0.006562878
# ground = np.array([[-1.067801999999999918e+00, 4.953440000000000065e-01, 1.372573999999999961e+00]])
# estimate = np.array([[0.000958465, -0.015630638, -0.006562878]])

#V202
# ground = np.array([[-1.004812999999999956e+00, 4.789249999999999896e-01, 1.331321999999999894e+00]])
# estimate = np.array([[0.000083575, -0.006027541, -0.002623373]])

#V203   -1.047128000000000059e+00 4.339850000000000096e-01 1.362441000000000013e+00
# ground = np.array([[-1.047128000000000059e+00, 4.339850000000000096e-01, 1.362441000000000013e+00]])
# estimate = np.array([[0.003970685, -0.011897060, -0.007381093]])

# ground = np.empty(shape=[0, 3])  # n行3列
# estimate = np.empty(shape=[0, 3])
saver = ''
#若没有finally, try里代码出异常时，程序就会报错停止，f.close()就不会被执行，但这样f的这个文件对象会白白占用资源。而finally 就保证了文件的关闭。
def get_observation(done=False):
    global num_frames
    while True:
        num = 0
        try:
            f = open('/home/gavin/ORB_SLAM3-master/result.txt', 'r')
            line = f.readline()  #该方法每次读出一行内容，返回一个字符串
            line = line.strip()  #str.strip()就是把字符串(str)的头和尾的空格，以及位于头尾的\n \t之类给删掉。
            fields = line.split(' ') #拆分字符串。通过指定分隔符对字符串进行切片，并返回分割后的字符串列表（list）
            if len(fields[0]) > 0:
                num = int(fields[0])
        finally:
            if f:
                f.close()
        if num > num_frames:
            num_frames = num
            print('new frame:', num, num_frames)
            break
        # print num, num_frames, 'wait for a new frame'
        time.sleep(0.001)
    # 换数据集要修改
    # im = Image.open('/home/gavin/ORB_SLAM3-master/data/dataset-corridor5_512_16/mav0/cam0/data/' + fields[1] + '.png')
    # im = Image.open('/home/gavin/ORB_SLAM3-master/data/dataset-room6_512_16/mav0/cam0/data/' + fields[1] + '.png')
    im = Image.open('/home/gavin/ORB_SLAM3-master/data/V101/mav0/cam0/data/' + fields[1] + '.png')
    # im = Image.open('/home/gavin/ORB_SLAM3-master/test_data/rgb/mav0/cam0/data/' + fields[1] + '.png')

    global imgname
    imgname = fields[1]
    im = np.array(im.resize((MAZE_H, MAZE_W)))  # resize为输出图像尺寸大小，np.array将数据转化为矩阵
    # observation = im.flatten()  #flatten()函数可以执行展平操作，将二维数组转化为一维数组。
    # im = im/255
    # reward = 1 - float(fields[2])
    errs = getError(fields, done)
    reward = 1 - errs
    return im, reward

def getError(fields,done):
    global ground, ground_temp, estimate_temp
    global estimate
    global true_traj
    global true_traj_count
    timeStep = fields[2]
    timeStep_temp = timeStep.replace(".", "")[:-1] + "0000"    #取出估计轨迹的时间戳，并对其处理（去小数点并把末位数去掉补0000）
    flag = False
    count = true_traj_count
    while count < len(true_traj):
        line_true = true_traj[count].strip()  # str.strip()就是把字符串(str)的头和尾的空格，以及位于头尾的\n \t之类给删掉。
        fields_true = line_true.split(' ')  # 拆分字符串。通过指定分隔符对字符串进行切片，并返回分割后的字符串列表（list）
        # 换数据集要修改
        # mh01:1403636580863550000   MH02:1403636859551660000
        # mh03=1403637133238310000,1403637133438310000   mh04=1403638129545090000  mh05=1403638522877820000
        # v101:1403715278762140000  1403715274312140000  1403715279012140000  V102:1403715529662140000
        # v103:1403715893884060000
        # v201:1413393217055760000   v202:1413393889205760000   v203:1413394887355760000    1413394887305760000
        if fields_true[0] == timeStep_temp:    #如果估计轨迹的时间戳等于真实轨迹的时间戳
            if done and (timeStep_temp != "1403715279012140000 "):
                true_traj_count = true_traj_count + 1
                ground = np.append(ground, [[float(fields_true[1]),float(fields_true[2]),float(fields_true[3])]], axis=0)
                estimate = np.append(estimate, [[float(fields[3]),float(fields[4]),float(fields[5])]], axis=0)
            else:
                count = count + 1
                ground[len(ground)-1][0]=float(fields_true[1])
                ground[len(ground)-1][1]=float(fields_true[2])
                ground[len(ground)- 1][2] = float(fields_true[3])
                estimate[len(estimate)- 1][0] = float(fields[3])
                estimate[len(estimate)- 1][1] = float(fields[4])
                estimate[len(estimate)- 1][2] = float(fields[5])
            flag = True
        count = count + 1
        if flag:
            break

    if len(ground) > 2:
        ground_temp = np.transpose(ground)
        estimate_temp = np.transpose(estimate)
        # Recover R and t
        ret_R, ret_t = rigid_transform_3D(estimate_temp,ground_temp)
        # Compare the recovered R and t with the original
        estimate_temp_val = (ret_R @ estimate_temp) + ret_t
        err = ground_temp - estimate_temp_val
        err = err * err  # 点乘
        err = np.sum(err)
        rmse = np.sqrt(err / len(ground))  # 均方根误差
    else:
        rmse = 0.50001
    # 换数据集要修改
    if len(ground) == 2777:  #mh01 = 3637  MH02=2995   MH03=2624   mh04=1965  mh05=1509
                             #v101 = 2777    v102=1597   v103=1984   v201=2076   v202=2274  v203=1857
        ground_3D = open('/home/gavin/ORB_SLAM3-master/ground.txt', 'a', encoding='utf-8')
        estimate_3D = open('/home/gavin/ORB_SLAM3-master/estimate.txt', 'a', encoding='utf-8')
        ground_temp_save = np.transpose(ground_temp).tolist()
        estimate_temp_save = np.transpose(estimate_temp).tolist()   #estimate_temp_val
        # f.write('%d %d %f %d %f\n' % (num_params, num_episodes, params[0], params[1], params[2]))
        for i in range(len(ground)):
            gr_str = str(ground_temp_save[i][0]) + ' ' + str(ground_temp_save[i][1]) +' '+str(ground_temp_save[i][2])+'\n'
            es_str = str(estimate_temp_save[i][0]) + ' ' + str(estimate_temp_save[i][1]) + ' ' + str(estimate_temp_save[i][2])+'\n'
            estimate_3D.write(es_str)
            ground_3D.write(gr_str)
        estimate_3D.close()
        ground_3D.close()
    return rmse

def run_maze():
    global num_params # 声明变量为全局变量
    global num_frames
    global num_episodes
    step = 0
    t = 0
    # store_reward = open('/home/gavin/DRQN_self/reward.txt', 'a', encoding='utf-8')

    #使用mh01训练
    for episode in range(3680):
    # for episode in range(2801):
        # initial observation
        # observation = env.reset()
        countt = 0
        pre_reward = 0     #预先将奖励值设为0
        control_reward_max_number = 0  #控制奖励值  防止奖励值
        while True:
            if countt > 5 or control_reward_max_number > 10:      #表示持续多少次不做优化
                done = True
                num_episodes += 1
            else:
                done = False
            observation, reward = get_observation(done=done)
            # store_reward.write(reward)
            print('num_frames', num_frames)

            # RL.choose_init_state(observation)
            # RL choose action based on observation  # 基于观测选择动作
            action = RL.choose_action(observation)  #根据观测得到Q值最大的索引值
            logging.info("choose_action"+str(action))
            # observe = observation.flatten()
            print('reward,pre_reward', reward, pre_reward)
            # RL take action and get next observation and reward
            control_reward_max_number += 1
            if reward > pre_reward:
                pre_reward = reward
                countt = 0
            else:
                countt += 1
            observation_ = observation
            num_params = num_params + 1
            params = env.parameter_space[action]
            while True:
                succ = False
                try:
                    f = open('/home/gavin/ORB_SLAM3-master/read.txt', 'w')
                    f_all = open('/home/gavin/ORB_SLAM3-master/read_all.txt', 'a')
                    f.write('%d %d %f %d %f\n' % (num_params, num_episodes, params[0], params[1], params[2]))
                    f_all.write('%d %d %f %d %f\n' % (num_params, num_episodes, params[0], params[1], params[2]))
                    print('num_params,countt,num_episodes', num_params, countt, num_episodes)
                    succ = True
                finally:
                    if f:
                        f.close()
                    if f_all:
                        f_all.close()
                if succ:
                    break
                print('wait for SLAM processing ...')
                time.sleep(0.001)
            # 向內存回忆单元（s,a,r,s_）值num_params,countt,num_episodes
            RL.store_transition(observation, action, reward, observation_, t)
            if(step % 1000 == 0):    #step % 10000
                RL.update_target()
            # 控制学习起始时间和频率 (先累积一些记忆再开始学习)
            if (step % 8 == 0) and (step > 2000):   #step > 200 step > 1000
            # if (step % 8 == 0) and (step > 1000):
                RL.learn()
            # swap observation 将下一个 state_ 变为 下次循环的 state
            observation = observation_
            step += 1
            t += 1
            # break while loop when end of this episode
            if done:
                # num_episodes += 1
                break
        #MH01
        # if episode > 3500:
        #V101
        # if episode > 1900:
        #MH02
        if episode > 2500:
            RL.save_model(step, imgname=imgname)
        if (episode % 800 == 0) and (episode >= 800):
            RL.plot_cost(episode)
        # store_reward.close()
    # end of game
    # RL.plot_cost()
    print('game over')
    env.destroy()


def test():
    global num_params
    global num_frames
    global num_episodes
    RL.load_model()
    # 换数据集要修改
    for episode in range(2912):  #MH01 = 3680  MH02=3040  MH03=2700  mh04=2033  mh05=2273
                                 #v101 = 2912  v102=1710  v103=2149  v201=2280  v202=2348  v203=1922
        done = True
        observation, reward = get_observation(done=done)
        print('num_frames', num_frames)
        # RL choose action based on observation
        # action = RL.test_action(observation)
        action = RL.choose_action(observation)  # 根据观测得到Q值最大的索引值
        params = env.parameter_space[action]
        num_params = num_params + 1
        num_episodes += 1
        while True:
            succ = False
            try:
                f = open('/home/gavin/ORB_SLAM3-master/read.txt', 'w')
                f_all = open('/home/gavin/ORB_SLAM3-master/read_all.txt', 'a')
                f.write('%d %d %f %d %f\n' % (num_params, num_episodes, params[0], params[1], params[2]))
                f_all.write('%d %d %f %d %f\n' % (num_params, num_episodes, params[0], params[1], params[2]))
                succ = True
                print('num_params, num_episodes', num_params, num_episodes)
            finally:
                if f:
                    f.close()
            if succ:
                break
            print('wait for SLAM processing ...')
            time.sleep(0.001)
    print('game over')
    env.destroy()

def getTrueTraj():
    global true_traj
    # 换数据集要修改
    f_true = open('/home/gavin/ORB_SLAM3-master/truth_deal/true_m1.txt', 'r')
    while True:
        line_true = f_true.readline()  # 该方法每次读出一行内容，返回一个字符串
        true_traj.append(line_true)
        if not line_true:
            break
    f_true.close()


num_params = 0
num_frames = 0
num_episodes = 1
true_traj = []
true_traj_count = 0


is_training = False
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DRQN")
    parser.add_argument("--gym", type=str, default="gym", help="Type of the environment. Can either be 'gym' or 'retro'")
    parser.add_argument("--network_type", type=str, default="drqn", help="Type of the network to build, can either be 'dqn' or 'drqn'")
    parser.add_argument("--env_name", type=str, default="Breakout-v0", help="Name of the gym/retro environment used to train the agent")
    parser.add_argument("--retro_state", type=str, default="Start", help="Name of the state (level) to start training. This is only necessary for retro envs")
    parser.add_argument("--train", type=str, default="True", help="Whether to train a network or to play with a given network")
    parser.add_argument("--model_dir", type=str, default="saved_session/net/", help="directory to save the model and replay memory during training")
    parser.add_argument("--net_path", type=str, default="", help="path to checkpoint of model")
    parser.add_argument("--steps", type=int, default=50000000, help="number of frames to train")
    args, remaining = parser.parse_known_args()
    conf = GymConfig()
    conf.env_name = args.env_name
    conf.network_type = args.network_type
    conf.train = args.train
    conf.dir_save = args.model_dir
    conf.train_steps = args.steps
    getTrueTraj()
    # maze game
    env = Maze()
    if is_training:
        RL = DeepQNetwork(env.n_actions, env.n_features,
                          learning_rate=0.01,
                          reward_decay=0.9,
                          e_greedy=0.9,
                          replace_target_iter=200,
                          memory_size=2000,
                          output_graph=False,
                          config=conf
                          )
        run_maze()
    else:
        RL = DeepQNetwork(env.n_actions, env.n_features,
                          learning_rate=0.01,
                          reward_decay=0.9,
                          e_greedy=1.0,
                          replace_target_iter=200,
                          memory_size=2000,
                          output_graph=True,
                          config=conf
                          )
        test()

