# coding=UTF-8
"""
Reinforcement learning maze example.
Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].
This script is the environment part of this example.
The RL is in RL_brain.py.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

"""定义三个参数的取值"""
import numpy as np
import time
import sys

UNIT = 40   # pixels  像素
MAZE_H = 84  # grid height  网格高度
MAZE_W = 84  # grid width   网格宽度
# class Maze(tk.Tk, object):
class Maze(object):    #继承父类object
    def __init__(self):
        super(Maze, self).__init__()   #对继承自父类的属性进行初始化
        para1 = list(range(100, 140)) #1.1-1.4
        para2 = [2, 3, 4, 5, 6, 7, 8] #2-8
        para3 = list(range(60, 90))  # 0.6-0.9
        self.parameter_space = []
        for i in range(len(para1)):
            for j in range(len(para2)):
                for k in range(len(para3)):
                    self.parameter_space.append((para1[i]/100.0, para2[j], para3[k]/100.0))
                    #append() 方法用于在列表末尾添加新的对象。
        #print(self.parameter_space)
        ###self.action_space = ['u', 'd', 'l', 'r']
        self.action_space = []   #三个参数的取值空间
        for i in range(len(para1)*len(para2)*len(para3)):
            self.action_space.append(i)   #[0-10500]
        self.n_actions = len(self.action_space)
        self.height = MAZE_H
        self.width = MAZE_H
        self.n_features = self.height * self.width   #特征网格的大小
        # self.title('maze')
        # self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_H * UNIT))
        # self._build_maze()

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white',  
                           height=MAZE_H * UNIT,
                           width=MAZE_W * UNIT)

        # create grids
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # create origin
        origin = np.array([20, 20])

        # hell
        hell1_center = origin + np.array([UNIT * 2, UNIT])
        self.hell1 = self.canvas.create_rectangle(
            hell1_center[0] - 15, hell1_center[1] - 15,
            hell1_center[0] + 15, hell1_center[1] + 15,
            fill='black')                  #绘制黑色矩形框

        # create oval
        oval_center = origin + UNIT * 2
        self.oval = self.canvas.create_oval(
            oval_center[0] - 15, oval_center[1] - 15,
            oval_center[0] + 15, oval_center[1] + 15,
            fill='yellow')

        # create red rect
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')              #绘制红色圆形

        # pack all
        self.canvas.pack()
    #重置环境状态，回到初始环境，方便下一次观测
    def reset(self):
        observation = np.zeros(self.n_features)  #创建一个一维数组,大小为n_features
        return observation
    #推进时间步长，返回下一状态和奖励
    def step(self, action):

        s_ = np.zeros(self.n_features)
        parameter = self.parameter_space[action]
        print(parameter)
        reward = parameter[0] + parameter[1] + parameter[2]
        done = True
        return s_, reward, done

    def render(self):
        # time.sleep(0.01)
        self.update()


