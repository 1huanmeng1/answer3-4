import numpy as np
from pylab import *
'''-------------------------------------创建函数，并且绘图-----------------------------'''
x = np.linspace(-4,4,500)[:,np.newaxis]  #范围和点数都不要太大
target = -25 + x**2
sample = target + 1*np.random.normal(0,1,x.shape)# 给目标函数加入高斯白噪声作为学习样本
                                                 # 就是加入随机变量，让点分散
plot(x,sample,'.')
show()
'''-----------------------------------调用神经网络进行运算---------------------------'''
# 定义训练次数
TrainNum = 500                          #太大会崩
# 构建输入矩阵
x1 = np.ones([len(x),1])
x2 = x**2
xx = np.hstack((x1,x2))
# 定义初始的theta值
st1 = 1
st2 = 1
st = np.vstack((st1,st2))
# 定义学习率（注意：学习率过大会导致算法不收敛）
alpha = 0.00007                  #点的范围和数量较多时要调小，不然曲线巨特喵诡异
# 预设向量保存损失函数值
loss = np.zeros(TrainNum)
# 开始迭代求解theta
for i in range(TrainNum):
    st1 = st1 - alpha*(np.dot(np.transpose(np.dot(xx,st)-sample),x1))
    st2 = st2 - alpha*(np.dot(np.transpose(np.dot(xx,st)-sample),x2))
    st = np.vstack((st1,st2))
    loss[i] = np.sum(np.power(np.transpose(np.dot(xx,st)-target),2))/(2*len(x))
prediction = np.dot(xx,st)
# 绘制损失函数变化图
plot(x,loss)
legend(['Loss'])
show()
plot(x,prediction,'-')
plot(x,sample,'.')
legend(['Prediction','Sample'])
show()
