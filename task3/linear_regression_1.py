import  numpy as np
import  torch
from  torch.utils import data
from  d2l import  torch as d2l

true_w = torch.tensor([2,-3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w,true_b,1000)

#可以采样数据，还可以高效处理并读取数据
def load_array(data_arrays,batch_size,is_train = True) :
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)#传入X,y，*号是元组拆包
    return  data.DataLoader(dataset,batch_size,shuffle=is_train)#每次选b个样本出来，shuffle为true表示随机打乱顺序

#定义超参
batch_size = 10
#数据迭代器Dataloader
data_iter = load_array((features,labels),batch_size)

next(iter(data_iter))#转成了python中的iterator

from torch import nn
net = nn.Sequential(nn.Linear(2, 1))#输入2，输出1（维度）

# net[0].weight.data.normal_(0, 0.01)#normol用均值为0，标准层为0.01替换data
# net[0].bias.data.fill_(0)

#计算均方误差使用的是MSELoss类，也称为平方L2范数
loss =nn.MSELoss()

#实例化一个SGD实例,定义优化器，输入模型参数和学习率
trainer = torch.optim.SGD(net.parameters(),lr=0.03)#parameters包括了w和b

num_epochs = 3#数据扫3次
for epoch in range(num_epochs):
    for X,y in data_iter:
        l = loss(net(X),y)#net(X)为训练值
        trainer.zero_grad()#清零梯度
        l.backward()#调用backward，自动求sum
        trainer.step()#进行模型更新
    l = loss(net(features),labels)#求所有的loss
    print(f'epoch{epoch+1},loss{l:f}')