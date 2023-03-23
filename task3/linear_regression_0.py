
import matplotlib # 注意这个也要import一次
import matplotlib.pyplot as plt
import  random
import torch
from  d2l import  torch as d2l



def synthetic_data(w,b,num_eg):
    """生成y = XW+b+噪声"""
    #这里的X指房屋的关键因素集，长度len(w)即列数，表明有len(w)个关键因素，这里是2，比如“卧室个数”和“房屋面积”两个关键因素，X的行数num_examples=房屋数量
    X = torch.normal(0,1,(num_eg,len(w))) #均值为0，标准差为1，列数为w
    y = torch.matmul(X,w)+b
    y += torch.normal(0,0.01,y.shape) #加入随机噪音
    #根据numpy官网的介绍,reshape(-1,1)或reshape(1,-1)中的-1表示未指定，如果需要1行，列就-1，反之亦然
    return X,y.reshape((-1,1))


def data_iter(batch_size, features, labels):
    """定义一个data_iter函数， 该函数接收批量大小、特征矩阵和标签向量作为输入，生成大小为batch_size的小批量"""
    num_eg = len(features)#features 是1000＊2的矩阵，len()是取其第一维度长度
    indices = list(range(num_eg))#生成0到num_eg-1的数，转为list
    random.shuffle(indices)#打乱顺序，随机读取
    for i in range(0,num_eg,batch_size):
        batch_indices = torch.tensor(
            indices[i:min(i+batch_size,num_eg)]#每次拿batch_size个
        )
        yield  features[batch_indices],labels[batch_indices]#yield就是 return 返回一个值，并且记住这个返回的位置，下次迭代就从这个位置后开始。

true_w = torch.tensor([2,-3.4])
true_b = 4.2
features, labels = synthetic_data(true_w,true_b,1000)
# print('features:',features[0],'\nlabel:',labels[0])
# d2l.set_figsize()
# d2l.plt.scatter(features[:, 1].detach().numpy(),labels.detach().numpy(),1)
# d2l.plt.show()


batch_size = 10

# for X, y in data_iter(batch_size, features, labels):
#     print(X, '\n', y)
#     break

w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

def linreg(X, w, b):
    """线性回归模型"""
    return torch.matmul(X, w) + b

def squared_loss(y_hat, y):
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def sgd(params,lr,batch_size):
    """小批量随机梯度下降"""
    with torch.no_grad():#梯度存起来了，这里应该是读取值，并不需要更新梯度
        for param in params:
                param -= lr*param.grad/batch_size
                param.grad.zero_()#pytorch会不断的累加变量的梯度，所以每更新一次参数，就要让其对应的梯度清零

lr = 0.03#学习率
num_epochs = 3#数据扫3次
net = linreg#定义模型为线性回归
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)#x,y是小批量损失
        #l是batch_size大小的向量
        l.sum().backward()
        sgd([w, b], lr, batch_size)#最后一个可能没有batch_size大小，这里设batch_size为10，样例1000，最后刚好整除
        #对当前模型进行评估，不需要反向传播求梯度
        with torch.no_grad():
            train_l = loss(net(features, w, b), labels)
            print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')#f就代表花括号里表达式可以用表达式的值代替

        print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
        print(f'b的估计误差: {true_b - b}')