import torch
from torch import nn
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

#实现一个具有单隐藏层的多层感知机， 它包含256个隐藏单元
num_inputs, num_outputs, num_hiddens = 784, 10, 256

#nn.Parameter声明是一个nn.Parameter，可以不加。
#W1行数为784,列数为256
#设置为零的话梯度为0，参数不会更新，相当于隐藏层只有一个单元
W1 = nn.Parameter(torch.randn(
    num_inputs,num_hiddens,requires_grad=True)*0.01)
#偏差设为0
#这里的*0.01的作用是为了缩小张量的数值范围，使其更接近于0。这样可以避免梯度爆炸或消失的问题，提高模型的稳定性和收敛速度
b1 = nn.Parameter(torch.zeros(num_hiddens,requires_grad=True)*0.01)

W2 = nn.Parameter(torch.randn(
    num_hiddens,num_outputs,requires_grad=True)*0.01)
b2 = nn.Parameter(torch.zeros(num_outputs,requires_grad=True)*0.01)

params= [W1,b1,W2,b2]

def relu(X):
    a = torch.zeros_like(X)
    return  torch.max(X,a)

def net(X):
    X = X.reshape((-1,num_inputs))#拉成矩阵
    H = relu(X @ W1+b1)#@ 是numpy里面的点积运算符号，相当于np.dot()
    return  (H @ W2 +b2)

loss = nn.CrossEntropyLoss(reduction='none')

num_epochs, lr = 10, 0.1
updater = torch.optim.SGD(params,lr=lr)
d2l.train_ch3(net,train_iter,test_iter,loss,num_epochs,updater)

d2l.predict_ch3(net, test_iter)
d2l.plt.show()
