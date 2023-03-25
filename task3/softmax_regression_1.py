import torch
from torch import nn
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# PyTorch不会隐式地调整输入的形状。因此，
# 我们在线性层前定义了展平层（flatten），来调整网络输入的形状
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))#flatten变为二维向量，第一维保留，其他合并为1维


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights);#这里只是在定义模型以及初始化

loss = nn.CrossEntropyLoss(reduction='none')
#loss = nn.CrossEntropyLoss()默认是reduction = 'mean'，把图片往下拉，会发现loss曲线是0，一条直线
trainer = torch.optim.SGD(net.parameters(), lr=0.1)

num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
d2l.predict_ch3(net, test_iter)
d2l.plt.show()