import torch
x = torch.arange(12) #size = 12,元素值为0,1,2..11
# print(x)  # tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
#print(x.size()) #torch.Size([12])
print(x.numel()) #元素数量
X = x.reshape(3,4) #改变一个张量的形状而不改变元素数量和元素值，可以调用reshape函数
print(X)

#使用全0、全1、其他常量，或者从特定分布中随机采样的数字
#Y = torch.zeros((2,3,4)) #ones
#print("Y\n",Y)
Y = torch.randn((3,4))
print("Y\n",Y)

#通过提供包含数值的Python列表（或嵌套列表），来为所需张量中的每个元素赋予确定值
#torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])

X = torch.arange(12, dtype=torch.float32).reshape(3, 4)
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
Z = torch.cat((X, Y), dim=0), torch.cat((X, Y), dim=1)
print("X\n",X)
print("Y\n",Y)
print("Z\n",Z)