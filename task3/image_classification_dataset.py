#MNIST数据集 是图像分类中广泛使用的数据集之一，但作为基准数据集过于简单。 我们将使用类似但更复杂的Fashion-MNIST数据集
import  torch
import torchvision
from torch.utils import data
from torchvision import transforms#对数据进行操作的模具
from d2l import torch as d2l

d2l.use_svg_display()#svg显示，清晰度高一点

#通过框架中的内置函数将Fashion-MNIST数据集下载并读取到内存中

trans = transforms.ToTensor()#将PIL换为32位浮点数格式
mnist_train = torchvision.datasets.FashionMNIST(
    root="../data", train=True, transform=trans, download=True)#transform转换格式为tensor
mnist_test = torchvision.datasets.FashionMNIST(
    root="../data", train=False, transform=trans, download=True)

#print(len(mnist_test),len(mnist_train))
#1w 6w
print(mnist_train[0][0].shape)#1为黑白图，rgb=1，长宽为28
#torch.Size([1, 28, 28])


#两个可视化数据集的函数
def get_fashion_mnist_labels(labels):
    """返回Fashion-MNIST数据集的文本标签"""
    test_labels=['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [test_labels[int(i)] for i in labels]

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            ax.imshow(img.numpy())
        else:
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

X,y = next(iter(data.DataLoader(mnist_train,batch_size=18)))#得到批量为18的数据
show_images(X.reshape(18,28,28),2,9,titles=get_fashion_mnist_labels(y))#18为数据大小,大小为28，画2行，每行9个图片

batch_size = 256

def get_dataloader_workers():
    """使用4个进程来读取数据"""
    return 4

train_iter = data.DataLoader(mnist_train,batch_size,shuffle=True,num_workers=get_dataloader_workers())

# timer = d2l.Timer()
# for X,y in train_iter:
#     continue
# print(f'{timer.stop():.2f} sec')

#封装,resize换图片大小
def load_data_fashion_mnist(batch_size,resize=None):
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0,transforms.Resize(resize))#变图片大小
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))

train_iter, test_iter = load_data_fashion_mnist(32, resize=64)
for X, y in train_iter:
    print(X.shape, X.dtype, y.shape, y.dtype)
    break

d2l.plt.show()
