# 深度学习

目标：介绍深度学习经典和最新的模型：Lenet，ResNet等

内容：<img src="C:\Users\86159\AppData\Roaming\Typora\typora-user-images\image-20230228231515249.png" alt="image-20230228231515249" style="zoom:25%;" />



资源

<img src="C:\Users\86159\AppData\Roaming\Typora\typora-user-images\image-20230302194020925.png" alt="image-20230302194020925" style="zoom: 25%;" />

### 深度学习的介绍

有效$≠$可解释性

领域专家：甲方；

数据科学家：乙方；

##### 数据操作

机器学习和神经网络的主要数据结构是N维数组

##### 实现

[d2l-zh-pytorch-slides/ndarray.ipynb at main · d2l-ai/d2l-zh-pytorch-slides (github.com)](https://github.com/d2l-ai/d2l-zh-pytorch-slides/blob/main/chapter_preliminaries/ndarray.ipynb)



<img src="C:\Users\86159\AppData\Roaming\Typora\typora-user-images\image-20230307164214320.png" alt="image-20230307164214320" style="zoom: 50%;" />

##### 

reshape()：改变x的形状，不改变值

初始化全0：troch.zeros()

<img src="C:\Users\86159\AppData\Roaming\Typora\typora-user-images\image-20230307164436153.png" alt="image-20230307164436153" style="zoom:50%;" />

x**y表示求幂运算



![image-20230307164641063](C:\Users\86159\AppData\Roaming\Typora\typora-user-images\image-20230307164641063.png)

dim=0表示第0维合并（行），=1表示列



x==y对每个元素进行判断



![image-20230307164914313](C:\Users\86159\AppData\Roaming\Typora\typora-user-images\image-20230307164914313.png)

广播机制，a 1->2, b 1->3  形成一个3*2矩阵（**注意，容易出错**）



![image-20230307165211473](C:\Users\86159\AppData\Roaming\Typora\typora-user-images\image-20230307165211473.png)

x =0;x<2;赋值



##### 数据预处理

[d2l-zh-pytorch-slides/pandas.ipynb at main · d2l-ai/d2l-zh-pytorch-slides (github.com)](https://github.com/d2l-ai/d2l-zh-pytorch-slides/blob/main/chapter_preliminaries/pandas.ipynb)



关于浅复制：

<img src="C:\Users\86159\AppData\Roaming\Typora\typora-user-images\image-20230307172722925.png" alt="image-20230307172722925" style="zoom:50%;" />



##### **线性代数**

[d2l-zh-pytorch-slides/linear-algebra.ipynb at main · d2l-ai/d2l-zh-pytorch-slides (github.com)](https://github.com/d2l-ai/d2l-zh-pytorch-slides/blob/main/chapter_preliminaries/linear-algebra.ipynb)

![image-20230307214947524](C:\Users\86159\AppData\Roaming\Typora\typora-user-images\image-20230307214947524.png)

![image-20230307215106524](C:\Users\86159\AppData\Roaming\Typora\typora-user-images\image-20230307215106524.png)

![image-20230307215909781](C:\Users\86159\AppData\Roaming\Typora\typora-user-images\image-20230307215909781.png)

![image-20230307220036239](C:\Users\86159\AppData\Roaming\Typora\typora-user-images\image-20230307220036239.png)

关于向量求导

![image-20230307221223591](C:\Users\86159\AppData\Roaming\Typora\typora-user-images\image-20230307221223591.png)

![image-20230307221546691](C:\Users\86159\AppData\Roaming\Typora\typora-user-images\image-20230307221546691.png)



### 08 线性回归 + 基础优化算法

#### 线性回归

简化模型：

![image-20230321215008963](C:\Users\86159\AppData\Roaming\Typora\typora-user-images\image-20230321215008963.png)

对应的线性模型：

![image-20230321215040001](C:\Users\86159\AppData\Roaming\Typora\typora-user-images\image-20230321215040001.png)

对应为单层神经网络

![image-20230321215111976](C:\Users\86159\AppData\Roaming\Typora\typora-user-images\image-20230321215111976.png)

1/2方便求导消去

![image-20230321215310676](C:\Users\86159\AppData\Roaming\Typora\typora-user-images\image-20230321215310676.png)

这里X是一个矩阵，x_{i}的意思是第i条数据，每一条数据都包含了（房间个数，居住面积）等决策信息

![image-20230321215432867](C:\Users\86159\AppData\Roaming\Typora\typora-user-images\image-20230321215432867.png)



![image-20230321215739323](C:\Users\86159\AppData\Roaming\Typora\typora-user-images\image-20230321215739323.png)



![image-20230321215929859](C:\Users\86159\AppData\Roaming\Typora\typora-user-images\image-20230321215929859.png)

####  基础优化算法

##### 梯度下降法

学习率不能太小，也不能太大（详见吴恩达机器学习）

![image-20230321220105520](C:\Users\86159\AppData\Roaming\Typora\typora-user-images\image-20230321220105520.png)

随机取部分样本来近似损失模拟



![image-20230321220410324](C:\Users\86159\AppData\Roaming\Typora\typora-user-images\image-20230321220410324.png)

![image-20230321220523157](C:\Users\86159\AppData\Roaming\Typora\typora-user-images\image-20230321220523157.png)

![image-20230321220705779](C:\Users\86159\AppData\Roaming\Typora\typora-user-images\image-20230321220705779.png)

[d2l-zh-pytorch-slides/linear-regression-scratch.ipynb at main · d2l-ai/d2l-zh-pytorch-slides (github.com)](https://github.com/d2l-ai/d2l-zh-pytorch-slides/blob/main/chapter_linear-networks/linear-regression-scratch.ipynb)



##### QA

batch_size小更好。收敛越好。随机梯度下降理论上是带来了噪音。但是对于神经网络来说，比较好，神经网络比较复杂，容易不过拟合。



![image-20230322122115094](C:\Users\86159\AppData\Roaming\Typora\typora-user-images\image-20230322122115094.png)

是的。



![image-20230322122440764](C:\Users\86159\AppData\Roaming\Typora\typora-user-images\image-20230322122440764.png)

个人而言这段回答很有深度，面对实际的问题我们不可能拿到精确的模型，在大样本的情况下建模不准确快速准确的求解模型反而没有大方向是对的随机计算好。



![image-20230322154000576](C:\Users\86159\AppData\Roaming\Typora\typora-user-images\image-20230322154000576.png)

1. 拿一个小的样本，即最后一个不满batch_size
2. 忽略最后一次epoch的迭代 
3. 不足部分从原有数据中抽出差额补齐。



![image-20230322154404244](C:\Users\86159\AppData\Roaming\Typora\typora-user-images\image-20230322154404244.png)是的，求解是NP问题。
