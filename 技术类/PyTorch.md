# 第一章 PyTorch 简介与安装
略
# 第二章 PyTorch 核心模块
## 2.1 PyTorch 模块结构
路径：Python安装路径/Lib/site-packages/torch
- __pycache__ : 存放python解释器生成的字节码，后缀通常pyc/pyo，空间换时间
- _C : pyi 文件，校验数据类型，底层计算代码采用C++编写，并封装成库，供pytorch的python调用。
- include : c++的头文件
- lib : c++的静态和动态链接库
- autograd : **核心模块与概念**, 实现了梯度的自动求导，极大地简化了深度学习研究者开发的工作量，开发者只需要编写前向传播代码，反向传播部分由autograd自动实现，再也不用手动去推导数学公式，然后编写代码了
- nn : **99%开发者使用频率最高的模块，搭建网络层就在nn.modules里面**
- onnx : pytorch模型转换到onnx模型表示的核心模块
- optim : 优化模块，深度学习的学习过程，就是不断的优化，而优化使用的方法函数，都暗藏在了optim文件夹中。常见的优化方法：Adam, SGD, ASGD. 以及非常重要的学习率调整模块，lr_scheduler.py
- utils : 各类常用工具，其中比较关键的是data文件夹，tensorboard文件夹

路径: Python安装路径/Lib/site-packages/torchvision
- datasets : 官方常用的数据集写的**数据读取函数**
- models : 存放了经典的、可复现的、有训练权重参数可下载的视觉模型，例如分类的alexnet,densenet,efficientnet,mobilenet-v1/2/3,resnet等，分割模型、检测模型、视频任务模型、量化模型。这个库中的模型实现，也是可以借鉴学习的好资料，可以模仿它们的代码结构、函数、类的组织。
- ops : 视觉任务特殊的功能函数，例如检测中用到的roi_align,roi_pool,boxes的生成，以及focal_loss实现，都在这里边有实现。
- transformers : 数据增强库，是pytorch自带的图像预处理、增强、转换工具，可以满足日常的需求。Albumentations。




张量：表示一个数值组成的数组，可能有多个维度。具有一个轴的张量对应数学上的向量(vector)。    
具有俩个轴的张量对应数学上的矩阵(matrix);          
```
x = torch.arange(12)  # 创建行向量 x    
x.shape   # 张量(沿每个轴的长度)的形状    
x.size   # 张量中元素的总数，即形状的所有元素乘积。    
x.reshape(3,4)  # 修改张量的形状而不改变元素的数量和元素值    
torch.zeros((2,3,4))  # 创建形状是(2，3，4)且其他元素都是0的张量    
torch.randn(3,4)  # 创建一个形状为(3,4)的张量，其中每个元素都从均值为0，标准差为1的标准高斯分布(正太分布)中随机采样
``` 

适用于同一形状的任意俩个张量上按元素操作。    
标准算术运算符：+ - * / **     
幂运算：torch.exp(x)    
线性代数运算，包括点积和矩阵乘法    

多张张量的连接在一起
```
torch.cat((X,Y),dim=0)    # 按行，纵向拼接    
torch.cat((X,Y),dim=1)    # 按列，横向拼接

X==Y  # 每个元素判断相等，相等处为真，否则该位置为0    
x.sum()  # 求和
```

广播机制
```
a = torch.arange(3).reshape((3,1))    
b = torch.arange(2).reshape((1,2))    
a+b     # 形状不匹配，广播之后为一个更大的 3x2 的矩阵，矩阵a将复制列，
```

索引和切片    
```
x[-1]  # 最后一个索引
x[1:3]  # 第一个元素和最后一个之前的元素
x[0:2,:]  # 超过2个维度的张量
```

节省内存        
```
z = troch.zeros_like(y)     
z[:] = x+y    # 原地更新 不重新分配内存
```

转换为python的其他对象    
```
np_a = x.numpy()    # tensor -> numpy    
y = torch.tensor(np_a)  # numpy -> tensor        
b = torch.tensor([3.5])
b.item()    # 转换为python标量    
float(b)    
int(b)
```

数据的预处理    
用pandas来预处理数据，处理缺失值
```
inputs, outputs = data.iloc[:,0:2],data.iloc[:,2]
inputs = inputs.fillna(input.mean())

inputs = pd.get_dummies(inputs,dummy_na=True)  # 对inputs的类别值或离散值，将NaN视为一个类别
```

pandas亦可转换为标量    
```
x,y = torch.tensor(inputs.values),torch.tensor(outputs.values)
```

**线性代数**    
标量: 只有一个元素的张量表示 `x=torch.tensor(3.0)`    
向量: 标量组成的列表。 `x=torch.arange(3) # tensor([0,1,2])`










