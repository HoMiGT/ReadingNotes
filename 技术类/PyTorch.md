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










