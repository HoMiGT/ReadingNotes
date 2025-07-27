# 引言
据估计，超过80%的数据是非结构化的。以文本、图像、音频、视频等形式存在。文档又占非结构化数据的50%。    
涉及到的Python库：
- NLTK: 自然语言工具包
- SpaCy: 涵盖自然语言处理，并加入深度学习框架
- TextBlob: 基于NLTK和Pattern，受欢迎，但也不是最快或最完整的
- CoreNLP: Stanford CoreNLP的Python的包装器。为各种语言提供文本标记、句法分析和文本分析的技术。

11种自然语言处理的应用
- 情感分析: 客户对企业提供的产品的情感
- 主题建模: 从一组文档中提取唯一的主题
- 投诉分类/邮件分类/电子商务产品分类等
- 使用不同的聚类技术进行文档分类和管理
- 使用相似度方法进行简历筛选和职位描述匹配
- 利用先进的特征工程技术(word2vec和fastText)来捕获上下文
- 信息/文档检索系统，例如搜索引擎
- 聊天机器人、问答(Q&A)以及语音到文本的应用程序(Siri和Alexa)
- 使用神经网络进行语言检测和翻译
- 使用图形方法和先进技术进行文本总结
- 使用深度学习算法进行文本生成/预测下一个单词序列

# 第1章 提取数据
## 方法1-1 适用API收集文本数据
问题：使用推特API收集文本数据    
实现：    
```Python
# pip install tweepy
import numpy as np
import tweepy
import json
import pandas as pd
from tweepy import OAuthHandler
# 验证
consumer_key = "adjbiejfaaoeh"
consumer_secret = "had73haf78af"
access_token = "jnsfby5u4yuawhafjeh"
access_token_secret = "jhdfgay768476r"
# 调用api
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)
# 提供查询的数据, 搜索ABC时将提供最靠前的10个推文
Tweets = api.search(query, count=10, lang='en', exclude='retweets', tweet_mode='extended')
```
## 方法1-2 从PDF中收集数据
问题：读取一个pdf文件    
实现：
```Python
# pip install PyPDF2
import PyPDF2
from PyPDF2 import PdfFileReader
# 创建一个pdf的read object
pdf = open('file.pdf', 'rb')
pdf_reader = PyPDF2.PdfFileReader(pdf)
print(pdf_reader.numPages)
page = pdf_reader.getPage(0)
# 打印pdf的内容
print(page.extractText())
pdf.close()
```
## 方法1-3 从Word文件中收集数据
问题：读取一个word文件    
实现： 
```Python
# pip install docx
from docx import Document
doc = open('file.docx','rb')
document = Document(doc)
content = ""
for para in document.paragraphs:
    content += para.text
print(content)
```
## 方法1-4 从Json中收集数据
问题：读取一个json文件/对象    
实现： 
```Python
# pip install requests
import request
import josn
r = requests.get("https://quotes.rest/qod.json")
res = r.json()
print(json.dumps(res,indent=4))
# 提取内容
q = res['content']['quotes'][0]
print(q)
```
## 方法1-5 从HTML中收集数据
问题：读取/解析HTML页面    
实现：
```Python
# pip install bs4
# 关于爬虫，scrapy框架以及Playwright、Selenium
import urllib.request as urllib2
from bs4 import BeautifulSoup
response = urllib2.urlopen("https://en.wikipedia.org/wiki/Natural_language_processing")
html_doc = response.read()
# 解析
soup = BeautifulSoup(html_doc, 'html.parser')
# 格式化
strhtm = soup.prettify()
print(strhtm[:1000])
print(soup.title)
print(soup.title.string)
print(soup.a.string)
print(soup.b.string)

for x in soup.find_all('a'):
    print(x.string)

for x in soup.find_all('p'):
    print(x.text)
```
## 方法1-6 使用正则表达式解析文本
问题：希望使用正则表达式来解析文本数据    
补充：
- 基本标志
  - re.I 忽略大小写
  - re.L 查找本地从属项
  - re.M 通过多行找处对象
  - re.S 查找点匹配
  - re.U 采用统一字符编码标准的数据
  - re.X 以更可读的格式编写正则表达式
- 表达式功能
  1. 查找单词出现的字符 [ab]
  2. 查找除a和b以外的字符 [^ab]
  3. 查找a到z之间的字符 [a-z]
  4. 查找a到z以外的字符 [^a-z]
  5. 匹配任意单个字符 .
  6. 匹配任意空白字符 \s
  7. 匹配任意非空白字符 \S
  8. 匹配任意数字 \d
  9. 匹配任意非数字 \D
  10. 匹配任意词 \w
  11. 匹配任意非词 \W
  12. 匹配a或b (a|b)
  13. 匹配0次或1次，且不超过1次的正则 ?
  14. 出现0次及以上 *
  15. 出现1次及以上 +
  16. 出现3次 {3}
  17. 出现3次及以上 {3,}
  18. 出现3到6次 {3,6}
  19. 开始字符串 ^
  20. 结束字符串 $
  21. 匹配词的边界 \b
  22. 匹配非词的边界 \B 
- re.match与re.search的区别
  - re.match 只检查字符串开头是否匹配
  - re.search 检查字符串任意位置是否匹配
实现：    
```Python
# Python内置库 re
import re
# 词汇单元化
re.split('\s+','I like this book.')  # output ['I','like','this','book.']
# 提取邮件IDs
doc = 'For more detials please email us at: xyz@abc.com,pqr@mno.com'
addresses = re.findall(r'[\w\.-]+@[\w\.-]+', doc)  
for address in addresses:
    print(address)  # output xyz@abc.com     pqr@mno.com
# 替换邮件IDs
doc = 'For more details please mail us at xyz@abc.com'
new_doc = re.sub(r'([\w\.-]+)@(\w\.-)+', r'pqr@mno.com', doc)
```
## 方法1-7 处理字符串
问题：希望探索字符串处理    
补充：
- s.find(t) 返回s中字符串t第一次出现的位置的索引，没有找到返回-1
- s.rfind(t) 最后一次出现
- s.index(t) 同 s.find(t) 未找到报 ValueError
- s.rindex(t) 同 s.rfind(t) 未找到报 ValueError
- s.join(text) 拼接字符串
- s.split(t) 拆分字符串，默认以空格
- s.splitlines(t) 将每一行拆分成一系列字符串
- s.lower() 小写
- s.upper() 大写
- s.title() 首字母大写
- s.strip() 剔除字符串开头和结尾的空格
- s.replace(t,u) 将s中的t用u替换

实现：
```Python
# 替换内容
str_v1 = 'I am exploring NLP'
print(str_v1[0])
print(str_v1[5:14])
str_v2 = str_v1.replace('exploring','learning')
print(str_v2)
# 连接俩个字符串
s1 = 'nlp'
s2 = 'machine learning'
s3 = s1+s2
# 搜索字符串中的子串
var = 'I am learning NLP'
f = 'learn'
var.find(f)  # output 5
```
## 方法1-8 从网页抓取文本
问题：从网页中抓取数据，以IMDB网站抓取排名靠前的电影为例    
实现：
```Python
# pip install bs4
# pip install requests
# 代码省略 每个网页都不同
```

# 第2章 探索和处理文本数据
预处理包括将原始文本数据转换成可理解的格式。
## 方法2-1 将文本数据转换为小写形式
问题：将文本数据转换成小写形式    
实现：
```Python
# lower()
import pandas as pd
text = ['This is introduction to NLP', 'It is likely to be useful, to people','Machine learning is the new eletrity','There would be less hype around AI and more action going forward',
'python is the best tool!','R is good langauage','I like this book','I want more books like this']
df = pd.DataFrame({'tweet':text})
x = 'Testing'
x2 = x.lower()

df['tweet'] = df['tweet'].apply(lambda x: ' '.join(x.lower() for x in s.split()))
```
## 方法2-2 删除标点符号
问题：希望从文本数据中删除标点符号    
解决：
```Python
# 最简单实用正则表达式和replace()函数
text = ['This is introduction to NLP', 'It is likely to be useful, to people','Machine learning is the new eletrity','There would be less hype around AI and more action going forward',
'python is the best tool!','R is good langauage','I like this book','I want more books like this']
import pandas as pd
df = pd.DataFrame({'tweet':text})
import re
s = 'I.like.This book!'
s1 =  sub(r'[^\w\s]','',s)

# 或
df['tweet'] = df['tweet'].str.replace('[^\w\s]','')

# 或
import string
s = 'I.like.This book!'
for c in string.punctuation:
    s = s.replace(c,'')
```
## 方法2-3 删除停止词
问题：希望从文本数据中删除停止词    
实现：
```Python
# pip install nltk
text = ['This is introduction to NLP', 'It is likely to be useful, to people','Machine learning is the new eletrity','There would be less hype around AI and more action going forward',
'python is the best tool!','R is good language','I like this book','I want more books like this']
import pandas as pd
df = pd.DataFrame({'tweet':text})
import nltk
nltk.download()
for nltk.corpus import stopwords
stop = stopwords.words('english')
df['tweet'] = df['tweet'].apply(lambda x: ' '.join(x for x in x.split() if x not in stop))
print(df['tweet'])
```
## 方法2-4 文本标准化
问题：希望实现文本标准化    
实现：
```Python
# 创建一个自定义字典
lookup_dict = {'nlp': 'natural language processing', 'ur': 'your', 'wbu': 'what about you'}
import re
def text_std(input_text):
    words = input_text.split()
    new_words = []
    for word in words:
        word = re.sub(r'[^\w\s]','',word)
        if word.lower in lookup_dict:
            word = lookup_dict[word.lower()]
            new_words.append(word)
            new_text = ' '.join(new_words)
    return new_text
text_std('I like nlp it\'s ur choice')
```
## 方法2-5 拼写校正
问题：希望拼写矫正    
实现：    
```Python
# pip install TextBlob
text = ['Introduction to NLP','It is likely to be useful, to people','Machine learning is the new electrcity','R is good langauage','I like this book','I want more books like this']
import pandas as pd
df = pd.DataFrame({'tweet':text})
from textblob import TextBlob
df['tweet'].apply(lambda x:  str(TextBlob(x).correct()))
print(df)  # 纠正了electricity，language

# 自动纠正库
# pip install autocorrect
from autocorrect import spell
print(spell(u'mussage'))  # message
print(spell(u'sirvice'))  # service
```
## 方法2-6 文本分词
问题：希望实现分词    
实现：
```Python
text = ['This is introduction to NLP', 'It is likely to be useful, to people','Machine learning is the new eletrity','There would be less hype around AI and more action going forward',
'python is the best tool!','R is good language','I like this book','I want more books like this']
import pandas as pd
df = pd.DataFrame({'tweet':text})
import nltk
nltk.download('punkt_tab')
from textblob import TextBlob
print(TextBlob(df['tweet'][3]).words)  # WordList(['Would','less','hype','around','ai','action','going','forward'])
mystring = 'My favorite animal is cat'
nltk.word_tokenize(mystring)  # ['My','favorite','animal','is','cat']
```
## 方法2-7 词干提取
问题：词干提取    
实现
```Python
# pip install nltk
# 或 pip install textblob
text = ['I like fishing','I eat fish','There are many fishes in pound']
import pandas as pd
df = pd.DataFrame({'tweet':text})
from nltk.stem import PorterStemmer
st = PorterStemmer()
df['tweet'][:5].apply(lambda x: ' '.join([st.stem(word) for word in x.split()]))
print(df)   #  都提取出了 fish
```
## 方法2-8 词形还原
问题：希望实现词形还原
实现：
```Python
text = ['I like fishing','I eat fish','There are many fishes in pound']
import pandas as pd
df = pd.DataFrame({'tweet':text})
from textblob import Word
df['tweet'] = df['tweet'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
print(df['tweet'])  # 可以发现fish和fishes 被还原成fish，leaves和leaf都被词形还原成leaf 
```
## 方法2-9 探索文本数据
问题：希望对文本数据进行探索和理解
实现：
```Python
# 读取文本数据
import nltk
from nltk.corpus import webtext
nltk.download('webtext')
wt_sentences = webtext.sents('firefox.txt')
wt_words = webtext.words('firefox.txt')
# 导入必要的库，用于计算单词出现的频率
from nltk.probability import FreqDist
from nltk.corpus import stopwords
import string
# 计算所有单词在评论中出现的频率
frequency_dist = nltk.FreqDist(wt_words)
sorted_frequency_dist =  sorted(frequency_dist,key=frequency_dist.__getitem__,reverse=True)
# 考虑长度大于3的单词并绘图
large_words = dict([(k,v) for k, v in frequency_dist.items() if len(k)>3])
frequency_dist = nltk.FreqDist(large_words)
frequency_dist.plot(50,cumulative=False)
# pip install wordcloud
from wordcloud import WordCloud
wcloud = WordCloud().generate_from_frequencies(frequency_dist)
import matplotlib.pyplot as plt
plt.imshow(wcloud,interpolation='bilinear')
plt.axis('off')
plt.show()
```
## 方法2-10 建立一个文本预处理流水线
问题：希望构建一个端到端的文本预处理流水线。无论何时相对任何自然语言处理应用流程进行预处理，都可以直接将数据输入这个流水线处理，并获得干净文本数据作为输出。
实现：
```Python
# 省略，例子是推文，无实际参考意义
# 本质就是用函数封装上述的文本处理
```

# 第3章 文本特征工程
## 方法3-1 使用One-Hot编码将文本转换为特征
问题：希望使用One-Hot编码将文本转换为特征    
缺点：未考虑单词出现的频率，每个单词都当成一个特征
实现：
```Python
text = 'I am learning NLP'
import pandas as pd
pd.get_dummies(text.split())  # 即可实现onehot编码
```
## 方法3-2 使用统计向量器将文本转换为特征
问题：如何通过统计向量器将文本转换为特征
缺点：每个单词都当成一个特征
实现：    
```Python
# pip install sklearn
# sklearn 有一个特征抓取函数，可以从文本中提取特征
from sklearn.feature_extraction.text import CountVectorizer
text =['I love NLP and I will learn NLP in 2month']
vectorizer = CountVectorizer()
vectorizer.fit(text)
vector = vectorizer.transform(text)
print(vectorizer.vocabulary_)  # {'love':4, 'nlp':5, 'will':6,'learn':3,'in':2,'2month':0} 
print(vector.toarray())  # [[1,1,1,1,1,2,1]]
```
## 方法3-3 生成N-grams
问题：对给定的句子生成N-grames    
实现：
```Python
# 使用textblob生成N-grames
text = 'I am learning NLP'
from textblob import TextBlob
print(TextBlob(text).ngrams(1))
print(TextBlob(text).ngrams(2))
# 基于Bigram的文档特征
from sklearn.feature_extraction.text import CountVectorizer
text = ['I love NLP and I will learn NLP in 2month']
vectorizer = CountVectorizer(ngram_range=(2,2))
vectorizer.fit(text)
vector = vectorizer.transform(text)
print(vectorizer.vocabulary_)
print(vector.toarray())
```
## 方法3-4 生成共生矩阵
问题：理解并生成共生矩阵    
实现：
```Python
import numpy as np
import nltk
from nltk import bigrams
import pandas as pd
import itertools
# 创建共生矩阵
def co_occurence_matrix(corpus):
    vocab = set(corpus)
    vocab = list(vocab)
    vocab_to_index = {word:i for i, word in enumerate(vocab)}
    # 创建bigrams
    bi_grams = list(bigrams(corpus))
    bigram_freq = nltk.FreqDist(bi_grams).most_common(len(bi_grams))
    co_occurence_matrix = np.zeros((len(vocab),len(vocab)))
    for bigram in bigram_freq:
        current = bigram[0][1]
        previous = bigram[0][0]
        count = bigram[1]
        pos_current = vocab_to_index[current]
        pos_previous = vocab_to_index[previous]
        co_occurence_matrix[pos_current][pos_previous] = count
    co_occurence_matrix = np.matrix(co_occurence_matrix)
    return co_occurence_matrix,vocab_to_index

# 生成共生矩阵
sentences = [['I','love','nlp'],['I','love','to','learn'],['nlp','is','future'],['nlp','is','cool']]
merged = list(itertools.chain.from_iterable(sentences))
vocab_to_index = {word:i for i, word in enumerate(list(set(merged)))}
matrix = co_occurence_matrix(merged)
CoMatrixFinal = pd.DataFrame(matrix[0],index=vocab_to_index,columns=vocab_to_index)
print(CoMatrixFinal)   
```
## 方法3-5 使用哈希向量器
问题：理解并生成一个哈希向量器
实现：
```Python
# 哈希向量器是一种内存高效，缺点是单向，一旦向量化，则无法检索其特征
# pip install scikit-learn
from sklearn.feature_extraction.text import HashingVectorizer
text =['The quick brown forx jumped over the lazy dog.']
vectorizer = HashingVectorizer(n_features=10)
vector = vectorizer.transform(text)
print(vector.shape)
print(vector.toarray())
```
## 方法3-6 使用词频-逆文档频率将文本转化为特征
## 方法3-7 实现词嵌入
## 方法3-8 实现fastText

# 第4章 高级自然语言处理
## 方法4-1 提取名词短语
## 方法4-2 查找文本之间的相似度
## 方法4-3 词性标注
## 方法4-4 从文本中提取实体
## 方法4-5 从文本中提取主题
## 方法4-6 文本分类
## 方法4-7 情感分析
## 方法4-8 消除文本二义性
## 方法4-9 语音转换文本
## 方法4-10 文本转换语音
## 方法4-11 语言翻译

# 第5章 自然语言处理的行业应用
## 方法5-1 消费者投诉分类
## 方法5-2 实现情感分类
## 方法5-3 应用文本相似度函数
## 方法5-4 文本数据总结
## 方法5-5 文档聚类
## 方法5-6 搜索引擎中的自然语言处理

# 第6章 基于深度学习的自然语言处理
## 方法6-1 利用深度学习进行信息检索
## 方法6-2 使用深度学习对文本进行分类
## 方法6-3 对邮件使用长短时记忆预测下一个单词/序列







