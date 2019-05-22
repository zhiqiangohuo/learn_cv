# 图像基本处理

##### 1. 图像的基本处理
   ###### 灰度图像
   ![1557818649582](https://github.com/zhiqiangohuo/pytorch/blob/master/images/1557818649582.png)
  

   ###### 图像格式转换

```python
from PIL import Image
import os
filelist = os.listdir('images/')
print(filelist)
for infile in filelist:
    outfile = os.path.splitext(infile)[0] + ".jpeg"
    print(outfile)
    print(infile)
    if infile != outfile:
        Image.open('images/'+infile).save(outfile)
```

输出：

![1557822176079](https://github.com/zhiqiangohuo/pytorch/blob/master/images/1557822176079.png)

###### 1.1.2 创建略缩图

![1557822676400](https://github.com/zhiqiangohuo/pytorch/blob/master/images/1557822676400.png)

```python
from PIL import Image
pil_im = Image.open('empire.jpg').convert('L')
pil_im.thumbnail((128,128))
pil_im.show()
```

###### 1.1.3 复制和粘贴图像区域

![1557822859897](https://github.com/zhiqiangohuo/pytorch/blob/master/images/1557822859897.png)

```python
from PIL import Image
pil_im = Image.open('empire.jpg')
box = (100,100,400,400)
region = pil_im.crop(box)
region = region.transpose(Image.ROTATE_180)
pil_im.paste(region,box)
pil_im.show()
```

###### 1.1.4调整尺寸和旋转

resize

![1557823151674](https://github.com/zhiqiangohuo/pytorch/blob/master/images/1557823151674.png)

```python
from PIL import Image
pil_im = Image.open('empire.jpg')
out = pil_im.resize((128,128))
out.show()
```

旋转

![1557823213290](https://github.com/zhiqiangohuo/pytorch/blob/master/images/1557823213290.png)

```python
from PIL import Image
pil_im = Image.open('empire.jpg')
#out = pil_im.resize((128,128))
out = pil_im.rotate(45)
out.show()
```

###### 1.2.1 绘制图像点和线

![1557825399602](https://github.com/zhiqiangohuo/pytorch/blob/master/images/1557825399602.png)

```python
from PIL import Image
from pylab import *
# 读取图像到数组中
im = array(Image.open('empire.jpg'))
# 绘制图像
imshow(im)
# 一些点
x = [100,100,400,400]
y = [200,500,200,500]
# 使用红色星状标记绘制点
plot(x,y,'r*')
# 绘制连接前两个点的线
plot(x[:2],y[:2])
# 添加标题，显示绘制的图像
title('Plotting: "empire.jpg"')
show()
```

###### 1.2.2  图像轮廓和直方图

![1557825724539](https://github.com/zhiqiangohuo/pytorch/blob/master/images/1557825724539.png)

![1557825966159](https://github.com/zhiqiangohuo/pytorch/blob/master/images/1557825966159.png)

```python
from PIL import Image
from pylab import *
# 读取图像到数组中
im = array(Image.open('empire.jpg').convert('L'))
# 新建一个图像
figure()
# 不使用颜色信息
gray()
# 在原点的左上角显示轮廓图像
contour(im, origin='image')
axis('equal')
axis('off')
# 绘制直方图
figure()
hist(im.flatten(),128)
show()
```

###### 1.2.3 交互式标注

![1557826419582](https://github.com/zhiqiangohuo/pytorch/blob/master/images/1557826419582.png)

```python
from PIL import Image
from pylab import *
im = array(Image.open('empire.jpg'))
imshow(im)
print("Please click 3 points")
# 获取用户点击坐标 可以[x,y]
x = ginput(3)
print("you clicked",x)
show()
```

###### 1.3.1 图像数组表示

输出：

```python
(1200, 1920, 3) uint8
(1200, 1920) float32
```

```python
im = array(Image.open('empire.jpg'))
print (im.shape, im.dtype)
im = array(Image.open('empire.jpg').convert('L'),'f')
print (im.shape, im.dtype)
```

tip:

im[i,:] = im[j,:]   # 将第 j 行的数值赋值给第 i 行 
im[:,i] = 100 # 将第 i 列的所有数值设为 100
im[:100,:50].sum() # 计算前 100 行、前 50 列所有数值的和
im[50:100,50:100] # 50~100 行，50~100 列（不包括第 100 行和第 100 列）
im[i].mean() # 第 i 行所有数值的平均值
im[:,-1] # 最后一列
im[-2,:] (or im[-2]) # 倒数第二行

###### 1.3.2灰度变换

![1557827505327](https://github.com/zhiqiangohuo/pytorch/blob/master/images/1557827505327.png)

```python
from PIL import Image
from numpy import *
from pylab import *
im = array(Image.open('empire.jpg').convert('L'))
im2 = 255 - im # 对图像进行反相处理
im3 = (100.0/255) * im + 100 # 将图像像素值变换到 100...200 区间
im4 = 255.0 * (im/255.0)**2 # 对图像像素值求平方后得到的图像
imshow(im2)
show()
imshow(im3)
show()
imshow(im4)
show()
```

###### 1.3.4直方图均衡化

处理前

![1557838382030](https://github.com/zhiqiangohuo/pytorch/blob/master/images/1557838382030.png)

处理后

![1557838195684](https://github.com/zhiqiangohuo/pytorch/blob/master/images/1557838195684.png)



```python
from PIL import Image
from numpy import *
from pylab import *
import imtool
im = array(Image.open('timg.jpg').convert('L'))
imshow(im)
figure()
hist(im.flatten(),128)
show()
im2,cdf = imtool.histeq(im)
imshow(im2)
#save('im2.jpg')
print(im2,cdf)
figure()
hist(im2.flatten(),128)
show()
```

###### 1.3.6 图像的主成分分析

![1557838569262](https://github.com/zhiqiangohuo/pytorch/blob/master/images/1557838569262.png)

```python
from numpy import *
from PIL import Image
from pylab import *
import imtool
import pickle
import pca
im2 = 'imlist/b.jpg'

imlist = [im2,im2,im2,im2,im2,im2,im2,im2]

im = array(Image.open(imlist[0]))
imshow(im)
m,n = im.shape[0:2]
imnbr = len(imlist)
immatrix = array([array(Image.open(im).convert('L')).flatten() for im in imlist],'f')
print(imlist)
# 执行PCA操作
V,S,immean = imtool.pca(immatrix)
# 显示一些图像
figure()
gray()
subplot(2,4,1)
imshow(immean.reshape(m,n))
for i in range(7):
    subplot(2,4,i+2)
    imshow(immean.reshape(m, n))
    imshow(V[i].reshape(m,n))
show()
```

###### 1.4.1图像模糊

![1557838712805](https://github.com/zhiqiangohuo/pytorch/blob/master/images/1557838712805.png)

```python
from PIL import Image
from numpy import *
from pylab import *
from scipy.ndimage import filters
# im = array(Image.open('test.jpg').convert('L'))
# im2 = filters.gaussian_filter(im,2)
# imshow(im2)
# show()
im = array(Image.open('timg.jpg'))
im2 = zeros(im.shape)
for i in range(3):
    im2[:,:,i] = filters.gaussian_filter(im[:,:,i],5)
im2 = uint8(im2)
imshow(im2)
show()
```

###### 1.4.2图像导数

![1557838777429](https://github.com/zhiqiangohuo/pytorch/blob/master/images/1557838777429.png)

```python
from PIL import Image
from numpy import *
from scipy.ndimage import filters
from pylab import *
im = array(Image.open('test.jpg').convert('L'))
# Sobel 导数滤波器
# imx = zeros(im.shape)
# filters.sobel(im,1,imx)
# imy = zeros(im.shape)
# filters.sobel(im,0,imy)
sigma = 5 # 标准差
imx = zeros(im.shape)
filters.gaussian_filter(im, (sigma,sigma), (0,1), imx)
imy = zeros(im.shape)
filters.gaussian_filter(im, (sigma,sigma), (1,0), imy)
magnitude = sqrt(imx**2+imy**2)
for im in [imx,imy,magnitude]:
    imshow(im)
    show()
```

###### 1.4.3 二值图像

![1557839868740](https://github.com/zhiqiangohuo/pytorch/blob/master/images/1557839868740.png)

```python
from scipy.ndimage import measurements,morphology
# 载入图像,然后使用阈值化操作,以保证处理的图像为二值图像
from PIL import Image
from numpy import *
im = array(Image.open('3.jpg').convert('L'))
im = 1*(im<128)
labels, nbr_objects = measurements.label(im)
print ("Number of objects:", nbr_objects,labels)
```

###### 1.4.4 图片保存

```python
from scipy.misc import imsave
from PIL import Image
im = Image.open('1.jpg')
imsave('test.jpg',im)
```

###### 1.5 图片去噪

![1557840172015](https://github.com/zhiqiangohuo/pytorch/blob/master/images/1557840172015.png)

```python
from PIL import Image
from pylab import *
import rof
im = array(Image.open('2.jpg').convert('L'))
U,T = rof.denoise(im,im)
figure()
gray()
imshow(U)
axis('equal')
axis('off')
show()
```

###### 2.1 Harris角点检测器

![1557840339173](https://github.com/zhiqiangohuo/pytorch/blob/master/images/1557840339173.png)

![1557840504939](https://github.com/zhiqiangohuo/pytorch/blob/master/images/1557840504939.png)

```python
from PIL import Image
from numpy import *
from pylab import *
import harris
im = array(Image.open('ca.jpg').convert('L'))
harrisim = harris.compute_harris_response(im)
imshow(harrisim)
show()
filtered_coords = harris.get_harris_points(harrisim,6)
imshow(filtered_coords)
show()
harris.plot_harris_points(im, filtered_coords)
```

###### 图像间寻找对应点
```python
from numpy import *
from PIL import Image
from pylab import *
import harris

wid = 5
im1 = array(Image.open('sw.jpg').convert('L'))
im2 = array(Image.open('sw.jpg').convert('L'))
harrisim = harris.compute_harris_response(im1,5)
filtered_coords1 = harris.get_harris_points(harrisim,wid+1)
d1 = harris.get_descriptors(im1,filtered_coords1,wid)
harrisim = harris.compute_harris_response(im2,5)
filtered_coords2 = harris.get_harris_points(harrisim,wid+1)
d2 = harris.get_descriptors(im2,filtered_coords2,wid)
print ('starting matching')
matches = harris.match_twosided(d1,d2)
figure()
gray()
harris.plot_matches(im1,im2,filtered_coords1,filtered_coords2,matches)
show()
```
![1557840689141](https://github.com/zhiqiangohuo/pytorch/blob/master/images/1557840689141.png)

###### 2.2 SIFT检测兴趣点

![1557882842712](https://github.com/zhiqiangohuo/pytorch/blob/master/images/1557882842712.png)

```python
from pylab import *
from PIL import Image
import sift

imname = '3.jpg'
im1 = array(Image.open(imname).convert('L'))
sift.process_image(imname,'empire.sift')

l1,d1 = sift.read_features_from_file('empire.sift')
figure()
print("1")
gray()
print("2")
sift.plot_features(im1,l1,circle=True)
print("3")
imshow(im1)
print("4")
show()
```

![1557886030643](https://github.com/zhiqiangohuo/pytorch/blob/master/images/1557886030643.png)

![1557888677446](https://github.com/zhiqiangohuo/pytorch/blob/master/images/1557888677446.png)

![1557889316369](https://github.com/zhiqiangohuo/pytorch/blob/master/images/1557889316369.png)

```python

```
# pytorch作业

<https://pytorch.apachecn.org/docs/1.0/#/blitz_neural_networks_tutorial>

### 1. pytorch实现一个分类

使用pytorch搭建神经网络的过程，训练过程

![1557891619962](https://github.com/zhiqiangohuo/pytorch/blob/master/images/1557891619962.png)



![1557892316763](https://github.com/zhiqiangohuo/pytorch/blob/master/images/1557892316763.png)

代码链接：[https://github.com/zhiqiangohuo/pytorch/blob/master/Picture%20classification.py](https://github.com/zhiqiangohuo/pytorch/blob/master/Picture%20classification.py)

### 2.迁移学习处理图像

1. 数据集`hymenoptera_data`
2. 迁移模型`resnet18-5c106cde.pth`

```python
model_ft = models.resnet18(pretrained=True) # 加载原始模型
```

![1557902308292](https://github.com/zhiqiangohuo/pytorch/blob/master/images/1557902308292.png)

1. 训练过程

![1557902245541](https://github.com/zhiqiangohuo/pytorch/blob/master/images/1557902245541.png)

4. 分类效果

![1557902262047](https://github.com/zhiqiangohuo/pytorch/blob/master/images/1557902262047.png)

5. 代码链接


https://github.com/zhiqiangohuo/pytorch/blob/master/rg_bee_ant.py




### 3. 传统方法实现行人检测

![1557976427933](https://github.com/zhiqiangohuo/pytorch/blob/master/images/1557976427933.png)

代码链接：<https://github.com/zhiqiangohuo/pytorch/tree/master/PythonApplication6>



### 4. 人脸识别

代码链接：<https://github.com/zhiqiangohuo/pyqt_face_rg>

 	1. 效果展示
      	1.人脸录入
      	
![人脸录入](https://github.com/zhiqiangohuo/pytorch/blob/master/images/人脸录入.gif)
	

2.人脸识别表情年龄识别
	
	
![识别](https://github.com/zhiqiangohuo/pytorch/blob/master/images/识别.gif)

![年龄](https://github.com/zhiqiangohuo/pytorch/blob/master/images/年龄.gif)



检测模块使用MTCNN进行人脸检测

```python
https://github.com/zhiqiangohuo/pyqt_face_rg/blob/master/mtcnn.py
```

人脸对齐方法:

仿射变换

- 识别出人脸，并标出关键点。比如通过双眼两个标点进行变换。
- 方法1 `cv2.getAffineTransform`
- 方法2 `AlignFace类中的getAffineTransform() API`
- 参考链接 <https://zhuanlan.zhihu.com/p/61343643>

特征点提取

- <https://zhuanlan.zhihu.com/p/39499030>

匹配

- <https://zhuanlan.zhihu.com/p/39499030>



### 5.C++加载pytorch模型

1.将Pytorch模型转化为Torch script

```python
import torch
 
class MyModule(torch.jit.ScriptModule):
    def __init__(self, N, M):
        super(MyModule, self).__init__()
        self.weight = torch.nn.Parameter(torch.rand(N, M))
 
    @torch.jit.script_method
    def forward(self, input):
        if input.sum() > 0:
          output = self.weight.mv(input)
        else:
          output = self.weight + input
        return output
 
my_script_module = MyModule()
traced_script_module.save("model.pt")
```

2. 使用C++加载脚本模块

```c++

#include <torch/script.h> // One-stop header.
 
#include <iostream>
#include <memory>
 
int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    return -1;
  }
 
  // Deserialize the ScriptModule from a file using torch::jit::load().
  std::shared_ptr<torch::jit::script::Module> module = torch::jit::load(argv[1]);
 
  assert(module != nullptr);
  std::cout << "ok\n";
}

```



![1557934713563](https://github.com/zhiqiangohuo/pytorch/blob/master/images/1557934713563.png)

4. 安装Libtorch

5. 执行下面命令
```
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
make
```

产生如下目录结构

![1557970754520](https://github.com/zhiqiangohuo/pytorch/blob/master/images/1557970754520.png)

6. 执行`./example-app model.pt`

```python
# 执行前
tensor([-0.2698, -0.0381,  0.4023, -0.3010, -0.0448], grad_fn=<SliceBackward>)
# 输出
-0.2698 -0.0381  0.4023 -0.3010 -0.0448
[ Variable[CPUFloatType]{1,5} ]
```




