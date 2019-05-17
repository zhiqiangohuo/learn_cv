# pytorch作业

<https://pytorch.apachecn.org/docs/1.0/#/blitz_neural_networks_tutorial>

### 1. pytorch实现一个分类

使用pytorch搭建神经网络的过程，训练过程

![1557891619962](https://github.com/zhiqiangohuo/pytorch/blob/master/images/1557891619962.png)



![1557892316763](https://github.com/zhiqiangohuo/pytorch/blob/master/images/1557892316763.png)

代码链接：[https://github.com/zhiqiangohuo/pytorch/blob/master/Picture%20classification.py](https://github.com/zhiqiangohuo/pytorch/blob/master/Picture classification.py)

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



