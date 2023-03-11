# LOWB_CNN

一个异常暴躁、巨LOW的卷积神经网络c++实现

它只能推理没有训练过程，优化巨烂，仅支持CPU

只是利用vector实现了tensor的一些基本操作、任意kernel的2d卷积（甚至没有padding）、relu、fc（也是转化成2d卷积实现）

实现了一个LENET5的Demo，用pytorch预先训练了权重文件，直接顺序存在`wb.in`中，然后递归读入，图片用Opencv读入

一张$28 \times 28$的手写数字图片要算好久，大概四五秒

