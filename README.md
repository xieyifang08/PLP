# PLP
对抗机器学习总项目,以数据集分类，包三个子项目，子项目代码结构类似，分别使用cifar10、imagenet、mnist数据集

## cifar10
1. attacks/
   1. creat_adv_cifar10: 生成对抗样本，用于测试防御模型的效果，主要使用deepfool包中的攻击api
   2. fsy111111、nsgaii、advdigitalmark：数字水印攻击实验。
2. defenses/
   1. cifar10_**:使用各防御策略训练防御模型的代码
   2. evaluate_**:模型防御力测试代码
3. models/ 防御实验中涉及到的模型
4. process/ 数据增强代码
5. visualize/ 
   1. gradcam_*: grad-cam热力图代码