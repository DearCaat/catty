# Welcom to the Catty
# Please change the timm default transform code, test_center_crop，默认这里是用的短边，然后保持比例的scale，如果想不保持比例就要对代码进行修改，不然就不行
# shell中添加bool要用True，.yaml文件中用true

# 目前新的框架还在搭建中
任何一个训练脚本，应该被拆分成dataloader，criterion，models，engine。
## 20220511,目前暂时不考虑多GPU的训练，这是下一步计划。主要的担心在于多个模型训练时，怎么进行一个分布式的处理

## 20220711，目前正在整合前面几篇论文，同时优化框架

## 20220805，yacs的config包有严格的类型检测，考虑是不是自己在其基础上进行一点改动

# 文件说明
## configs
该目录下以项目划分，注意：每一个项目都应该有一个defualt配置文件，以免在不同配置文件下缺失配置的情况