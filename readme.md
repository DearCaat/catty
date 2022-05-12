# Please change the timm default transform code, test_center_crop，默认这里是用的短边，然后保持比例的scale，如果想不保持比例就要对代码进行修改，不然就不行
# shell中添加bool要用True，.yaml文件中用true

# 目前新的框架还在搭建中
任何一个训练脚本，应该被拆分成dataloader，criterion，models，engine。
## 20220511,目前暂时不考虑多GPU的训练，这是下一步计划。主要的担心在于多个模型训练时，怎么进行一个分布式的处理

