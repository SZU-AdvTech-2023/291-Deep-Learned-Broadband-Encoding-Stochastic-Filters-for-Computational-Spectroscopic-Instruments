文件结构如下：

1. data文件夹：包含测试数据，训练数据由于体积过大未提交；
2. nets文件夹：已训练网络参数；
3. config.yml：训练配置文件；
4. HybridNet.py：网络设计文件；
5. NoiseLayer.py：测试过程噪声添加层；
6. test_hybnet.py：测试代码；
7. tmm_torch.py：网络设计文件；
8. train_hybnet_plus.py：训练代码；



主要环境及版本如下：

|     Name     |   Version    |
| :----------: | :----------: |
|    numpy     |    1.24.1    |
|    pandas    |    2.1.4     |
|    python    |   3.10.13    |
| scikit-learn |    1.3.0     |
|    scipy     |    1.11.4    |
|    torch     | 2.1.1+cu118  |
|  torchaudio  | 2.1.1+cu118  |
| torchvision  | 0.16.1+cu118 |



测试代码：python test_hybnet.py .\nets\hybnet\20231222_124336

训练代码：python train_hybnet_plus.py

注：未上传训练数据。





测试结果：

![image-20240111181738899](https://gitee.com/chen-weipeng-Host/typora/raw/master/img/image-20240111181738899.png)