### 功能描述

在界面上点击某个**按钮**, 后端执行相似度计算，指定前端 **token view** 的显示内容的不同token地颜色

### 输入输出格式

**输入**

1. comment token attention：读取 `/home/yiming/cophi/training_dynamic/gcb_tokens_visactor/dataset/features/nl_attention_{i}.npy`
2. code token attention：读取 `/home/yiming/cophi/training_dynamic/gcb_tokens_visactor/dataset/features/code_attention_{i}.npy`
3. comment token embedding & code token embedding：读取 `/home/yiming/cophi/training_dynamic/gcb_tokens_visactor/dataset/features/train_data_{i}.npy`，
4. 读取 /home/yiming/cophi/training_dynamic/gcb_tokens_visactor/code_index.json 以及 /home/yiming/cophi/training_dynamic/gcb_tokens_visactor/comment_index.json 获取comment length和code length
5. 根据comment length和code length，进行切割，得到comment token embedding & code token embedding
6. 读取/home/yiming/cophi/training_dynamic/gcb_tokens_visactor/full_text.json得到tokens信息进行后续的染色计算

**输出**

1. html的结果，包含了每个token的展示颜色和展示大小

### 后端实现

1. 读取输入的文件
读取attention，计算得到attention比较高的token坐标，只对这些token进行alignment计算
2. 计算相似度
计算两两token之间的相似度矩阵
3. 根据相似度，确定每个token的展示颜色和展示大小
根据相似度矩阵进行聚类，把相似度满足阈值的tokens聚成一个类别
4. 生成html文件
根据得到的计算类别和attention大小，确定每个token的展示颜色和展示大小

### 前端实现

TODO