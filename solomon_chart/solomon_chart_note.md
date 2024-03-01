
# 环境部署

```Bash
conda create --name solomon_chart
conda info -e
conda activate solomon_chart

# 按照xtuner
mkdir ~/xtuner && cd ~/xtuner
git clone https://github.com/InternLM/xtuner.git
cd xtuner
pip install -e '.[all]'

# 模型下载
# 创建一个目录，放模型文件
mkdir ~/solomon_chart/internlm2_chat_7b
cd ~/model
```

download_model.py

```Bash
import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os
model_dir = snapshot_download('internlm2-chat-7b', cache_dir='/root/model')
```

```Bash
# 创建一个微调 solomon 数据集的工作路径，进入
mkdir ~/solomon && cd ~/solomon

cd ~/solomon
ln -s /root/model/Shanghai_AI_Laboratory/internlm2-chat-7b ~/solomon/
```

# 准备Qlora数据集
```Bash
# example
# 单轮对话数据格式
[{
    "conversation":[
        {
            "system": "请你扮演哲学家亚里士多德，请以他的哲学思想和口吻回答问题。",
            "input": "xxx",
            "output": "xxx"
        }
    ]
},
{
    "conversation":[
        {
            "system": "请你扮演哲学家苏格拉底，请以他的哲学思想和口吻回答问题。",
            "input": "xxx",
            "output": "xxx"
        }
    ]
},
{
    "conversation":[
        {
            "system": "请你扮演哲学家柏拉图，请以他的哲学思想和口吻回答问题。",
            "input": "xxx",
            "output": "xxx"
        }
    ]
}]

# 多轮对话数据格式
[{
    "conversation":[
        {
            "system": "xxx",
            "input": "xxx",
            "output": "xxx"
        },
        {
            "input": "xxx",
            "output": "xxx"
        }
    ]
},
{
    "conversation":[
        {
            "system": "xxx",
            "input": "xxx",
            "output": "xxx"
        },
        {
            "input": "xxx",
            "output": "xxx"
        }
    ]
}]
```Bash

# 微调
