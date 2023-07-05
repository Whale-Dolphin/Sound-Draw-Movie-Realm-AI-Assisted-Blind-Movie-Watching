# 声绘影境——SoundDrawMovieRealm-AI-AssistedBlindMovieWatching
近些年来，许多城市和志愿者都致力于给盲人朋友组织观看电影的活动，他们从四面八方赶往城里为了看专门给他们讲述的电影，让看不见的他们也能感受到一份影视的魅力和张力。结合本次Hackday和AIGC+Accesiible的主题，我们希望利用多种AI技术便捷的生成电影的讲述，让盲人朋友不再需要在路上花费几十分钟，也不需要担心自己没有时间，随时随地能“看到”自己想看的电影真正让盲人群体能通过AI技术ACCESS到影视的魅力，让他们的生活中有更多的娱乐，能见识到更美丽的世界。
本次项目涉及到了计算机视觉、自然语言和计算机语音等多个人工智能领域方向，使用了slowfast、t5_base等多个模型，希望能达到更好的效果
## 技术路线
### 1.cv部分
cv部分主要的作用就是获取视频中的场景、人物信息、人物动作和人物标定，由于时间不足和能力有限我们只完成了前两部分的构建，分别使用了t5-base模型和slowfast模型。我们设想的技术路线中人物信息识别可以用yolo来做人物标定可以使用基于opencv的人脸识别来做，通过相对位置来标定每个人做了什么并通过人脸作为唯一标识符进行标定。刚刚展示的是下一步需要用的csv格式的文件，其中第一列表示获取的帧数，也就是这是SLOWFAST处理的第几个场景，后面一列是缓冲帧数，再后面是识别到的框的位置坐标和大小，最后一列是每个动作的置信度和编号。在我们的设想中还会有基于yolo的人物形象识别和基于opencv的人脸识别，通过slowfast识别框的位置可以将这些信息绑定在一起作为唯一一个人使用，这样可以展现丰富的效果。
### 2.nlp部分
nlp部分我们调用了openai的api来进行自然语言的转换，我们也注意到了清华的大语言模型ChatGLM2可以做本地的部署，但碍于时间问题和能力有限并没有完成，日后有机会可以本地部署并做一些讲故事方面的专项训练。我们使用的nlp调用了chatgpt3.5模型，通过导入本地的预设让其能获取标准输入并输出故事。它从本地读取slowfast生成的.csv文件并转换成文本和csv后存储在本地。我们希望以后能进行本地部署并进行部分的调参使之能更好的贴合我们的项目。
### 3.tts部分
这一部分本来是想做一个基于VITS的本地部署的，但由于本来负责这一部分的是电气的同学，她实在是太忙了，加上不太熟悉python和tts，我们现在现在只能通过微软的api制作的应用来实现和完成，日后有时间我一定会补上（XD我觉得这部分不算难才给的时间少的同学QAQ）我们也希望tts的部分日后能根据图像的输出文本和视频中本就有的声音来增加部分感情使讲述的电影更有感染力和张力。
## 环境配置及项目配置
### 环境配置
该项目的环境配置极其复杂所有需要的库已经打包并存在了environment.yml中，可以使用anaconda自带的命令进行一键导入。本项目基于AI的部分都是基于cuda运行的，如果想移植到mac或其他没有cuda的设备上需要更改torch和torchvision的版本。另外需要注意的是该项目会和torch2.0.0以上的版本有冲突，请下载2.0.0以下的版本。本项目的本地运行是基于3070和cuda12.2（向下兼容），仅作为参考。安装完所需的python库后需要进行一些列包的安装所有包均已放在项目的一级目录中，所需要的包有cocoapi、detectron2、fvcore、fairscale、pytorchvideo除了cocoapi以外其他进入二级目录并运行
```
    python setupo.py install
```
即可，其中cocoapi需要进入三级目录PythonAPI并运行上面的命令行指令。同时你也可以直接通过下面的方式从github获取并安装
```
    pip install 'git+https://github.com/facebookresearch/fvcore'
    pip install -U 'git+https://github.com/facebookresearch/fvcore.git' 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
    pip install -e detectron2_repo
```
其中pytorchvideo并不是官方要求安装的包，是在项目配置过程中出现了问题我们添加上去的
最后运行下面指令或直接将slowfast添加到环境变量后建立slowfast
```
    export PYTHONPATH=/path/to/SlowFast/slowfast:$PYTHONPATH
    cd SlowFast
    python setup.py build develop
```
### 项目配置
项目配置相对简单，首先需要在一级目录中的confg.py中填写视频的路径和输出的路径，其次需要进入到/Slowfast/demo/AVA/SLOWFAST_32x2_R101_50_50.yaml中填写所需路径，为什么不能使用config.py我在该文件中已进行说明QAQ。然后需要在https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/ava/SLOWFAST_32x2_R101_50_50.pkl 下载slowfast模型并放入/Slowfast/demo/model文件夹中，之后在下载https://huggingface.co/t5-base/resolve/main/pytorch_model.bin 下载t5-base的权重文件并放入ImageCaptioning\t5_base中。最后你需要将自己的openaiAPI填写到NLP\nlp.py 至此，所有的配置都已进行完成
## 结语
由于项目的队伍是四个大一学生并且有两个人很忙，加上技术不太熟练，技术力有限，所以本项目严格来说还是一个半成品，我非常希望能完成yolo、opencv、nlp、vits的部分使他成为一个更完善的项目，能够更好的帮助盲人群体会到新鲜的影视体验和更丰富的娱乐活动。感谢联创提供给我和我们这个平台和机会让我们能提出这个想法并完成了最初的工作。
## 附件
演示Demo视频：https://www.aliyundrive.com/s/VD1cpC4v4qP 提取码：28ox