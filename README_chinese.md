# 运行手册：Windows / Linux

## 1. 独立运行条件

`research_mvp` 可以独立于历史研究目录运行：只需要本目录的源代码、数据集、Python依赖和初始化权重，不依赖父目录的历史Python脚本、日志、论文或旧实验权重。

支持的流程只有：

```text
yolov8s.pt预训练初始化 → 全量train/val训练 → 本次best.pt
→ test按时间顺序推理 → Clean → Interpolate → 三阶段统一评价
```

本次smoke只有8张训练图、4张验证图、24张测试图，使用320输入尺寸并仅训练1 epoch，不能代表正常训练效果。使用全量train/val、1280输入尺寸和50或100 epochs，是后续新实验的设置；增加数据和训练轮数不保证必然改善，应根据验证集和独立测试集结果判断。

**本手册中的全量训练命令没有在本次操作中执行。** 已有验证仅为Windows CPU上的小规模1 epoch流程；Linux步骤按现有入口和路径机制编写，尚未实际验证。

## 2. 下载与摆放数据集

数据集地址：

[Badminton360-Shuttlecock — Hugging Face](https://huggingface.co/datasets/aissur320/Badminton360-Shuttlecock)

从上述地址下载数据集，解压后将数据目录命名为 **`project`**，与 **`research_mvp` 放在同一目录下**：

```text
your_workspace/
├── project/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── labels/
│       ├── train/
│       ├── val/
│       └── test/
└── research_mvp/
    ├── mvp.py
    ├── fetch_weights.py
    ├── configs/
    ├── src/
    ├── tests/
    └── ...
```

确保`images`和`labels`直接位于`project/`下，没有多套一层同名目录。训练、验证和测试图像必须具有对应同名TXT标签；背景图像保留空标签文件。

外部`project/`是源数据。下面的`prepare-data`会把全部数据复制到`research_mvp/data/project/`，并仅在副本中将已确认的类别0、3统一为羽毛球类别0；不改变原始数据。需要为副本预留磁盘空间。

## 3. Windows：安装环境

使用PowerShell，先安装可正常运行的Python 3.12。在包含`project/`和`research_mvp/`的目录打开终端，再进入MVP：

```powershell
cd research_mvp
python -m venv .venv
```

每台设备重新建立自己的`.venv`，不要复用从其他设备复制来的虚拟环境。下列命令直接指定虚拟环境中的Python，不需要激活环境。

### CPU方式

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-cpu.txt
```

全量1280尺寸训练在CPU上可能需要较长时间。若设备有可用的NVIDIA GPU，可选择下述GPU方式。

### NVIDIA GPU方式

在新虚拟环境中，根据驱动和平台选择兼容的PyTorch CUDA安装命令：

[PyTorch官方历史版本安装说明](https://pytorch.org/get-started/previous-versions/)

选用相互兼容的`torch==2.7.1`与`torchvision==0.22.1`，安装后执行：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

确认最后输出为`True`后，才能在第5节配置`device: "0"`。CPU与GPU安装方式择一；不要在需要GPU的环境中先安装`requirements-cpu.txt`。

### 检查依赖并下载初始化权重

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe fetch_weights.py
```

下载后应有`weights/yolov8s.pt`。如果该文件已经存在且来源可信，跳过下载；下载器不会覆盖已有文件。

## 4. Linux：安装环境

使用Bash，准备Python 3.12及其`venv`支持。在包含`project/`和`research_mvp/`的目录打开终端；示例假定`python3`指向所需版本：

```bash
cd research_mvp
python3 --version
python3 -m venv .venv
```

### CPU方式

```bash
./.venv/bin/python -m pip install -r requirements-cpu.txt
```

### NVIDIA GPU方式

按[PyTorch官方历史版本安装说明](https://pytorch.org/get-started/previous-versions/)在新虚拟环境中安装兼容的CUDA版`torch==2.7.1`与`torchvision==0.22.1`，再执行：

```bash
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

GPU方式要求CUDA可用性检查为`True`。CPU与GPU安装方式择一。

### 检查依赖并下载初始化权重

```bash
./.venv/bin/python -m pip check
./.venv/bin/python fetch_weights.py
```

已有可信的`weights/yolov8s.pt`时跳过下载。不要把其他型号的权重重命名成`yolov8s.pt`。

## 5. 建立全量训练配置

复制已有完整示例为自己的本地配置。

Windows：

```powershell
Copy-Item configs/experiment.example.json configs/local.full.json
```

Linux：

```bash
cp configs/experiment.example.json configs/local.full.json
```

用文本编辑器打开`configs/local.full.json`，保存为**UTF-8无BOM**。保留其他字段，按设备设置以下字段：

| 字段 | 设置说明 |
|---|---|
| `run_name` | 新的唯一名称，例如`yolov8s_full_50e_cpu`或`yolov8s_full_50e_gpu` |
| `paths.data_root` | 保持`data/project`，供导入副本使用 |
| `paths.pretrained_weights` | 保持`weights/yolov8s.pt` |
| `paths.output_root` | 保持`outputs`，或指定MVP目录内与数据分离的新输出目录 |
| `train.epochs` | 设为`50`；需要100轮时设为`100`，示例文件原值为100 |
| `train.imgsz` | 保持`1280` |
| `train.batch` | CPU可先设为`2`；GPU按显存设置，示例原值为16 |
| `train.device` | CPU填`"cpu"`，第一块可用NVIDIA GPU填`"0"` |
| `predict.imgsz` | 保持`1280` |
| `predict.device` | 与所选设备一致：`"cpu"`或`"0"` |

配置中`epochs`、`batch`等为数字，`device`为字符串。路径字段相对于`research_mvp/`解析；`--config`后面的相对文件路径按当前工作目录解析，因此本手册始终在MVP目录执行命令。

程序在运行时会根据当前项目位置自动解析绝对路径，不包含开发者电脑的固定路径。保存到`config.resolved.json`和训练`args.yaml`的文件路径统一相对于`research_mvp/`，使用`/`分隔；`dataset.yaml`中的train/val路径相对于该YAML文件所在目录，且不设置机器相关的`path`字段。整体移动项目时，保留`data/`、`weights/`、`outputs/`之间的相对结构即可。

新下载者从示例配置开始创建自己的run。`outputs/`是被Git忽略的本地运行产物；`.venv/`和`.cache/`也不发布或跨设备复制，依赖按前述步骤在目标设备重新安装。

`source_class_map`保留`{"0": 0, "3": 0}`。如果下载数据出现其他类别ID，先核实含义，不要自行猜测映射。

**训练开始后不要修改该配置**，包括预测设备、阈值和`run_name`；后续阶段会核对它与训练时保存的配置是否一致。

## 6. Windows：全量运行

以下命令使用第5节的同一配置。每个步骤成功后再执行下一步。

```powershell
# 只读检查外部源数据，并在MVP中保存检查报告
.\.venv\Scripts\python.exe -X utf8 mvp.py audit --data-root ../project

# 全量复制并映射标签；不触发训练
.\.venv\Scripts\python.exe -X utf8 mvp.py prepare-data --source ../project --config configs/local.full.json

# 全量train/val训练；50/100 epochs必须显式带此开关
.\.venv\Scripts\python.exe -X utf8 mvp.py train --config configs/local.full.json --allow-full-train

# 读取本次训练的best.pt，按序推理完整test
.\.venv\Scripts\python.exe -X utf8 mvp.py predict --config configs/local.full.json

# 分别保存Clean和Interpolate结果
.\.venv\Scripts\python.exe -X utf8 mvp.py postprocess --config configs/local.full.json

# 对raw、Clean和Interpolate使用同一评价器
.\.venv\Scripts\python.exe -X utf8 mvp.py evaluate --config configs/local.full.json
```

## 7. Linux：全量运行

```bash
./.venv/bin/python -X utf8 mvp.py audit --data-root ../project
./.venv/bin/python -X utf8 mvp.py prepare-data --source ../project --config configs/local.full.json
./.venv/bin/python -X utf8 mvp.py train --config configs/local.full.json --allow-full-train
./.venv/bin/python -X utf8 mvp.py predict --config configs/local.full.json
./.venv/bin/python -X utf8 mvp.py postprocess --config configs/local.full.json
./.venv/bin/python -X utf8 mvp.py evaluate --config configs/local.full.json
```

`prepare-data`只需对同一个目标数据目录执行一次。如果`data/project/`已是此前成功导入并映射的完整副本，后续新实验复用它，不重复导入。若目标目录只导入了一部分，应另设新的`paths.data_root`再导入，避免覆盖混用。

不要使用`run-smoke`执行全量训练；该命令强制1 epoch并限制样本数量。全量训练使用上述分阶段命令。

## 8. 查看结果与重新实验

以`run_name: "yolov8s_full_50e_gpu"`为例：

```text
outputs/yolov8s_full_50e_gpu/
├── config.resolved.json
├── dataset_audit.json
├── manifest.json
├── training.json
├── train/
│   ├── args.yaml
│   ├── results.csv
│   └── weights/
│       ├── best.pt
│       └── last.pt
├── raw.json
├── clean.json
├── interpolated.json
└── metrics.json
```

- `best.pt`按训练过程中的val表现选取；推理只接受该run保存的best.pt及其SHA256，不使用旧smoke权重。
- `metrics.json`包含三个阶段的TP/FP/FN、P/R@0.5、AP50、mAP50–95。训练内部val指标与此统一评价器的定义不同，不直接混为同一指标。
- `manifest.json`显式保存序列ID、数字帧号及顺序。缺帧、重号、乱序或尺寸变化会报错；后处理不跨序列。
- Clean/Interpolate只使用预测和允许的帧信息；test GT只用于最终评价。train/val标签用于监督训练和验证。
- 这是采用实际图像宽度和新AP101协议的新实验，不能直接将结果称为论文复现。

每次新训练使用新的`run_name`。已有输出拒绝覆盖；改变配置后也不能继续沿用旧run做预测或评价。当前入口固定`resume=False`，未提供断点续训命令；训练中断后重新实验应更换run名称。

如果显存或内存不足，结束当前失败运行，在新配置、新run名称下减小`train.batch`。保持1280可避免把正式实验不经说明地改成smoke的320尺寸；若确需修改尺寸，应作为新的实验设置记录。

可选运行现有单元测试：

Windows：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Linux：

```bash
./.venv/bin/python -m unittest discover -s tests -v
```

`data/`、`weights/`、`outputs/`、`.venv/`、缓存和`configs/local*.json`已由MVP的`.gitignore`排除。外部同级`project/`也不要加入公开仓库；只发布MVP源代码。

本手册仅提供操作步骤，没有自动执行下载、安装、数据导入、测试或训练。
