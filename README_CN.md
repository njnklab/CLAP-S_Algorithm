````markdown
# 「双层网络优化控制」的CLAP-S算法实现

许多现实世界的复杂系统可以被建模为*多层*网络，其中每一层代表相同实体之间的一组不同交互。控制此类系统至关重要，但对每一层分别施加控制通常会产生一组冗余的驱动节点，从而增加成本和复杂性。

本代码库包含论文**《Optimized Control of Duplex Networks》**的官方实现。我们通过为双层网络构建通用最小驱动节点集（MinUDS）问题来解决这一问题，该问题旨在寻找能同时控制两层的最小驱动节点集。我们提出了一种新颖的算法，**基于最短可控性的分层路径搜索（CLAP-S）**，该算法能有效导航控制配置的组合搜索空间。通过引入基于可控性的分层路径（CLAP），该算法迭代地重新调整每层的最小驱动集（MDS），以最大化它们的重叠。

本代码库提供了CLAP-S算法、所有基准算法的源代码，以及用于复现论文中图表的脚本。

## 文件结构

项目组织如下：

```
.
├── assets/
│   ├── log/                  # 用于调试和跟踪的日志文件
│   ├── net/                  # 网络数据集
│   │   ├── real/             # 真实世界网络数据集
│   │   ├── synthetic/        # 模型生成的合成网络
│   │   └── test/             # 用于快速实验的测试网络
│   └── result/               # 实验结果
│       ├── figure/           # 用于可视化的图表
│       └── *.csv             # CSV格式的原始结果
├── utils/
│   ├── generator.py          # 生成合成网络（ER、BA、SF）
│   ├── plot.py               # 用于绘图和可视化的辅助函数
│   └── utils.py              # 实用函数（例如，读取网络）
├── scripts/
│   ├── experiments.ipynb     # 用于运行实验的Jupyter notebook
│   ├── real_networks.ipynb   # 用于在真实世界网络上进行实验的notebook
│   └── results.ipynb         # 用于可视化和分析结果的notebook
├── config.py                 # 项目设置的配置文件
├── main.py                   # 运行和比较算法的主脚本
├── matching.py               # 所有匹配算法的核心实现
└── README.md                 # 本文件
```

## 快速入门

### 先决条件

- Python 3.8+
- 所需的包可以通过pip安装：
  ```bash
  pip install networkx numpy pandas matplotlib
  ```
- 对于`ILP_exact`基准算法，需要一个额外的求解器。您可以安装`ortools`（推荐）或`pulp`：
  ```bash
  pip install ortools
  # 或
  pip install pulp
  ```

### 运行代码

您可以通过执行`main.py`来运行一个简单的测试用例，以比较所有已实现的算法：

```bash
python main.py
```

该脚本将：
1. 生成一个合成的双层网络。
2. 运行RSU、CLAP-S、CLAP-G和ILP算法。
3. 打印每种算法的结果和执行时间。

您可以在`main.py`的底部修改网络参数（节点数`n`，平均度`k`）。

## 如何使用

### 1. 读取或生成网络

#### 生成合成网络
您可以使用`utils/generator.py`中的生成器来创建合成网络。以下示例展示了如何生成一个双层Erdős-Rényi（ER）网络。

```python
from utils.generator import ERGenerator

# 初始化一个网络生成器，节点数为1000，平均度为4
generator = ERGenerator(n=1000, k=4)

# 生成一个2层网络
graphs = generator.generate_networks(num_layers=2)
```
支持的生成器包括`ERGenerator`（Erdős-Rényi）、`BAGenerator`（Barabási-Albert）和`SFGenerator`（Scale-Free）。

#### 从文件读取网络
您可以使用`read_network`实用工具从边列表文件中读取网络。

```python
from utils.utils import read_network

# 从文件中读取一个有100个节点的网络
graph = read_network(file_path="assets/net/real/SomeNet.txt", n=100)
```

> **注意：** `assets/net/test`目录包含我们论文中使用的真实世界网络数据的副本，以及具有不同节点数和平均度的各种合成测试网络。我们鼓励您使用这些文件进行测试。例如：
> ```bash
> python main.py --files assets/net/test/Cannes_MT.txt assets/net/test/Cannes_RT.txt -n 36
> ```
> **重要提示：** 当从文件读取任何网络时，您**必须**使用`-n`参数指定总节点数`n`。这对于确保各层之间的数据一致性至关重要，特别是对于可能在某一层中是孤立的（没有边）但在另一层中存在的节点。未能指定正确的总`n`可能导致不正确的图结构和运行时错误，因为孤立节点本身就是驱动节点，必须包含在分析中。

### 2. 运行匹配算法

所有算法的核心逻辑都在`matching.py`中。要运行一个算法，您首先需要为每个网络层创建`Matching`对象，然后创建一个`MultiMatching`对象来管理它们。

```python
import copy
from matching import Matching, MultiMatching

# 假设'graphs'是networkx.Graph对象的列表
matchings = []
for g in graphs:
    matching = Matching(g)
    matching.HK_algorithm()  # 查找初始最大匹配
    matchings.append(matching)

# 创建一个MultiMatching对象用于算法比较
clap_s_matching = MultiMatching(matchings)
rsu_matching = copy.deepcopy(clap_s_matching)
glde_matching = copy.deepcopy(clap_s_matching)

# 运行不同的算法
print("Running CLAPS...")
clap_s_matching.CLAPS()

print("Running RSU...")
rsu_matching.RSU()

print("Running CLAPG...")
glde_matching.CLAPG()
```

`MultiMatching`中可用的算法有：
- `CLAPS()`: 我们提出的核心算法。
- `CLAPG()`: 我们提出的贪心算法。
- `RSU()`: 一种基准算法。
- `ILP_exact()`: 一种使用整数线性规划的精确算法。

### 3. 算法比较

`main.py`脚本提供了一个模板，用于在同一网络上比较不同算法的性能和执行时间。您可以根据自己的需要调整其`test()`函数。

## 实验复现和可视化

`scripts/`目录包含Jupyter notebooks，用于复现我们论文中的实验并可视化结果。

- **`experiments.ipynb`**: 该notebook包含在合成网络上运行实验的代码，可变参数包括网络大小、密度和重叠度。它将结果保存到`assets/result/`目录。
- **`real_networks.ipynb`**: 该notebook专门用于在`assets/net/real/`中的真实世界网络数据集上运行实验。
- **`results.ipynb`**: 使用此notebook加载结果目录中的`.csv`文件，并生成论文中呈现的图表。它依赖于`utils/plot.py`中的绘图函数。

````