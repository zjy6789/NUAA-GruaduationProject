# 无人集群协同对抗仿真系统

基于 **PySC2 / StarCraft II** 与 **HARL（Heterogeneous-Agent Reinforcement Learning）** 的多智能体强化学习仿真项目，用于构建无人机集群协同对抗、训练、评估与可视化验证流程。

## 项目简介

本项目将 StarCraft II 微操对抗场景抽象为无人集群协同对抗环境：

- 将地图中的己方可控单位视为无人集群智能体；
- 通过 `sc2_env_adapter.py` 将 PySC2 环境封装为多智能体强化学习接口；
- 通过 `harl_wrapper.py` 对接 HARL 算法，支持 HAPPO、MAPPO 等策略训练；
- 通过 `trainer.py` 管理采样、训练、评估、日志与模型保存；
- 通过 `sim_gui.py` 提供图形化参数配置和运行控制界面。

系统支持命令行运行，也支持 GUI 启动，适合用于毕业设计、算法实验和无人集群协同对抗仿真验证。

## 主要功能

- 多智能体强化学习训练与评估；
- 支持 HAPPO、MAPPO 以及 `simplebattle` 自博弈评估模式；
- 自动检测 StarCraft II 地图中的己方可控单位数量；
- 支持 CPU / CUDA 自动选择；
- 支持模型加载、定期保存和手动保存停止；
- 支持训练与评估参数自定义；
- 支持隐藏 SC2 原生窗口进行后台训练；
- 支持评估时开启可视化画面；
- 支持 Tkinter 图形化控制界面。

## 项目结构

```text
.
├── main.py                       # 命令行入口：训练、评估和参数解析
├── sim_gui.py                    # 图形化仿真控制界面
├── trainer.py                    # 训练管道：采样、更新、评估、保存
├── harl_wrapper.py               # HARL 算法包装器
├── sc2_env_adapter.py            # PySC2 多智能体环境适配器
├── simplebattle_adapter.py       # simplebattle 自博弈评估适配器
├── simplebattle.py               # simplebattle 相关逻辑
├── list_maps.py                  # 本地 SC2 地图扫描工具
├── test_imports.py               # 依赖导入测试
├── test.py                       # 测试脚本
├── HARL/                         # HARL 框架源码
└── checkpoints/                  # 默认模型保存目录
```

## 环境要求

建议环境：

- Windows 10/11；
- Python 3.10 或兼容版本；
- StarCraft II 客户端；
- PySC2；
- PyTorch；
- Gym；
- NumPy；
- PyYAML；
- HARL 本地框架。

> 注意：PySC2 运行需要本机已正确安装 StarCraft II，并准备可用的 `.SC2Map` 地图文件。

## 安装依赖

### 1. 安装 Python 依赖

```bash
pip install pysc2 torch gym numpy pyyaml
```

### 2. 安装 HARL

如果项目根目录下已有 `HARL/` 目录，可执行：

```bash
cd HARL
pip install -e .
cd ..
```

### 3. 检查依赖导入

```bash
python test_imports.py
```

## 地图准备

PySC2 通常会在以下目录搜索地图：

```text
%USERPROFILE%\Documents\StarCraft II\Maps
%PROGRAMFILES%\StarCraft II\Maps
%PROGRAMFILES(X86)%\StarCraft II\Maps
```

可使用以下脚本查看本地可用地图：

```bash
python list_maps.py
```

运行时通过 `--map` 指定地图名称，例如：

```bash
python main.py --map Simple64
```

## 命令行使用

### 基础训练

```bash
python main.py --episodes 1000 --map Simple64
```

### 指定算法训练

```bash
python main.py --algo happo --episodes 1000 --map Simple64
```

可选算法：

- `happo`：异构智能体 PPO，适合异构无人集群；
- `mappo`：多智能体 PPO，适合同构集群；
- `simplebattle`：用于 simplebattle 自博弈评估路径。

### 评估模型

```bash
python main.py --eval --map Simple64 --model-path checkpoints/run_xxx/model_episode_100.pth
```

### 开启评估画面

训练阶段默认隐藏 SC2 窗口；只有在评估模式下才允许显示画面：

```bash
python main.py --eval --visualize --map Simple64
```

### simplebattle 自博弈评估

```bash
python main.py --eval --algo simplebattle --map Simple64 --visualize
```

### 指定设备

```bash
python main.py --device auto
python main.py --device cuda
python main.py --device cpu
```

### 设置保存目录和运行名

```bash
python main.py --episodes 500 --save-dir checkpoints --run-name exp_happo_001
```

模型会保存到：

```text
checkpoints/exp_happo_001/
```

## 图形化界面使用

启动 GUI：

```bash
python sim_gui.py
```

图形界面支持：

- 选择地图和算法；
- 设置训练回合、最大步数、学习率、clip 参数等；
- 设置训练/评估 `step_mul`；
- 选择模型文件或模型目录；
- 选择保存目录；
- 启动、停止、保存并停止仿真；
- 查看训练/评估指标和运行日志。

## 配置文件

也可以使用 YAML 配置文件覆盖默认参数，例如创建 `config.yaml`：

```yaml
algo: happo
num_agents: 5
lr: 0.0005
clip_param: 0.2
ppo_epoch: 10
num_episodes: 1000
max_steps_per_episode: 1000
update_interval: 1000
save_interval: 100
obs_dim: 10
act_dim: 5
hidden_sizes: [128, 128]
use_recurrent_policy: false
share_param: false
```

运行：

```bash
python main.py --config config.yaml --map Simple64
```

命令行参数会覆盖配置文件中的同名或相关字段。

## 常用参数

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--config` | YAML 配置文件路径 | `None` |
| `--map` | SC2 地图名称 | `Simple64` |
| `--episodes` | 训练回合数 | `1000` |
| `--eval` | 启用评估模式 | 关闭 |
| `--eval-episodes` | 评估局数 | `1` |
| `--model-path` | 预训练模型路径 | `None` |
| `--algo` | 算法选择：`happo` / `mappo` / `simplebattle` | 配置决定 |
| `--num-agents` | 期望智能体数量 | 自动检测为准 |
| `--lr` | 学习率 | 配置决定 |
| `--clip-param` | PPO clip 参数 | 配置决定 |
| `--update-interval` | 策略更新间隔 | 配置决定 |
| `--save-interval` | 模型保存间隔 | 配置决定 |
| `--max-steps` | 每回合最大步数 | 配置决定 |
| `--step-mul` | SC2 每次环境步进帧数 | `8` |
| `--visualize` | 评估时显示 SC2 画面 | 关闭 |
| `--save-dir` | 模型保存根目录 | `checkpoints` |
| `--run-name` | 本次运行名称 | 自动时间戳 |
| `--device` | 运行设备：`auto` / `cpu` / `cuda` | `auto` |
| `--torch-threads` | PyTorch CPU 线程数，0 表示不设置 | `0` |
| `--debug-force-spread` | 前 30 步强制分散调试 | 关闭 |

## 训练输出与模型保存

默认情况下，模型保存路径形如：

```text
checkpoints/run_YYYYMMDD_HHMMSS/model_episode_100.pth
```

训练日志会输出：

- 当前设备；
- 环境初始化信息；
- 检测到的可控单位数量；
- 训练奖励；
- 胜率；
- 评估指标；
- 模型保存路径。

## 系统流程

```text
main.py / sim_gui.py
        |
        v
读取配置与命令行参数
        |
        v
初始化 PySC2 多智能体环境
        |
        v
自动检测己方可控单位数量
        |
        v
初始化 HARL 智能体策略
        |
        v
Trainer 执行采样、训练、保存和评估
        |
        v
输出指标并保存模型
```

## 注意事项

- 训练时默认设置 `PYSC2_HIDE_WINDOW=1`，避免频繁弹出 SC2 客户端窗口；
- `--visualize` 仅在 `--eval` 模式下生效；
- `--num-agents` 不能强制改变地图实际单位数量，最终以地图中检测到的可控单位为准；
- 若 CUDA 不可用但指定了 `--device cuda`，程序会回退到 CPU；
- 不同 SC2 地图的单位数量、阵营与动作可用性可能不同，需要根据地图调整观测、动作与奖励设计；
- 当前项目更偏向研究与仿真验证，实际工程部署前建议进一步完善奖励函数、动作映射和场景随机化。

