# 无人机集群协同对抗仿真系统

基于PySC2和HARL的多智能体强化学习框架，用于无人机集群协同对抗仿真。

## 概述

本项目实现了基于StarCraft II的无人机集群仿真系统，使用HARL（Heterogeneous-Agent Reinforcement Learning）框架训练多智能体策略。

## 架构

- **算法层**: HARL算法包装器，支持HAPPO、MAPPO等算法
- **环境适配器**: PySC2环境适配器，将SC2环境转换为多智能体格式
- **训练管道**: PPO风格的训练循环
- **主脚本**: 训练和评估入口

## 文件结构

```
.
├── harl_wrapper.py      # HARL算法包装器
├── sc2_env_adapter.py   # PySC2环境适配器
├── trainer.py           # 训练管道
├── main.py              # 主脚本
└── HARL/                # HARL框架
```

## 安装依赖

1. 安装PySC2:
   ```bash
   pip install pysc2
   ```

2. 安装HARL:
   ```bash
   cd HARL
   pip install -e .
   ```

3. 安装其他依赖:
   ```bash
   pip install torch gym numpy
   ```

## 使用方法

### 训练

```bash
python main.py --episodes 1000 --map Simple64
```

### 评估

```bash
python main.py --eval --map Simple64
```

### 自定义配置

创建config.yaml文件：

```yaml
algo: happo
num_agents: 5
lr: 0.0005
clip_param: 0.2
num_episodes: 1000
max_steps_per_episode: 1000
```

然后运行：

```bash
python main.py --config config.yaml
```

## 算法选择

- **HAPPO**: 异构智能体PPO，适用于异构无人机集群
- **MAPPO**: 多智能体PPO，适用于同构集群

## 环境配置

环境假设StarCraft II地图中有多个单位作为智能体。观测包括：
- 单位位置、健康、护盾
- 相对距离到其他单位
- 敌方单位信息

动作空间映射到SC2动作，如移动、攻击等。

## 扩展

- 支持可变集群大小
- 易于切换算法
- 兼容SMAC-like微操作场景

## 注意事项

- 需要StarCraft II游戏安装
- 当前实现为简化版本，实际部署时需完善奖励函数和动作映射
- 异构智能体支持通过不同的观测/动作空间实现</content>
<parameter name="filePath">G:\code\2026\无人集群协同对抗仿真\README.md
