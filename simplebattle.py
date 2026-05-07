"""
简单的星际争霸2 AI 对抗程序
两个电脑玩家进行对战，AI 会执行一些基础操作
"""

import random
import sys
from absl import app
from pysc2 import maps
from pysc2.env import sc2_env
from pysc2.agents import base_agent
from pysc2.lib import actions, features, units

# 常用动作 ID 的别名（方便记忆）
_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_CAMERA_MOVE = actions.FUNCTIONS.move_camera.id
_HARVEST_GATHER = actions.FUNCTIONS.Harvest_Gather_screen.id
#_SMART = actions.FUNCTIONS.smart.id


class SimpleBattleAgent(base_agent.BaseAgent):
    """
    一个简单的对战 AI
    功能：
    1. 自动采集资源
    2. 训练基础单位
    3. 攻击敌人
    4. 随机添加一些"战术"动作
    """

    def __init__(self):
        super(SimpleBattleAgent, self).__init__()
        self.iteration = 0
        self.last_attack_frame = 0
        self.attack_cooldown = 50  # 攻击冷却帧数

    def step(self, obs):
        """每帧调用的主函数"""
        self.iteration += 1

        # 获取当前可用动作
        available_actions = obs.observation["available_actions"]

        # 获取玩家信息
        player_info = obs.observation["player"]
        minerals = player_info[0]  # 当前矿物数
        food_used = player_info[2]  # 已用人口
        food_cap = player_info[3]   # 人口上限

        # 获取单位信息
        units_info = self._get_units_info(obs)

        # === 主要决策逻辑 ===

        # 策略1：如果人口小于上限，尝试训练单位
        if food_used < food_cap and minerals >= 50:
            action = self._train_units(obs, available_actions)
            if action:
                return action

        # 策略2：如果有空闲农民，采集矿物
        if units_info["idle_workers"] > 0:
            action = self._harvest_minerals(obs, available_actions)
            if action:
                return action

        # 策略3：如果有战斗单位，攻击敌人
        if units_info["combat_units"] > 0 and self.iteration - self.last_attack_frame > self.attack_cooldown:
            action = self._attack_enemy(obs, available_actions)
            if action:
                self.last_attack_frame = self.iteration
                return action

        # 策略4：如果没有单位，选中所有单位
        if _SELECT_ARMY in available_actions:
            return actions.FunctionCall(_SELECT_ARMY, [[0]])

        # 策略5：随机移动镜头（增加战术随机性）
        if random.random() < 0.02 and _CAMERA_MOVE in available_actions:
            return self._random_camera_move(obs)

        # 默认：什么都不做
        return actions.FunctionCall(_NO_OP, [])

    def _get_units_info(self, obs):
        """解析单位信息"""
        units_info = {
            "workers": 0,           # 农民数量
            "combat_units": 0,      # 战斗单位数量
            "idle_workers": 0,      # 空闲农民
            "enemy_units": 0,       # 敌方单位数量
            "enemy_positions": []   # 敌方位置
        }

        # 解析单位数据
        if "units" in obs.observation:
            for unit in obs.observation["units"]:
                if len(unit) >= 5:
                    unit_type = unit[0]  # 单位类型

                    # 识别单位类型
                    # 农民类型（虫族工蜂、人族SCV、神族探机）
                    if unit_type in [units.Zerg.Drone, units.Terran.SCV, units.Protoss.Probe]:
                        units_info["workers"] += 1
                        if unit[4]:  # is_idle
                            units_info["idle_workers"] += 1

                    # 战斗单位（简化判断：不是农民、不是建筑、不是王虫）
                    elif unit_type not in [units.Zerg.Drone, units.Terran.SCV, units.Protoss.Probe,
                                           units.Zerg.Overlord, units.Zerg.Hatchery,
                                           units.Terran.CommandCenter, units.Protoss.Nexus,
                                           units.Zerg.Larva, units.Zerg.Egg]:
                        units_info["combat_units"] += 1

        # 获取敌人位置（从玩家相对关系中判断）
        try:
            if "feature_screen" in obs.observation:
                player_relative = obs.observation["feature_screen"][features.PLAYER_RELATIVE]
                # 敌人标记为 4
                enemy_positions = (player_relative == 4).nonzero()
                if len(enemy_positions[0]) > 0:
                    units_info["enemy_units"] = len(enemy_positions[0])
                    # 记录第一个敌人的位置
                    units_info["enemy_positions"] = [int(enemy_positions[1][0]), int(enemy_positions[0][0])]
        except:
            pass

        return units_info

    def _harvest_minerals(self, obs, available_actions):
        """让空闲农民采集矿物"""
        if _HARVEST_GATHER not in available_actions:
            return None

        try:
            if "feature_screen" in obs.observation:
                unit_type_screen = obs.observation["feature_screen"][features.UNIT_TYPE]

                # 寻找矿物晶石
                mineral_types = [
                    units.Neutral.MineralField,
                    units.Neutral.MineralField750,
                ]
                for mineral_type in mineral_types:
                    positions = (unit_type_screen == mineral_type).nonzero()
                    if len(positions[0]) > 0:
                        target_x = int(positions[1][0])
                        target_y = int(positions[0][0])
                        return actions.FunctionCall(_HARVEST_GATHER, [[0], [target_x, target_y]])
        except Exception:
            pass

        return None

    def _train_units(self, obs, available_actions):
        """训练基础单位"""
        # 训练单位的功能ID（简化，实际需要根据种族判断）
        train_actions = [
            actions.FUNCTIONS.Train_Marine_quick.id,      # 人族机枪兵
            actions.FUNCTIONS.Train_Zergling_quick.id,    # 虫族小狗
            actions.FUNCTIONS.Train_Zealot_quick.id       # 神族狂战士
        ]

        for action_id in train_actions:
            if action_id in available_actions:
                return actions.FunctionCall(action_id, [[0]])

        return None

    def _attack_enemy(self, obs, available_actions):
        """攻击敌人"""
        if _ATTACK_SCREEN not in available_actions:
            return None

        try:
            # 获取敌人位置
            if "feature_screen" in obs.observation:
                player_relative = obs.observation["feature_screen"][features.PLAYER_RELATIVE]
                # 寻找敌人（标记为 4）
                enemy_positions = (player_relative == 4).nonzero()

                if len(enemy_positions[0]) > 0:
                    # 随机选择一个敌人攻击
                    target_idx = random.randint(0, len(enemy_positions[0]) - 1)
                    target_x = int(enemy_positions[1][target_idx])
                    target_y = int(enemy_positions[0][target_idx])
                    return actions.FunctionCall(_ATTACK_SCREEN, [[0], [target_x, target_y]])
        except:
            pass

        # 如果没有找到敌人，移动镜头寻找
        return self._random_camera_move(obs)

    def _random_camera_move(self, obs):
        """随机移动镜头"""
        if _CAMERA_MOVE not in obs.observation["available_actions"]:
            return actions.FunctionCall(_NO_OP, [])

        # 随机移动镜头到地图某个位置
        map_size = 84  # 标准地图尺寸
        target_x = random.randint(0, map_size)
        target_y = random.randint(0, map_size)

        return actions.FunctionCall(_CAMERA_MOVE, [[target_x, target_y]])


class RandomAgent(base_agent.BaseAgent):
    """
    纯随机动作的 AI（用于对比）
    """
    def step(self, obs):
        available_actions = obs.observation["available_actions"]
        if len(available_actions) == 0:
            return actions.FunctionCall(_NO_OP, [])

        # 随机选择一个可用动作
        action_id = random.choice(available_actions)
        action = actions.FUNCTIONS[action_id]

        # 为动作生成正确格式的参数
        args = []
        for arg in action.args:
            if arg.name == "screen":
                args.append([random.randint(0, 83), random.randint(0, 83)])
            elif arg.name == "minimap":
                args.append([random.randint(0, 63), random.randint(0, 63)])
            elif arg.name == "screen2":
                args.append([random.randint(0, 83), random.randint(0, 83)])
            elif arg.name == "queued":
                args.append([0])
            elif arg.name == "select_point_act":
                args.append([0])
            elif arg.name == "select_add":
                args.append([0])
            elif arg.name == "select_unit_act":
                args.append([0])
            elif arg.name == "select_unit_id":
                args.append([0])
            elif arg.name == "select_worker":
                args.append([0])
            elif arg.name == "build_queue_id":
                args.append([0])
            elif arg.name == "unload_id":
                args.append([0])
            else:
                args.append([0])

        return actions.FunctionCall(action_id, args)


def main(unused_argv):
    """主函数：运行对战"""
    print("启动星际争霸2 AI 对战...")
    print("AI 1: SimpleBattleAgent (智能AI)")
    print("AI 2: RandomAgent (随机AI)")
    print("按 Ctrl+C 停止游戏\n")

    try:
        # 创建游戏环境
        with sc2_env.SC2Env(
            map_name="Simple64",                    # 使用 Simple64 地图
            players=[
                sc2_env.Agent(sc2_env.Race.random),  # 第一个AI，随机种族
                sc2_env.Agent(sc2_env.Race.random),  # 第二个AI，随机种族
            ],
            agent_interface_format=sc2_env.AgentInterfaceFormat(
                feature_dimensions=sc2_env.Dimensions(
                    screen=84,      # 屏幕特征分辨率
                    minimap=64      # 小地图特征分辨率
                )
            ),
            step_mul=32,              # 每8个游戏帧调用一次 step
            game_steps_per_episode=0,  # 无限运行
            visualize=True           # 显示游戏画面
        ) as env:

            # 创建两个 AI 实例
            agents = [SimpleBattleAgent(), RandomAgent()]

            # 重置环境
            timesteps = env.reset()
            episode = 1
            step_count = 0

            # 主循环
            while True:
                step_count += 1

                # 获取每个 AI 的动作
                actions_list = []
                for i, agent in enumerate(agents):
                    # 如果游戏结束，跳过
                    if timesteps[i].last():
                        actions_list.append(actions.FunctionCall(_NO_OP, []))
                    else:
                        action = agent.step(timesteps[i])
                        actions_list.append(action)

                # 执行动作
                timesteps = env.step(actions_list)

                # 显示进度（每500步显示一次）
                if step_count % 500 == 0:
                    print(f"Episode {episode}, Step {step_count}")

                # 检查是否游戏结束
                if timesteps[0].last() or timesteps[1].last():
                    print(f"\n=== Episode {episode} 结束 ===")

                    # 显示结果
                    for i, t in enumerate(timesteps):
                        if t.last():
                            score = t.observation.get('score', [0])[0] if t.observation else 0
                            print(f"AI {i+1} 的最终分数: {score}")

                    # 重置游戏开始新的一局
                    timesteps = env.reset()
                    episode += 1
                    step_count = 0

                    print("\n游戏继续中... 按 Ctrl+C 停止\n")

    except KeyboardInterrupt:
        print("\n游戏已停止")
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 使用 absl.app 来运行，自动处理 flags 初始化
    app.run(main)