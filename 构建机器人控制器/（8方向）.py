import pygame  # 导入pygame库，用于游戏开发
import time  # 导入time库，用于时间控制
import heapq  # 导入heapq库，用于优先队列操作
import numpy as np  # 导入numpy库，用于数组和矩阵操作
import random  # 导入random库，用于生成随机数
from matplotlib import rcParams  # 导入matplotlib的rcParams，用于设置图形参数

# 设置matplotlib的字体为微软雅黑，确保中文显示正常
rcParams['font.family'] = 'Microsoft YaHei'
rcParams['axes.unicode_minus'] = False  # 允许显示负号

# 定义颜色
WHITE = (255, 255, 255)  # 白色
BLACK = (0, 0, 0)  # 黑色
GREEN = (0, 255, 0)  # 目标符号的颜色
RED = (255, 0, 0)  # 已打击目标的颜色
BLUE = (0, 0, 255)  # 机器人颜色


# 地图类
class MapGame:
    def __init__(self, game_map):
        self.map = game_map  # 地图数据
        self.rows = len(game_map)  # 地图的行数
        self.cols = len(game_map[0]) if self.rows > 0 else 0  # 地图的列数
        self.screen = None  # Pygame屏幕对象
        self.cell_size = 20  # 每个格子的大小
        self.robot_pos = (0, 0)  # 机器人的起始位置
        self.finished = False  # 是否完成所有目标的标志
        self.steps = -0.5  # 记录步数

        # Q学习参数
        self.q_table = np.zeros((self.rows, self.cols, 8))  # 初始化Q表，8个方向的动作（上、下、左、右及对角线）
        self.learning_rate = 0.1  # 学习率
        self.discount_factor = 0.9  # 折扣因子
        self.epsilon = 1.0  # 探索率
        self.epsilon_decay = 0.995  # 探索率衰减
        self.epsilon_min = 0.01  # 最小探索率

    def is_valid_move(self, x, y):
        # 检查移动是否有效（在地图范围内）
        return (0 <= x < self.rows and
                0 <= y < self.cols)  # 允许经过其他位置

    def heuristic(self, a, b):
        # 曼哈顿距离计算，用于A*算法
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def move_robot(self, start, end):
        # 移动机器人到新位置并绘制
        steps = 10  # 分成10步移动
        for step in range(steps):
            # 计算插值位置
            x = start[0] + (end[0] - start[0]) * (step / steps)
            y = start[1] + (end[1] - start[1]) * (step / steps)
            self.robot_pos = (int(round(x)), int(round(y)))  # 更新机器人的位置
            self.draw_robot(self.robot_pos[0], self.robot_pos[1])  # 仅绘制机器人的最新位置
            pygame.display.flip()  # 更新屏幕
            time.sleep(0.001)  # 短暂停，控制速度

    def get_neighbors(self, pos):
        # 获取当前点的邻居（八个方向）
        x, y = pos
        neighbors = [(x + dx, y + dy) for dx, dy in
                     [(-1, 0), (1, 0), (0, -1), (0, 1),
                      (-1, -1), (-1, 1), (1, -1), (1, 1)]]  # 八个方向的邻居
        return [n for n in neighbors if self.is_valid_move(*n)]  # 仅返回有效的邻居

    def a_star_search(self, start, goal):
        # A*算法实现
        open_list = []  # 开放列表
        heapq.heappush(open_list, (0, start))  # 添加起点到开放列表
        came_from = {}  # 路径追溯字典
        g_score = {start: 0}  # 起点到当前点的实际成本
        f_score = {start: self.heuristic(start, goal)}  # 启发式成本

        while open_list:
            current = heapq.heappop(open_list)[1]  # 获取当前点

            # 如果到达目标
            if current == goal:
                return self.reconstruct_path(came_from, current)  # 重新构建路径

            for neighbor in self.get_neighbors(current):  # 获取邻居
                tentative_g_score = g_score[current] + 1  # 计算临时g值

                # 如果新路径更好，更新路径信息
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current  # 记录路径
                    g_score[neighbor] = tentative_g_score  # 更新g值
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)  # 更新f值

                    if neighbor not in [i[1] for i in open_list]:  # 如果邻居不在开放列表中
                        heapq.heappush(open_list, (f_score[neighbor], neighbor))  # 添加到开放列表

        return []  # 如果没有路径，返回空列表

    def reconstruct_path(self, came_from, current):
        # 重新构建从起点到目标的路径
        total_path = [current]  # 初始化路径
        while current in came_from:
            current = came_from[current]  # 沿路径回溯
            total_path.append(current)  # 添加到路径
        total_path.reverse()  # 反转路径
        return total_path

    def get_action(self):
        # ε-贪婪策略选择动作
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(range(8))  # 随机选择动作
        else:
            # 选择Q最高的动作
            return np.argmax(self.q_table[self.robot[0], self.robot_pos[1]])

    def update_q_value(self, action, reward, next_state):
        # 更新Q值
        current_q = self.q_table[self.robot_pos[0], self.robot_pos[1], action]  # 当前Q值
        max_future_q = np.max(self.q_table[next_state[0], next_state[1]])  # 下一状态的最大Q值
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_future_q - current_q)  # 更新公式
        self.q_table[self.robot_pos[0], self.robot_pos[1], action] = new_q  # 保存新的Q值

    def find_nearest_target(self):
        # 找到离机器人最近的目标
        nearest_target = None  # 初始化最近目标
        min_distance = float('inf')  # 初始化最小距离为无穷大

        for i in range(self.rows):
            for j in range(self.cols):
                if self.map[i][j] == '#':  # 目标符号
                    distance = self.heuristic(self.robot_pos, (i, j))  # 计算距离
                    if distance < min_distance:  # 更新最近目标
                        min_distance = distance
                        nearest_target = (i, j)

        return nearest_target  # 返回最近目标的位置

    def traverse(self):
        # 遍历地图并打击目标
        while True:
            next_target = self.find_nearest_target()  # 找到下一个目标
            if next_target is None:
                break  # 没有更多目标，退出循环

            path = self.a_star_search(self.robot_pos, next_target)  # 使用A*寻找路径
            if not path:
                print("没有找到路径！")  # 如果没有路径，打印信息
                break

            for step in path:
                self.move_robot(self.robot_pos, step)  # 移动机器人到目标位置
                self.steps += 0.5  # 增加步数计数

                # Q学习的过程
                action = self.get_action()  # 获取动作
                next_pos = step  # 下一步的位置

                # 检查移动是否有效
                if self.is_valid_move(*next_pos):
                    reward = -1  # 每次移动的惩罚
                    if next_pos == next_target:  # 到达目标
                        reward = 10  # 找到目标的奖励
                        self.map[next_target[0]][next_target[1]] = 'R'  # 将目标变为红色
                    self.update_q_value(action, reward, next_pos)  # 更新Q值
                    self.robot_pos = next_pos  # 更新机器人位置

                self.draw_map()  # 重新绘制地图
                self.draw_steps()  # 绘制当前步数
                self.draw_robot(self.robot_pos[0], self.robot_pos[1])  # 确保绘制机器人的最新位置
                pygame.display.flip()  # 更新屏幕

        # 所有目标打击完成后结束程序
        self.finished = True  # 设置完成标志
        print("所有目标已打击完成！")  # 输出完成信息
        print(f"总步数: {self.steps}")  # 打印总步数
        time.sleep(2)  # 等待2秒以便观察消息

    def draw_steps(self):
        # 在屏幕上绘制当前步数
        font = pygame.font.SysFont(None, 24)  # 设置字体和大小
        text = font.render(f"step: {self.steps}", True, BLACK)  # 创建文本
        self.screen.blit(text, (10, 10))  # 将文本绘制到屏幕上

    def run(self):
        # 初始化Pygame并运行游戏
        pygame.init()  # 初始化Pygame
        self.screen = pygame.display.set_mode((self.cols * self.cell_size, self.rows * self.cell_size))  # 创建游戏窗口
        pygame.display.set_caption("自动寻找目标")  # 窗口标题

        # 绘制初始地图
        self.draw_map()  # 绘制地图
        self.draw_robot(self.robot_pos[0], self.robot_pos[1])  # 绘制初始位置的机器人
        pygame.display.flip()  # 更新显示

        # 设置初始机器人位置
        self.traverse()  # 开始遍历

        # 等待用户关闭窗口
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:  # 处理退出事件
                    running = False
            # 如果所有目标已打击完成，退出循环
            if self.finished:
                running = False

        pygame.quit()  # 退出Pygame

    def draw_robot(self, x, y):
        # 绘制机器人
        pygame.draw.rect(self.screen, BLUE, (y * self.cell_size, x * self.cell_size, self.cell_size, self.cell_size))  # 绘制机器人矩形

    def draw_map(self):
        # 绘制地图
        self.screen.fill(WHITE)  # 填充背景为白色
        for i in range(self.rows):
            for j in range(self.cols):
                # 根据地图内容选择颜色
                color = GREEN if self.map[i][j] == '#' else RED if self.map[i][j] == 'R' else BLACK  # 根据状态选择颜色
                pygame.draw.rect(self.screen, color,  # 绘制地图方格
                                 (j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size))
                pygame.draw.rect(self.screen, WHITE, (  # 绘制方格的白边
                    j * self.cell_size + 1, i * self.cell_size + 1, self.cell_size - 2, self.cell_size - 2))


# 从文件读取地图
def load_map_from_file(filename):
    with open(filename, 'r') as file:  # 打开文件
        return [list(line.strip()) for line in file.readlines()]  # 将每一行转换为列表并返回


# 创建地图
game_map = load_map_from_file('target_map.txt')  # 从文件加载地图

# 创建并运行游戏
game = MapGame(game_map)  # 实例化游戏对象
game.run()  # 运行游戏
