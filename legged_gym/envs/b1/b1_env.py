# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 ETH Zurich, Nikita Rudin
# Adapted for Unitree B1 Robot

from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
import torch
from torch import Tensor
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi


class B1Robot(LeggedRobot):
    """宇树B1四足机器人环境类，继承自LeggedRobot基础类，适配B1硬件特性与自定义任务需求"""
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        # 1. 调用父类构造函数，初始化基础仿真环境
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.initial_base_pos = torch.zeros(self.num_envs, 3, device=self.device)  # 初始位置
        self.prev_base_pos = torch.zeros(self.num_envs, 3, device=self.device)    # 上一步位置
        self.total_distance = torch.zeros(self.num_envs, device=self.device)      # 累计移动距离
        # 2. B1特有初始化：目标点配置（从配置文件读取，默认x=5m、y=0m、z=0m）
        self.target_pos = torch.tensor(
            cfg.env.target_pos,
            device=self.device,
            dtype=torch.float32
        )
        
        # 3. B1关节映射验证（确保与URDF关节名匹配，12自由度：4条腿×3关节/腿）
        self._verify_b1_joints()

    def _verify_b1_joints(self):
        """验证B1关节数量与命名是否正确（防止URDF加载错误）"""
        expected_joint_count = 12  # B1为12自由度机器人（hip_yaw, hip_roll, hip_pitch, thigh, calf×4腿）
        print("elf.num_dof:",self.num_dof)
        if self.num_dof != expected_joint_count:
            raise ValueError(f"B1机器人需12个自由度，当前加载{self.num_dof}个关节，请检查URDF文件！")
        
        # 验证关键关节名（匹配宇树B1 URDF关节命名规范）
        required_joint_keywords = ["hip", "thigh", "calf"]
        for joint_name in self.dof_names:
            if not any(kw in joint_name.lower() for kw in required_joint_keywords):
                raise ValueError(f"关节名{joint_name}不匹配B1规范，需包含'hip'/'thigh'/'calf'")

    # def _get_noise_scale_vec(self, cfg):
    #     # 1. 获取父类生成的237维噪声向量（前235维正确，最后2维需替换）
    #     noise_vec = super()._get_noise_scale_vec(cfg)

    #     print("父类_get_noise_scale_vec已执行")  # 用于验证是否被调用
    #     # 验证维度是否为237（确保配置生效）
    #     assert noise_vec.shape[-1] == 237, f"噪声向量应为237维，实际为{noise_vec.shape[-1]}"
        
    #     # 2. 从配置读取新增维度的噪声参数
    #     target_noise_scale = cfg.noise.noise_scales.target_dist_noise_scale  # 你的自定义噪声缩放
    #     global_noise_level = cfg.noise.noise_level  # 全局噪声强度
        
    #     # 3. 计算新增2维（x差、y差）的噪声值
    #     new_noise = torch.tensor(
    #         [target_noise_scale * global_noise_level,  # 第236维：x差的噪声
    #         target_noise_scale * global_noise_level], # 第237维：y差的噪声
    #         device=self.device,
    #         dtype=torch.float32
    #     )
        
    #     # 4. 替换噪声向量的最后2维（关键操作！不拼接，只替换）
    #     noise_vec[-2:] = new_noise  # [-2:] 表示取最后2个元素并赋值
        
    #     print("子类_get_noise_scale_vec已执行")  # 用于验证是否被调用
    #     return noise_vec
    
    # def compute_observations(self):
    #     # 1. 生成父类基础观测（固定235维）
    #     self.obs_buf = torch.cat((
    #         self.base_lin_vel * self.obs_scales.lin_vel,
    #         self.base_ang_vel * self.obs_scales.ang_vel,
    #         self.projected_gravity,
    #         self.commands[:, :3] * self.commands_scale,
    #         (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
    #         self.dof_vel * self.obs_scales.dof_vel,
    #         self.actions
    #     ), dim=-1)
        
    #     # 2. 处理地形高度观测（可选，若开启则+1维，但需确保最终维度仍为237）
    #     # 注意：若开启地形高度，需调整基础观测维度，否则会超237（此处默认关闭，或后续按需调整）
    #     if self.cfg.terrain.measure_heights:
    #         heights = torch.clip(
    #             self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights,
    #             -1, 1.
    #         ) * self.obs_scales.height_measurements
    #         self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
    #         # 若开启地形高度，需从基础观测中移除1个维度（如某个冗余的命令维度），确保此时仍为235维
    #         # （简化方案：暂不开启地形高度，或调整配置 num_observations=238，此处按默认关闭处理）
        
    #     # 3. 关键：移出条件块！确保pos_diff必定义，且拼接2维x/y差
    #     current_base_xy = self.root_states[:, :2]
    #     target_xy = self.target_pos[:2].unsqueeze(0).repeat(current_base_xy.shape[0], 1)
    #     pos_diff = current_base_xy - target_xy  # 必定义，2维
        
    #     # 4. 拼接后维度：235（基础）+2（x/y差）=237维（与配置匹配）
    #     self.obs_buf = torch.cat([self.obs_buf, pos_diff], dim=-1)
        
    #     # 5. 添加噪声（此时obs_buf已为237维，与noise_scale_vec维度一致）
    #     if self.add_noise:
    #         self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
        
    #     # 验证维度（可选，避免隐藏错误）
    #     assert self.obs_buf.shape[-1] == self.cfg.env.num_observations, \
    #         f"观测维度 {self.obs_buf.shape[-1]} 与配置 {self.cfg.env.num_observations} 不匹配"

    def _reward_arrive_target_point(self):
        current_base_xy = self.root_states[:, :2]
        target_xy = self.target_pos[:2].unsqueeze(0).repeat(current_base_xy.shape[0], 1)
        dist_to_target = torch.norm(current_base_xy - target_xy, dim=1)
        
        max_reward = 10.0
        min_dist = 0.2
        max_dist = 15.0
        
        return torch.where(
            dist_to_target <= min_dist,
            torch.full_like(dist_to_target, max_reward),
            torch.clamp(max_reward * (1 - dist_to_target / max_dist), min=0.0)
        )

    def _reward_total_distance(self):
        """
        仅返回累计移动距离（纯数值），框架会自动乘以配置的 total_distance 权重
        无需手动乘权重，避免重复计算
        """
        # 1. 获取当前/上一步底座x/y坐标（忽略z轴）
        current_base_xy = self.root_states[:, :2].clone()
        prev_base_xy = self.prev_base_pos[:, :2]
        
        # 2. 计算单步距离（欧氏距离，非负）
        step_distance = torch.norm(current_base_xy - prev_base_xy, dim=1)
        
        # 3. 累加总距离（过滤微小抖动）
        self.total_distance += torch.clamp(step_distance, min=0.0)
        
        # 4. 更新上一步位置
        self.prev_base_pos[:, :2] = current_base_xy
        
        # 关键：仅返回累计距离（纯数值），权重由框架自动乘
        return self.total_distance


    def check_termination(self):
        """扩展终止条件：在基础条件中添加「到达目标点终止」，明确任务完成标志"""
        # 1. 调用父类方法，检查基础终止条件（跌倒、超时等）
        super().check_termination()
        
        # 2. 新增：到达目标点终止（距离<0.2m时，终止当前episode）
        current_base_xy = self.root_states[:, :2]
        target_xy = self.target_pos[:2].unsqueeze(0).repeat(current_base_xy.shape[0], 1)
        dist_to_target = torch.norm(current_base_xy - target_xy, dim=1)
        
        # 将「到达目标」加入终止缓冲区（True表示需要重置环境）
        self.reset_buf |= (dist_to_target < 0.2)

    def _compute_torques(self, actions):
        """适配B1关节特性的扭矩计算：继承PD位置控制，确保关节力符合B1硬件限制"""
        # 1. 调用父类PD控制逻辑（B1默认使用位置控制，与基础类一致）
        torques = super()._compute_torques(actions)
        
        # 2. B1特有：限制髋关节扭矩（避免过大扭矩损坏关节，根据B1硬件参数调整）
        hip_joint_indices = [i for i, name in enumerate(self.dof_names) if "hip" in name.lower()]
        if hip_joint_indices:
            hip_torque_limit = torch.tensor(250.0, device=self.device)  # B1髋关节最大扭矩约250N·m
            torques[:, hip_joint_indices] = torch.clip(
                torques[:, hip_joint_indices],
                -hip_torque_limit,
                hip_torque_limit
            )
        
        return torques

    # def reset_idx(self, env_ids):
    #     """继承重置逻辑，确保B1重置时目标点观测缓冲区同步清零"""
    #     super().reset_idx(env_ids)
        
    #     # 额外：重置时确保目标点相对距离的历史影响清除（若后续添加历史观测需扩展）
    #     if hasattr(self, "pos_diff_buf"):
    #         self.pos_diff_buf[env_ids] = 0.0

    def reset(self):
        """重置环境时，初始化位置和累计距离"""
        obs, privileged_obs = super().reset()
        # 记录初始位置和上一步位置
        self.initial_base_pos = self.root_states[:, :3].clone()  # 保存初始x/y/z坐标
        self.prev_base_pos = self.root_states[:, :3].clone()      # 上一步位置初始化为初始位置
        self.total_distance.zero_()  # 重置累计距离
        return obs, privileged_obs