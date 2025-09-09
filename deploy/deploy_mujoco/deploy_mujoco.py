import time

import mujoco.viewer
import mujoco
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import yaml
#from legged_gym.envs.base.legged_robot import LeggedRobot  

def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


if __name__ == "__main__":
    # get config file name from command line
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    args = parser.parse_args()
    config_file = args.config_file
    with open(f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]


        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)

        default_angles = np.array(config["default_angles"], dtype=np.float32)

        ang_vel_scale = config["ang_vel_scale"]
        lin_vel_scale = config["lin_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
        height_measurements = config.get("height_measurements", 1.0)
        measured_points_x = config.get("measured_points_x", [])
        measured_points_y = config.get("measured_points_y", [])
        clip_observations = config.get("clip_observations", 100.0)
        clip_actions = config.get("clip_actions", 100.0)
        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        
        cmd = np.array(config["cmd_init"], dtype=np.float32)

    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    counter = 0

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt
    print("Actuator force range:", m.actuator_ctrlrange)
    # load policy
    policy = torch.jit.load(policy_path)
    with mujoco.viewer.launch_passive(m, d) as viewer:
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
            joint_start = 7
            joint_end = joint_start + len(target_dof_pos)
            joint_vel_start = 6
            joint_vel_end = joint_vel_start + len(target_dof_pos)
            # tau = pd_control(
            #     target_dof_pos,
            #     d.qpos[joint_start:joint_end],
            #     kps,
            #     np.zeros_like(kds),
            #     d.qvel[joint_vel_start:joint_vel_end],
            #     kds
            # )
            # d.ctrl[:] = tau

            fixed_torque = np.array([1000.0] * len(d.ctrl), dtype=np.float32)  # 为每个关节设置固定力矩
            d.ctrl[:] = fixed_torque
            # print("Applied fixed torques:", fixed_torque)
            # print("Applied torques:", tau)
            # print("Actual torques:", d.actuator_force)
            #print("Target positions:", target_dof_pos)
            mujoco.mj_step(m, d)
            #print("Current joint positions (qpos):", d.qpos[joint_start:joint_end])
            counter += 1
            # print("Current joint positions (qpos):", d.qpos[joint_start:joint_end])
            # print("Current joint velocities (qvel):", d.qvel[joint_vel_start:joint_vel_end])
            if counter % control_decimation == 0:
                # Apply control signal here.

                qj = d.qpos[joint_start:joint_end]       
                dqj = d.qvel[joint_vel_start:joint_vel_end] 
                quat = d.qpos[3:7] 
                omega = d.qvel[3:6] 
                qj = (qj - default_angles) * dof_pos_scale
                dqj = dqj * dof_vel_scale
                gravity_orientation = get_gravity_orientation(quat)
                omega = omega * ang_vel_scale
                lin_vel = d.qvel[:3]  # 提取底座线速度
                lin_vel = lin_vel * lin_vel_scale  
                period = 0.8
                count = counter * simulation_dt
                phase = count % period / period
                sin_phase = np.sin(2 * np.pi * phase)
                cos_phase = np.cos(2 * np.pi * phase)
                # 地形高度观测（平地时全0，数量和训练一致）
                num_height = len(measured_points_x) * len(measured_points_y)
                heights = np.zeros(num_height, dtype=np.float32) * height_measurements
                obs[:3] = lin_vel
                obs[3:6] = omega  # 底座角速度
                obs[6:9] = gravity_orientation
                obs[9:12] = cmd * cmd_scale
                obs[12 : 12 + num_actions] = qj
                obs[12 + num_actions : 12 + 2 * num_actions] = dqj
                obs[12 + 2 * num_actions : 12 + 3 * num_actions] = action  # 上一步裁剪后的action
                obs[12 + 3 * num_actions : 12 + 3 * num_actions  + num_height] = heights
                # print("lin_vel:", lin_vel)
                # print("omega:", omega)
                # print("gravity_orientation:", gravity_orientation)
                # print("cmd * cmd_scale:", cmd * cmd_scale)
                # print("qj (scaled):", qj)
                # print("dqj (scaled):", dqj)
                # print("action (previous step):", action)
                """ 
                def compute_observations(self):
                    self.obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                                self.base_ang_vel  * self.obs_scales.ang_vel,
                                                self.projected_gravity,
                                                self.commands[:, :3] * self.commands_scale,
                                                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                                self.dof_vel * self.obs_scales.dof_vel,
                                                self.actions
                                                ),dim=-1)
                    # add perceptive inputs if not blind

                    if self.cfg.terrain.measure_heights:

                        heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
                        
                        self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)

                    # add noise if needed
                    if self.add_noise:

                        self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

                            obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                """
                add_noise = config.get("add_noise", False)
                if add_noise:
                    noise_scales = config.get("noise_scales", {})
                    noise_scale_vec = np.zeros_like(obs)
                    noise_level = config.get("noise_level", 1.0)
                    noise_scale_vec += noise_level * 0.01  # 你可以细分每一段
                    obs += (2 * np.random.rand(*obs.shape) - 1) * noise_scale_vec

                obs = np.clip(obs, -clip_observations, clip_observations)
                #print("obs:", obs)
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                #print("obs_tensor:", obs_tensor)
                action = policy(obs_tensor).detach().numpy().squeeze()

                action = np.clip(action, -clip_actions, clip_actions)

                #print("policy action:", action)
                #print("default_angles:", default_angles) 
                target_dof_pos = action * action_scale + default_angles  # 策略输出的是增量 并放缩+初始角度
                #print("target_dof_pos:", target_dof_pos)
            viewer.sync()

            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
