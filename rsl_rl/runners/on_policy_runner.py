from __future__ import annotations

import os
import statistics
import time
from collections import deque

import torch

import rsl_rl
from rsl_rl.algorithms import PPO, Distillation
from rsl_rl.env import VecEnv
from rsl_rl.modules import (
    ActorCritic,
    ActorCriticRecurrent,
    EmpiricalNormalization,
    StudentTeacher,
    StudentTeacherRecurrent,
    # TransformerActorCritic,
)
from rsl_rl.utils import store_code_state


def _migrate_train_cfg(train_cfg: dict) -> None:
    """将 mjlab v5 的 actor/critic 配置格式转换为旧版 policy 配置格式。"""
    if "policy" not in train_cfg and "actor" in train_cfg:
        actor_cfg = train_cfg.pop("actor")
        critic_cfg = train_cfg.pop("critic", {})
        # 移除 runner 层级的 class_name（例如 "OnPolicyRunner"）；此处不会使用。
        train_cfg.pop("class_name", None)
        policy_cfg: dict = {
            "class_name": "ActorCritic",
            "actor_hidden_dims": list(actor_cfg.get("hidden_dims", [256, 256, 256])),
            "critic_hidden_dims": list(critic_cfg.get("hidden_dims", [256, 256, 256])),
            "activation": actor_cfg.get("activation", "elu"),
        }
        dist_cfg = actor_cfg.get("distribution_cfg") or {}
        if dist_cfg:
            policy_cfg["init_noise_std"] = dist_cfg.get("init_std", 1.0)
            policy_cfg["noise_std_type"] = dist_cfg.get("std_type", "scalar")
        train_cfg["policy"] = policy_cfg
        train_cfg.setdefault("empirical_normalization", actor_cfg.get("obs_normalization", False))
    if "empirical_normalization" not in train_cfg:
        train_cfg["empirical_normalization"] = False


def _unpack_obs(result):
    """将 TensorDict 观测适配为旧版的 ``(obs, extras)`` 格式。"""
    if isinstance(result, tuple):
        obs, extras = result
    else:
        obs, extras = result, {}
    if hasattr(obs, "keys"):
        actor_key = "actor" if "actor" in obs.keys() else "policy"
        plain_obs = obs[actor_key]
        extras.setdefault("observations", {})
        for k in obs.keys():
            if k != actor_key:
                extras["observations"][k] = obs[k]
        return plain_obs, extras
    return obs, extras


def _unpack_step(obs, rew, dones, infos):
    """将 TensorDict 的 step 输出适配为旧版格式。"""
    if hasattr(obs, "keys"):
        actor_key = "actor" if "actor" in obs.keys() else "policy"
        plain_obs = obs[actor_key]
        infos.setdefault("observations", {})
        for k in obs.keys():
            if k != actor_key:
                infos["observations"][k] = obs[k]
        return plain_obs, rew, dones, infos
    return obs, rew, dones, infos


class OnPolicyRunner:
    """用于训练和评估的同策略（On-policy）运行器。"""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu"):
        _migrate_train_cfg(train_cfg)
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        # 检查是否启用了多 GPU。
        self._configure_multi_gpu()

        # 根据算法确定训练类型。
        if self.alg_cfg["class_name"] == "PPO":
            self.training_type = "rl"
        elif self.alg_cfg["class_name"] == "Distillation":
            self.training_type = "distillation"
        else:
            raise ValueError(f"Training type not found for algorithm {self.alg_cfg['class_name']}.")

        # 确定观测维度。
        obs, extras = _unpack_obs(self.env.get_observations())
        num_obs = obs.shape[1]
        # 确定特权观测的类型。
        if self.training_type == "rl":
            if "critic" in extras["observations"]:
                self.privileged_obs_type = "critic"  # Actor-Critic 强化学习，例如 PPO。
            else:
                self.privileged_obs_type = None
        if self.training_type == "distillation":
            if "teacher" in extras["observations"]:
                self.privileged_obs_type = "teacher"  # 策略蒸馏。
            else:
                self.privileged_obs_type = None

        # 确定特权观测的维度。
        if self.privileged_obs_type is not None:
            num_privileged_obs = extras["observations"][self.privileged_obs_type].shape[1]
        else:
            num_privileged_obs = num_obs

        # 获取策略类。
        policy_class = eval(self.policy_cfg.pop("class_name"))
        policy: (
            ActorCritic
            | ActorCriticRecurrent
            | StudentTeacher
            | StudentTeacherRecurrent
            # | TransformerActorCritic
        ) = policy_class(num_obs, num_privileged_obs, self.env.num_actions, **self.policy_cfg).to(self.device)

        # 确定 RND 门控状态的维度。
        if "rnd_cfg" in self.alg_cfg and self.alg_cfg["rnd_cfg"] is not None:
            # 检查是否存在 RND 门控状态。
            rnd_state = extras["observations"].get("rnd_state")
            if rnd_state is None:
                raise ValueError("Observations for the key 'rnd_state' not found in infos['observations'].")
            # 获取 RND 门控状态的维度。
            num_rnd_state = rnd_state.shape[1]
            # 将 RND 门控状态维度加入配置。
            self.alg_cfg["rnd_cfg"]["num_states"] = num_rnd_state
            # 随时间步缩小 RND 权重（类似 legged_gym 环境中奖励随时间步缩放的方式）。
            self.alg_cfg["rnd_cfg"]["weight"] *= env.unwrapped.step_dt

        # 如果使用对称性约束，则传入环境配置对象。
        if "symmetry_cfg" in self.alg_cfg and self.alg_cfg["symmetry_cfg"] is not None:
            # 对称性函数会用它来处理不同的观测项。
            self.alg_cfg["symmetry_cfg"]["_env"] = env

        # 初始化算法。
        alg_class = eval(self.alg_cfg.pop("class_name"))
        self.alg: PPO | Distillation = alg_class(
            policy, device=self.device, **self.alg_cfg, multi_gpu_cfg=self.multi_gpu_cfg
        )

        # 保存训练配置。
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.empirical_normalization = self.cfg["empirical_normalization"]
        if self.empirical_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=[num_obs], until=1.0e8).to(self.device)
            self.privileged_obs_normalizer = EmpiricalNormalization(shape=[num_privileged_obs], until=1.0e8).to(
                self.device
            )
        else:
            self.obs_normalizer = torch.nn.Identity().to(self.device)  # 不进行归一化。
            self.privileged_obs_normalizer = torch.nn.Identity().to(self.device)  # 不进行归一化。

        # 初始化存储空间和模型。
        self.alg.init_storage(
            self.training_type,
            self.env.num_envs,
            self.num_steps_per_env,
            [num_obs],
            [num_privileged_obs],
            [self.env.num_actions],
        )

        # 判断是否禁用日志记录。
        # 仅由 rank 0 的进程（主进程）记录日志。
        self.disable_logs = self.is_distributed and self.gpu_global_rank != 0
        # 日志记录。
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):  # noqa: C901
        # 初始化日志 writer。
        if self.log_dir is not None and self.writer is None and not self.disable_logs:
            # 启动 TensorBoard 或 Neptune 与 TensorBoard 的摘要 writer，默认使用 TensorBoard。
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise ValueError("Logger type not found. Please choose 'neptune', 'wandb' or 'tensorboard'.")

        # 检查是否已加载 teacher。
        if self.training_type == "distillation" and not self.alg.policy.loaded_teacher:
            raise ValueError("Teacher model parameters not loaded. Please load a teacher model to distill.")

        # 随机化初始 episode 长度（用于探索）。
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # 开始学习。
        obs, extras = _unpack_obs(self.env.get_observations())
        privileged_obs = extras["observations"].get(self.privileged_obs_type, obs)
        obs, privileged_obs = obs.to(self.device), privileged_obs.to(self.device)
        self.train_mode()  # 切换到训练模式（例如启用 dropout）。

        # 训练过程中的统计记录。
        ep_infos = []
        rewbuffer = deque(maxlen=500)
        lenbuffer = deque(maxlen=500)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # 创建用于记录外在奖励和内在奖励的缓冲区。
        if self.alg.rnd:
            erewbuffer = deque(maxlen=500)
            irewbuffer = deque(maxlen=500)
            cur_ereward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
            cur_ireward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # 确保所有参数已经同步。
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()
            # TODO：是否需要同步经验归一化器？
            #   目前不需要，因为它们最终都应该“渐近地”收敛到相同的值。

        # 开始训练。
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()
            # Rollout 采样。
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    # 采样动作。
                    actions = self.alg.act(obs, privileged_obs)
                    # 推进一步环境。
                    obs, rewards, dones, infos = _unpack_step(*self.env.step(actions.to(self.env.device)))
                    # 将数据移动到目标设备。
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    # 执行归一化。
                    obs = self.obs_normalizer(obs)
                    if self.privileged_obs_type is not None:
                        privileged_obs = self.privileged_obs_normalizer(
                            infos["observations"][self.privileged_obs_type].to(self.device)
                        )
                    else:
                        privileged_obs = obs

                    # 处理当前环境步。
                    self.alg.process_env_step(rewards, dones, infos)

                    # 提取内在奖励（仅用于日志记录）。
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.alg.rnd else None

                    # 更新训练过程统计。
                    if self.log_dir is not None:
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])
                        # 更新奖励统计。
                        if self.alg.rnd:
                            cur_ereward_sum += rewards
                            cur_ireward_sum += intrinsic_rewards  # type: ignore
                            cur_reward_sum += rewards + intrinsic_rewards
                        else:
                            cur_reward_sum += rewards
                        # 更新 episode 长度。
                        cur_episode_length += 1
                        # 清除已完成 episode 的数据。
                        # -- 通用部分。
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                        # -- 内在奖励与外在奖励。
                        if self.alg.rnd:
                            erewbuffer.extend(cur_ereward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            irewbuffer.extend(cur_ireward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            cur_ereward_sum[new_ids] = 0
                            cur_ireward_sum[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                start = stop

                # 收集误差最大的时间步信息。
                top_error_timesteps = {}
                if hasattr(self.env, 'get_top_error_timesteps'):
                    try:
                        top_error_timesteps = self.env.get_top_error_timesteps(top_k=10)
                    except:
                        pass  # 如果方法执行失败则忽略。

                # 计算回报。
                if self.training_type == "rl":
                    self.alg.compute_returns(privileged_obs)

            # 更新策略。
            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            # 记录日志信息。
            if self.log_dir is not None and not self.disable_logs:
                # 写入日志信息。
                self.log(locals())
                # 保存模型。
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            # 清空 episode 信息。
            ep_infos.clear()
            # 保存代码状态。
            if it == start_iter and not self.disable_logs:
                # 获取所有 diff 文件。
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                # 如果可以，则将这些文件保存到 wandb。
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        # 训练结束后保存最终模型。
        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        # 计算采样数据量。
        collection_size = self.num_steps_per_env * self.env.num_envs * self.gpu_world_size
        # 更新总时间步数和总耗时。
        self.tot_timesteps += collection_size
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        # -- Episode 信息。
        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # 处理标量和零维 Tensor 类型的信息。
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                # 写入 logger 并输出到终端。
                if "/" in key:
                    self.writer.add_scalar(key, value, locs["it"])
                    ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
                else:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        mean_std = self.alg.policy.action_std.mean()
        fps = int(collection_size / (locs["collection_time"] + locs["learn_time"]))

        # -- 损失。
        for key, value in locs["loss_dict"].items():
            self.writer.add_scalar(f"Loss/{key}", value, locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])

        # -- 策略。
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])

        # -- 性能。
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

        # -- 训练。
        if len(locs["rewbuffer"]) > 0:
            # 分别记录内在奖励和外在奖励。
            if self.alg.rnd:
                self.writer.add_scalar("Rnd/mean_extrinsic_reward", statistics.mean(locs["erewbuffer"]), locs["it"])
                self.writer.add_scalar("Rnd/mean_intrinsic_reward", statistics.mean(locs["irewbuffer"]), locs["it"])
                self.writer.add_scalar("Rnd/weight", self.alg.rnd.weight, locs["it"])
            # 记录其余训练指标。
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])

            # -- 如果存在，则记录误差最大的时间步。
            if "top_error_timesteps" in locs and locs["top_error_timesteps"]:
                top_errors = locs["top_error_timesteps"]
                if top_errors.get("error_values"):
                    # 记录误差最大的 10 个时间步的误差值。
                    for i, (timestep, time_val, error_val) in enumerate(zip(
                        top_errors["timestep_indices"][:10],
                        top_errors["time_values"][:10],
                        top_errors["error_values"][:10]
                    )):
                        self.writer.add_scalar(f"TopErrors/timestep_{i+1}_index", timestep, locs["it"])
                        self.writer.add_scalar(f"TopErrors/timestep_{i+1}_time", time_val, locs["it"])
                        self.writer.add_scalar(f"TopErrors/timestep_{i+1}_error", error_val, locs["it"])

                    # 记录汇总统计信息。
                    self.writer.add_scalar("TopErrors/max_error", max(top_errors["error_values"]), locs["it"])
                    self.writer.add_scalar("TopErrors/mean_top10_error", statistics.mean(top_errors["error_values"][:10]), locs["it"])

            if self.logger_type != "wandb":  # wandb 不支持使用非整数作为 x 轴进行日志记录。
                self.writer.add_scalar("Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), self.tot_time)
                self.writer.add_scalar(
                    "Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), self.tot_time
                )

        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                    'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            # -- 损失。
            for key, value in locs["loss_dict"].items():
                log_string += f"""{f'Mean {key} loss:':>{pad}} {value:.4f}\n"""
            # -- 奖励。
            if self.alg.rnd:
                log_string += (
                    f"""{'Mean extrinsic reward:':>{pad}} {statistics.mean(locs['erewbuffer']):.2f}\n"""
                    f"""{'Mean intrinsic reward:':>{pad}} {statistics.mean(locs['irewbuffer']):.2f}\n"""
                )
            log_string += f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
            # -- Episode 信息。
            log_string += f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""

            # -- 误差最大的时间步信息。
            if "top_error_timesteps" in locs and locs["top_error_timesteps"]:
                top_errors = locs["top_error_timesteps"]
                if top_errors.get("error_values"):
                    log_string += f"""{'Top 3 error timesteps:':>{pad}} """
                    for i in range(min(3, len(top_errors["error_values"]))):
                        log_string += f"T{top_errors['timestep_indices'][i]}({top_errors['error_values'][i]:.3f}) "
                    log_string += "\n"
                    log_string += f"""{'Max error value:':>{pad}} {max(top_errors['error_values']):.4f}\n"""
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                    'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            for key, value in locs["loss_dict"].items():
                log_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Time elapsed:':>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time))}\n"""
            f"""{'ETA:':>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time / (locs['it'] - locs['start_iter'] + 1) * (
                               locs['start_iter'] + locs['num_learning_iterations'] - locs['it'])))}\n"""
        )
        print(log_string)

    def save(self, path: str, infos=None):
        # -- 保存模型。
        saved_dict = {
            "model_state_dict": self.alg.policy.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        # -- 如果使用 RND，则保存 RND 模型。
        if self.alg.rnd:
            saved_dict["rnd_state_dict"] = self.alg.rnd.state_dict()
            saved_dict["rnd_optimizer_state_dict"] = self.alg.rnd_optimizer.state_dict()
        # -- 如果使用观测归一化器，则保存其状态。
        if self.empirical_normalization:
            saved_dict["obs_norm_state_dict"] = self.obs_normalizer.state_dict()
            saved_dict["privileged_obs_norm_state_dict"] = self.privileged_obs_normalizer.state_dict()

        # 保存模型。
        torch.save(saved_dict, path)

        # 将模型上传到外部日志服务。
        if self.logger_type in ["neptune", "wandb"] and not self.disable_logs:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True):
        loaded_dict = torch.load(path, weights_only=False)
        # -- 加载模型。
        resumed_training = self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
        # -- 如果使用 RND，则加载 RND 模型。
        if self.alg.rnd:
            self.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"])
        # -- 如果使用观测归一化器，则加载其状态。
        if self.empirical_normalization:
            if resumed_training:
                # 如果是恢复之前的训练，则为 actor/student 加载 actor/student 的归一化器，
                # 并为 critic/teacher 加载 critic/teacher 的归一化器。
                self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
                self.privileged_obs_normalizer.load_state_dict(loaded_dict["privileged_obs_norm_state_dict"])
            else:
                # 如果不是恢复训练、而只是加载模型，则本次运行应是在一次 RL 训练之后进行蒸馏训练。
                # 因此，将 actor 的归一化器加载给 teacher 模型；student 的归一化器不加载，
                # 因为当前的观测空间可能与之前的 RL 训练不同。
                self.privileged_obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
        # -- 如果需要，则加载优化器。
        if load_optimizer and resumed_training:
            # -- 算法优化器。
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            # -- 如果使用 RND，则加载 RND 优化器。
            if self.alg.rnd:
                self.alg.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
        # -- 加载当前学习迭代次数。
        if resumed_training:
            self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        self.eval_mode()  # 切换到评估模式（例如关闭 dropout）。
        if device is not None:
            self.alg.policy.to(device)
        policy = self.alg.policy.act_inference
        if self.cfg["empirical_normalization"]:
            if device is not None:
                self.obs_normalizer.to(device)

            def _normed_inference(x):
                if hasattr(x, "keys"):
                    actor_key = "actor" if "actor" in x.keys() else "policy"
                    x = x[actor_key]
                return self.alg.policy.act_inference(self.obs_normalizer(x))

            policy = _normed_inference
        else:

            def _plain_inference(x):
                if hasattr(x, "keys"):
                    actor_key = "actor" if "actor" in x.keys() else "policy"
                    x = x[actor_key]
                return self.alg.policy.act_inference(x)

            policy = _plain_inference
        return policy

    def train_mode(self):
        # -- PPO。
        self.alg.policy.train()
        # -- RND。
        if self.alg.rnd:
            self.alg.rnd.train()
        # -- 归一化。
        if self.empirical_normalization:
            self.obs_normalizer.train()
            self.privileged_obs_normalizer.train()

    def eval_mode(self):
        # -- PPO。
        self.alg.policy.eval()
        # -- RND。
        if self.alg.rnd:
            self.alg.rnd.eval()
        # -- 归一化。
        if self.empirical_normalization:
            self.obs_normalizer.eval()
            self.privileged_obs_normalizer.eval()

    def add_git_repo_to_log(self, repo_file_path):
        self.git_status_repos.append(repo_file_path)

    """
    辅助函数。
    """

    def _configure_multi_gpu(self):
        """配置多 GPU 训练。"""
        # 检查是否启用了分布式训练。
        self.gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.is_distributed = self.gpu_world_size > 1

        # 如果未启用分布式训练，则将 local rank 和 global rank 设为 0 后返回。
        if not self.is_distributed:
            self.gpu_local_rank = 0
            self.gpu_global_rank = 0
            self.multi_gpu_cfg = None
            return

        # 获取 rank 和 world size。
        self.gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.gpu_global_rank = int(os.getenv("RANK", "0"))

        # 构建配置字典。
        self.multi_gpu_cfg = {
            "global_rank": self.gpu_global_rank,  # 主进程的 rank。
            "local_rank": self.gpu_local_rank,  # 当前进程的 rank。
            "world_size": self.gpu_world_size,  # 进程总数。
        }

        # 检查用户指定的设备是否与 local rank 对应。
        if self.device != f"cuda:{self.gpu_local_rank}":
            raise ValueError(
                f"Device '{self.device}' does not match expected device for local rank '{self.gpu_local_rank}'."
            )
        # 校验多 GPU 配置。
        if self.gpu_local_rank >= self.gpu_world_size:
            raise ValueError(
                f"Local rank '{self.gpu_local_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )
        if self.gpu_global_rank >= self.gpu_world_size:
            raise ValueError(
                f"Global rank '{self.gpu_global_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )

        # 初始化 PyTorch 分布式训练。
        torch.distributed.init_process_group(backend="nccl", rank=self.gpu_global_rank, world_size=self.gpu_world_size)
        # 将当前 CUDA 设备设置为 local rank 对应的设备。
        torch.cuda.set_device(self.gpu_local_rank)