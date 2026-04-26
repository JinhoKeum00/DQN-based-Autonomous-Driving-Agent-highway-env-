from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from algorithms.base_agent import Agent, ReplayBufferLike
from networks.q_networks import DuelingQNetwork


@dataclass
class DQNConfig:
    gamma: float = 0.995
    lr: float = 1e-3
    batch_size: int = 64
    replay_buffer_capacity: int = 100_000

    # epsilon-greedy 탐험 관련
    eps_start: float = 0.7
    eps_end: float = 0.05
    eps_decay: int = 100_000

    # target network 업데이트 / train 빈도
    target_update_interval: int = 1_000
    train_freq: int = 1
    warmup_steps: int = 1_000

    max_grad_norm: float = 10.0
    hidden_dim: int = 256
    hidden_depth: int = 2

    # multi-step
    n_step: int = 5


class DQNAgent(Agent):
    """Double DQN + Dueling + n-step 학습을 하는 에이전트 (PER 제거 버전)."""

    def __init__(
        self,
        obs_shape,
        action_dim: int,
        device: torch.device,
        cfg: DQNConfig,
    ) -> None:
        super().__init__()

        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.device = device
        self.cfg = cfg
        self.total_steps = 0

        # ----- Q 네트워크 (social attention + dueling + noisy) ----- #
        self.q_network = DuelingQNetwork(
            obs_shape=obs_shape,
            action_dim=action_dim,
            hidden_dim=cfg.hidden_dim,
            hidden_depth=cfg.hidden_depth,
            num_heads=getattr(cfg, 'num_heads', 4),
            noisy_std=getattr(cfg, 'noisy_std', 0.5),
        ).to(device)
        # ----- Target Q 네트워크 ----- #
        self.target_q_network = DuelingQNetwork(
            obs_shape=obs_shape,
            action_dim=action_dim,
            hidden_dim=cfg.hidden_dim,
            hidden_depth=cfg.hidden_depth,
            num_heads=getattr(cfg, 'num_heads', 4),
            noisy_std=getattr(cfg, 'noisy_std', 0.5),
        ).to(device)

        self.target_q_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = Adam(self.q_network.parameters(), lr=cfg.lr)

        self.epsilon = cfg.eps_start
        
        # 디버깅 정보 저장용
        self._last_q_mean = 0.0
        self._last_target_q_mean = 0.0
        self._last_grad_norm = 0.0

    # ------------------------------------------------------------------ #
    # Agent 인터페이스 구현
    # ------------------------------------------------------------------ #

    def train(self, training: bool = True) -> None:
        self.q_network.train(training)
        # target network 는 항상 eval 모드
        self.target_q_network.train(False)

    def reset(self) -> None:
        """에피소드 시작할 때 특별히 초기화할 상태는 없음."""
        pass

    # ------------------------------------------------------------------ #
    # Epsilon-greedy policy
    # ------------------------------------------------------------------ #

    def _update_epsilon(self, step: int) -> None:
        """선형으로 epsilon을 감소시키는 예시 구현.

        step 인자는 warmup 이후의 경과 step 수를 의미한다고 가정한다.
        (즉, warmup 동안에는 epsilon decay가 일어나지 않는다.)
        """
        eps_start, eps_end, eps_decay = (
            self.cfg.eps_start,
            self.cfg.eps_end,
            self.cfg.eps_decay,
        )

        fraction = min(float(step) / eps_decay, 1.0)
        self.epsilon = eps_start + fraction * (eps_end - eps_start)

    def _obs_to_tensor(self, obs: Any) -> torch.Tensor:
        """공통 관찰 → 텐서 변환 유틸."""
        if isinstance(obs, np.ndarray):
            obs_array = obs
        else:
            obs_array = np.array(obs, dtype=np.float32)

        obs_tensor = torch.from_numpy(obs_array).float().to(self.device)
        obs_tensor = obs_tensor.view(1, *self.obs_shape)
        return obs_tensor

    def act(self, obs: Any, sample: bool = False) -> int:
        """학습 시 사용하는 epsilon-greedy 정책.

        - total_steps / epsilon 스케줄을 업데이트한다.
        - sample=True 일 때만 epsilon-greedy 탐색을 수행한다.
        """
        # step 증가에 따른 epsilon 업데이트 (학습 단계에서만 호출된다고 가정)
        self.total_steps += 1
        # warmup 기간 동안에는 epsilon decay를 하지 않고 eps_start를 유지
        if self.total_steps < self.cfg.warmup_steps:
            self.epsilon = self.cfg.eps_start
        else:
            # warmup 이후부터 decay 시작: warmup 이후 경과 step 기준으로 epsilon 갱신
            effective_step = self.total_steps - self.cfg.warmup_steps
            self._update_epsilon(effective_step)

        obs_tensor = self._obs_to_tensor(obs)

        # 탐험
        if sample and np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
            return int(action)

        # exploitation
        with torch.no_grad():
            q_values = self.q_network(obs_tensor)
            action = int(torch.argmax(q_values, dim=-1).item())
        return action

    def act_greedy(self, obs: Any) -> int:
        """평가 시 사용하는 완전 greedy 정책.

        - total_steps / epsilon 스케줄을 전혀 건드리지 않는다.
        - 항상 argmax Q(s, a)를 선택한다.
        """
        obs_tensor = self._obs_to_tensor(obs)
        with torch.no_grad():
            q_values = self.q_network(obs_tensor)
            action = int(torch.argmax(q_values, dim=-1).item())
        return action

    # ------------------------------------------------------------------ #
    # 학습 (Double DQN + n-step)
    # ------------------------------------------------------------------ #

    def update(self, replay_buffer: ReplayBufferLike, step: int) -> Optional[float]:
        """리플레이 버퍼에서 샘플을 뽑아 파라미터를 업데이트하고 loss를 반환."""
        # warmup 이전이면 학습하지 않음
        if step < self.cfg.warmup_steps:
            return None

        # train_freq 에 맞는 step 에서만 업데이트
        if step % self.cfg.train_freq != 0:
            return None

        sample = replay_buffer.sample(self.cfg.batch_size)

        loss = self._update_q_network(
            obs=sample["obses"],
            actions=sample["actions"],
            rewards=sample["rewards"],
            next_obs=sample["next_obses"],
            not_dones=sample["not_dones"],
            not_dones_no_max=sample["not_dones_no_max"],
            step=step,
        )

        return float(loss.item())

    def _update_q_network(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        not_dones: torch.Tensor,
        not_dones_no_max: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        """Double DQN + n-step 업데이트 (PER 없음)."""


        # 현재 Q(s,a)
        q_values = self.q_network(obs)                      # (B, A)
        q_values = q_values.gather(1, actions.long())       # (B, 1)

        # ---------------- Double DQN target ---------------- #
        with torch.no_grad():
            # online 네트워크로 next action 선택
            next_q_online = self.q_network(next_obs)        # (B, A)
            next_actions = next_q_online.argmax(dim=1, keepdim=True)  # (B, 1)

            # target 네트워크로 그 action 의 Q 값 평가
            next_q_target = self.target_q_network(next_obs)           # (B, A)
            next_q_values = next_q_target.gather(1, next_actions)     # (B, 1)

            # n-step 이므로 gamma^n 을 사용
            gamma_n = self.cfg.gamma ** self.cfg.n_step
            target_q = rewards + gamma_n * not_dones_no_max * next_q_values

        # TD error 에 대한 MSE loss (uniform 샘플링이므로 IS weight 없음)
        loss = F.mse_loss(q_values, target_q)

        # ----------------- 최적화 -------------------------- #
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient norm 계산 및 클리핑
        grad_norm = nn.utils.clip_grad_norm_(self.q_network.parameters(), self.cfg.max_grad_norm)
        self.optimizer.step()

        # ----------------- Target network 업데이트 ---------- #
        if step % self.cfg.target_update_interval == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())

        # 디버깅 정보 저장 (선택적)
        self._last_q_mean = float(q_values.mean().item())
        self._last_target_q_mean = float(target_q.mean().item())
        self._last_grad_norm = float(grad_norm.item())

        return loss
