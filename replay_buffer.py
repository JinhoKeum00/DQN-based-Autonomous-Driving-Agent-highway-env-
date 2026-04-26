from typing import Tuple
from collections import deque # 추가

import numpy as np
import torch


class ReplayBuffer:
    """일반적인 오프폴리시 알고리즘에서 사용할 수 있는 리플레이 버퍼.

    - obs_shape: 원본 관찰의 shape (이미지든 벡터든 그대로)
    - action_dim: 이산 행동 공간의 크기 (highway는 Discrete)
    """

    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        action_dim: int,
        capacity: int,
        device: torch.device,
    ) -> None:
        self.capacity = capacity
        self.device = device

        obs_dtype = np.float32

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, 1), dtype=np.int64)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones_no_max = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.full = False

    def __len__(self) -> int:
        return self.capacity if self.full else self.idx

    def add(
        self,
        obs,
        action: int,
        reward: float,
        next_obs,
        done: bool,
        done_no_max: bool,
    ) -> None:
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], float(not done))
        np.copyto(self.not_dones_no_max[self.idx], float(not done_no_max))

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size: int) -> dict:
        batch_size = min(len(self), batch_size)
        idxs = np.random.randint(0, len(self), size=batch_size)

        obses = torch.from_numpy(self.obses[idxs]).to(self.device)
        actions = torch.from_numpy(self.actions[idxs]).to(self.device)
        rewards = torch.from_numpy(self.rewards[idxs]).to(self.device)
        next_obses = torch.from_numpy(self.next_obses[idxs]).to(self.device)
        not_dones = torch.from_numpy(self.not_dones[idxs]).to(self.device)
        not_dones_no_max = torch.from_numpy(self.not_dones_no_max[idxs]).to(
            self.device
        )

        return {
            "obses": obses,
            "actions": actions,
            "rewards": rewards,
            "next_obses": next_obses,
            "not_dones": not_dones,
            "not_dones_no_max": not_dones_no_max,
        }


class MultiStepReplayBuffer:
    """n-step return 전용 리플레이 버퍼 (PER 제거 버전).

    - Double DQN / Dueling DQN 과 함께 사용할 수 있도록
      ReplayBuffer와 동일한 key를 사용해 sample() 을 반환.
    - priority, importance-sampling weight 같은 PER 기능은 포함하지 않는다.
    """

    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        action_dim: int,
        capacity: int,
        device: torch.device,
        n_step: int = 3,
        gamma: float = 0.99,
    ) -> None:
        """
        n_step: multi-step TD 길이
        gamma: discount factor
        """
        self.capacity = capacity
        self.device = device

        self.n_step = n_step
        self.gamma = gamma

        obs_dtype = np.float32

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)

        self.actions = np.empty((capacity, 1), dtype=np.int64)
        # multi-step return 을 reward 자리에 저장
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones_no_max = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.full = False

        # n-step transition 생성을 위한 버퍼
        self.n_step_buffer = deque(maxlen=n_step)

    def __len__(self) -> int:
        return self.capacity if self.full else self.idx

    # ---------------- n-step 내부 유틸 ---------------- #

    def _get_n_step_info(self):
        """n-step return 과 마지막 next_obs / done 정보 계산."""
        R = 0.0
        discount = 1.0
        next_obs = None
        done = False
        done_no_max = False

        for (obs, action, reward, next_o, d, d_no_max) in self.n_step_buffer:
            R += discount * reward
            discount *= self.gamma
            next_obs = next_o
            done = d
            done_no_max = d_no_max
            if done:
                break

        first_obs, first_action, _, _, _, _ = self.n_step_buffer[0]
        return first_obs, first_action, R, next_obs, done, done_no_max

    def _add_single_transition(
        self,
        obs,
        action: int,
        reward: float,
        next_obs,
        done: bool,
        done_no_max: bool,
    ) -> None:
        """이미 n-step 이 계산된 transition 을 실제 버퍼에 저장."""
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], float(not done))
        np.copyto(self.not_dones_no_max[self.idx], float(not done_no_max))

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    # ---------------- public API ---------------- #

    def add(
        self,
        obs,
        action: int,
        reward: float,
        next_obs,
        done: bool,
        done_no_max: bool,
    ) -> None:
        """1-step transition 을 받아 n-step return 으로 변환 후 저장."""
        self.n_step_buffer.append(
            (obs, action, reward, next_obs, done, done_no_max)
        )

        # n_step 만큼 쌓이면, 맨 앞에서 시작하는 n-step transition 하나 생성
        if len(self.n_step_buffer) == self.n_step:
            first_obs, first_action, R, n_next_obs, n_done, n_done_no_max = (
                self._get_n_step_info()
            )
            self._add_single_transition(
                first_obs,
                first_action,
                R,
                n_next_obs,
                n_done,
                n_done_no_max,
            )
            self.n_step_buffer.popleft()

        # 에피소드 끝이면 나머지 step 들에 대해서도 truncated n-step 추가
        if done:
            while len(self.n_step_buffer) > 0:
                first_obs, first_action, R, n_next_obs, n_done, n_done_no_max = (
                    self._get_n_step_info()
                )
                self._add_single_transition(
                    first_obs,
                    first_action,
                    R,
                    n_next_obs,
                    n_done,
                    n_done_no_max,
                )
                self.n_step_buffer.popleft()

    def sample(self, batch_size: int) -> dict:
        """uniform 샘플링으로 미니배치 반환."""
        length = len(self)
        if length == 0:
            raise ValueError("Replay buffer is empty!")
        batch_size = min(length, batch_size)

        indexes = np.random.randint(0, length, size=batch_size)

        obses = torch.from_numpy(self.obses[indexes]).to(self.device)
        actions = torch.from_numpy(self.actions[indexes]).to(self.device)
        rewards = torch.from_numpy(self.rewards[indexes]).to(self.device)
        next_obses = torch.from_numpy(self.next_obses[indexes]).to(self.device)
        not_dones = torch.from_numpy(self.not_dones[indexes]).to(self.device)
        not_dones_no_max = torch.from_numpy(
            self.not_dones_no_max[indexes]
        ).to(self.device)

        return {
            "obses": obses,
            "actions": actions,
            "rewards": rewards,
            "next_obses": next_obses,
            "not_dones": not_dones,
            "not_dones_no_max": not_dones_no_max,
        }
