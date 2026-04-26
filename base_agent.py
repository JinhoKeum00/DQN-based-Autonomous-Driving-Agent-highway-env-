import abc
from typing import Any, Protocol


class ReplayBufferLike(Protocol):
    """타입 힌트를 위한 간단한 프로토콜."""

    def sample(self, batch_size: int) -> dict:
        ...


class Agent(abc.ABC):
    """모든 RL 에이전트가 따를 공통 인터페이스.

    이 인터페이스만 지키면 알고리즘 구현이 무엇이든
    동일한 학습/검증 코드에서 사용할 수 있습니다.
    """

    def reset(self) -> None:
        """에피소드 시작 시 상태를 초기화할 때 사용(필요 없으면 pass)."""
        pass

    def train(self, training: bool = True) -> None:
        """학습/평가 모드 전환 (PyTorch의 model.train()/eval()과 유사)."""
        pass

    @abc.abstractmethod
    def act(self, obs: Any, sample: bool = False) -> Any:
        """관찰 obs를 받아 행동을 반환.

        obs 형태는 환경/전처리에 따라 달라질 수 있으므로 Any로 둡니다.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def update(self, replay_buffer: ReplayBufferLike, step: int) -> None:
        """리플레이 버퍼에서 샘플을 뽑아 파라미터를 업데이트."""
        raise NotImplementedError
