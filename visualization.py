from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from env_helpers import make_env


def record_episode_video(
    env_id: str,
    agent,
    obs_preprocessor: Callable,
    video_path: str = "episode.mp4",
    fps: int = 30,
    max_steps: int = 1000,
) -> str:
    """단일 에피소드를 실행하며 영상을 mp4로 저장.
    
    영상 저장 시에는 프레임 스킵 없이 매 프레임마다 액션을 선택하여 정상적으로 동작을 확인합니다.

    - env_id: highway-v0, merge-v0 등
    - agent: act(obs, sample=False)를 구현한 에이전트
    - obs_preprocessor: 원시 obs를 전처리하는 함수
    - video_path: 저장할 mp4 경로
    """
    try:
        from moviepy.editor import ImageSequenceClip  # type: ignore
    except Exception as e:  # pragma: no cover - 의존성 누락 시 안내
        raise ImportError(
            "record_episode_video를 사용하려면 'moviepy'가 필요합니다. "
            "pip install moviepy imageio imageio-ffmpeg 로 설치하세요."
        ) from e

    env = make_env(env_id, render_mode="rgb_array")
    
    # 프레임 스택 버퍼 초기화
    if hasattr(obs_preprocessor, "reset"):
        obs_preprocessor.reset()
    
    obs, info = env.reset()
    proc_obs = obs_preprocessor(obs)
    done = False
    frames = []
    steps = 0

    # 영상 저장 시에는 프레임 스킵 없이 매 프레임마다 액션 선택 (정상적으로 동작 확인)
    while not done and steps < max_steps:
        frame = env.render()
        frames.append(np.array(frame))

        action = agent.act_greedy(proc_obs) if hasattr(agent, 'act_greedy') else agent.act(proc_obs, sample=False)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        proc_obs = obs_preprocessor(next_obs)
        steps += 1

    env.close()

    Path(video_path).parent.mkdir(parents=True, exist_ok=True)
    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(video_path, codec="libx264")

    return video_path



