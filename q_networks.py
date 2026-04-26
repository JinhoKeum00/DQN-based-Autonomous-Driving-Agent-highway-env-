from typing import Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor = None,
    dropout: nn.Module = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Scaled Dot Product Attention 계산
    
    Args:
        query: (batch, heads, 1, features_per_head) - Ego만의 query
        key: (batch, heads, n_entities, features_per_head) - 모든 차량의 key
        value: (batch, heads, n_entities, features_per_head) - 모든 차량의 value
        mask: (batch, heads, 1, n_entities) - presence mask
        dropout: dropout layer
    
    Returns:
        output: (batch, heads, 1, features_per_head)
        attention_weights: (batch, heads, 1, n_entities)
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)
    
    attention_weights = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        attention_weights = dropout(attention_weights)
    
    output = torch.matmul(attention_weights, value)
    return output, attention_weights


class EgoAttention(nn.Module):
    """
    Social Attention: Ego 중심의 Multi-head Attention
    
    Ego 차량만 Query를 생성하고, 모든 차량(ego + 주변)에 대해 Attention을 계산
    이를 통해 Ego가 주변 차량들을 어떻게 "주목"하는지 모델링
    """
    
    def __init__(
        self,
        feature_size: int = 64,
        heads: int = 4,
        dropout_factor: float = 0.0,
    ):
        super().__init__()
        self.feature_size = feature_size
        self.heads = heads
        self.dropout_factor = dropout_factor
        self.features_per_head = feature_size // heads
        
        assert (
            feature_size % heads == 0
        ), f"feature_size({feature_size})은 heads({heads})로 나누어떨어져야 합니다."
        
        # Query는 Ego만 생성
        self.query_ego = nn.Linear(feature_size, feature_size, bias=False)
        
        # Key와 Value는 모든 차량(ego + others)에서 생성
        self.key_all = nn.Linear(feature_size, feature_size, bias=False)
        self.value_all = nn.Linear(feature_size, feature_size, bias=False)
        
        # Attention 결과를 원래 차원으로 복원하고 residual connection
        self.fc_out = nn.Linear(feature_size, feature_size, bias=False)
        
        self.dropout = nn.Dropout(dropout_factor)
    
    def forward(
        self,
        ego: torch.Tensor,
        others: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            ego: (batch, 1, feature_size) - Ego 차량의 embedding
            others: (batch, n_others, feature_size) - 주변 차량들의 embedding
            mask: (batch, n_entities) - presence mask (True면 무시)
        
        Returns:
            output: (batch, feature_size) - Ego의 final embedding
            attention_weights: (batch, heads, n_entities) - Attention weight
        """
        batch_size = ego.shape[0]
        
        # Ego와 others를 결합 (first position은 ego)
        all_entities = torch.cat([ego, others], dim=1)  # (batch, 1+n_others, feature_size)
        n_entities = all_entities.shape[1]
        
        # Query: Ego만
        query = self.query_ego(ego)  # (batch, 1, feature_size)
        
        # Key, Value: 모든 차량
        key = self.key_all(all_entities)  # (batch, n_entities, feature_size)
        value = self.value_all(all_entities)  # (batch, n_entities, feature_size)
        
        # Multi-head로 reshape
        # (batch, seq_len, feature_size) -> (batch, seq_len, heads, features_per_head)
        query = query.view(
            batch_size, 1, self.heads, self.features_per_head
        ).transpose(1, 2)  # (batch, heads, 1, features_per_head)
        
        key = key.view(
            batch_size, n_entities, self.heads, self.features_per_head
        ).transpose(1, 2)  # (batch, heads, n_entities, features_per_head)
        
        value = value.view(
            batch_size, n_entities, self.heads, self.features_per_head
        ).transpose(1, 2)  # (batch, heads, n_entities, features_per_head)
        
        # Presence mask를 multi-head 형태로 변환
        if mask is not None:
            # mask: (batch, n_entities) -> (batch, 1, 1, n_entities)
            mask = mask.view(batch_size, 1, 1, n_entities).expand(
                batch_size, self.heads, 1, n_entities
            )
        
        # Attention 계산
        attn_output, attention_weights = scaled_dot_product_attention(
            query, key, value, mask=mask, dropout=self.dropout
        )  # attn_output: (batch, heads, 1, features_per_head)
        
        # Multi-head 결과를 원래 차원으로 되돌림
        attn_output = attn_output.transpose(1, 2).contiguous()  # (batch, 1, heads, features_per_head)
        attn_output = attn_output.view(batch_size, 1, self.feature_size)  # (batch, 1, feature_size)
        
        # Output projection
        output = self.fc_out(attn_output)  # (batch, 1, feature_size)
        
        # Residual connection (skip connection)
        output = (output + ego) / 2
        
        # Batch dimension 제거 및 attention weight squeeze
        output = output.squeeze(1)  # (batch, feature_size)
        attention_weights = attention_weights.squeeze(2)  # (batch, heads, n_entities)
        
        return output, attention_weights


class NoisyLinear(nn.Module):
    """NoisyNet용 Linear layer (factorized Gaussian noise)."""

    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    def reset_noise(self) -> None:
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        # outer product
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    @staticmethod
    def _scale_noise(size: int) -> torch.Tensor:
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)

class DuelingQNetwork(nn.Module):
    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        action_dim: int,
        hidden_dim: int = 128,
        hidden_depth: int = 2,
        num_heads: int = 4,
        noisy_std: float = 0.5,
    ) -> None:
        super().__init__()

        # obs_shape는 (N, F) 형태 (frame stacking 없음)
        if len(obs_shape) == 2:
            self.num_vehicles, self.feature_dim = obs_shape
            self.num_frames = 1  # 단일 obs 사용
        else:
            raise ValueError(
                f"DuelingQNetwork는 (num_vehicles, feature_dim) 형태의 obs_shape을 기대합니다. obs_shape={obs_shape}"
            )

        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.hidden_depth = hidden_depth
        self.num_heads = num_heads

        # 각 차량 feature를 같은 차원으로 임베딩
        self.entity_encoder = nn.Linear(self.feature_dim, hidden_dim)

        # Social Attention: Ego 중심의 Multi-head Attention
        self.ego_attention = EgoAttention(
            feature_size=hidden_dim,
            heads=num_heads,
            dropout_factor=0.0,
        )

        # hidden_depth를 이용해서 MLP 깊이를 정함
        post_layers = []
        in_dim = hidden_dim
        for _ in range(hidden_depth):
            post_layers.append(nn.ReLU())
            post_layers.append(nn.Linear(in_dim, hidden_dim))
            in_dim = hidden_dim
        self.post_fc = nn.Sequential(*post_layers)

        # ----- Dueling 구조: Value / Advantage stream ----- #
        self.value_stream = nn.Sequential(
            NoisyLinear(hidden_dim, hidden_dim, sigma_init=noisy_std),
            nn.ReLU(),
            NoisyLinear(hidden_dim, 1, sigma_init=noisy_std),
        )

        self.advantage_stream = nn.Sequential(
            NoisyLinear(hidden_dim, hidden_dim, sigma_init=noisy_std),
            nn.ReLU(),
            NoisyLinear(hidden_dim, action_dim, sigma_init=noisy_std),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs:
          - (batch, N, F)  또는
          - (N, F)

        여기서:
          N = self.num_vehicles
          F = self.feature_dim
        """
        # 차원 보정
        if obs.dim() == 2:
            # (N, F) → (1, N, F)
            obs = obs.unsqueeze(0)
        elif obs.dim() != 3:
            raise ValueError(f"지원하지 않는 obs 차원: {obs.shape}")

        B, N, F = obs.shape
        assert N == self.num_vehicles and F == self.feature_dim, \
            f"입력 shape (N={N}, F={F})가 네트워크 설정 (N={self.num_vehicles}, F={self.feature_dim})과 다릅니다."

        # 1) 각 차량 feature embedding
        # obs: (B, N, F) → (B, N, hidden_dim)
        x = self.entity_encoder(obs)    # (B, N, hidden_dim)

        # 2) Presence mask 생성 (Highway-env의 Kinematics 관찰 기준)
        # obs의 첫 번째 feature는 presence (0: 부재, 1: 존재)
        # presence: (B, N) - True면 부재한 차량
        presence = obs[:, :, 0]  # (B, N) - presence feature
        presence_mask = presence < 0.5  # (B, N) - True면 무시할 차량
        
        # 3) Ego Attention (Social Attention) 적용
        # ego만 query, 모든 차량이 key/value
        ego = x[:, 0:1, :]                      # (B, 1, hidden_dim)
        others = x[:, 1:, :]                    # (B, N-1, hidden_dim)
        
        # Presence mask를 다시 전체 차량 기준으로 (ego + others)
        # ego는 항상 존재하므로 첫 번째는 False, 나머지는 presence_mask 사용
        full_presence_mask = presence_mask  # (B, N) - 전체 차량의 presence mask
        
        attn_out, attention_weights = self.ego_attention(ego, others, mask=full_presence_mask)
        # attn_out: (B, hidden_dim)
        # attention_weights: (B, heads, N)

        # 4) 후처리 MLP
        h = self.post_fc(attn_out)               # (B, hidden_dim)

        # 6) Dueling: V(s), A(s,a)
        value = self.value_stream(h)            # (B, 1)
        adv = self.advantage_stream(h)          # (B, action_dim)

        adv_mean = adv.mean(dim=1, keepdim=True)
        q_values = value + adv - adv_mean       # (B, action_dim)

        return q_values
