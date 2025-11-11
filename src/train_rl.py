"""
Phase 3: Reinforcement Learning
PPO-based training for trading optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from pathlib import Path
from tqdm import tqdm
import yaml
from typing import Dict, List, Tuple
from collections import deque

from model import NatronTransformer


class TradingEnvironment:
    """
    Trading environment for RL training.
    """
    
    def __init__(
        self,
        data_x: np.ndarray,
        data_y: Dict[str, np.ndarray],
        config: Dict
    ):
        """
        Args:
            data_x: Feature sequences (n_samples, seq_len, n_features)
            data_y: Labels dictionary
            config: Configuration dictionary
        """
        self.data_x = data_x
        self.data_y = data_y
        self.config = config
        
        self.initial_balance = config['rl']['env']['initial_balance']
        self.transaction_cost = config['rl']['env']['transaction_cost']
        self.max_position = config['rl']['env']['max_position_size']
        
        # Reward parameters
        self.profit_scale = config['rl']['reward']['profit_scale']
        self.turnover_penalty = config['rl']['reward']['turnover_penalty']
        self.drawdown_penalty = config['rl']['reward']['drawdown_penalty']
        self.holding_penalty = config['rl']['reward']['holding_penalty']
        
        # State
        self.reset()
    
    def reset(self, start_idx: int = None):
        """Reset environment"""
        if start_idx is None:
            self.current_idx = 0
        else:
            self.current_idx = start_idx
        
        self.balance = self.initial_balance
        self.position = 0.0  # -1 (short), 0 (neutral), +1 (long)
        self.entry_price = 0.0
        self.max_balance = self.initial_balance
        self.total_trades = 0
        
        return self._get_state()
    
    def _get_state(self) -> torch.Tensor:
        """Get current state"""
        if self.current_idx >= len(self.data_x):
            return None
        return torch.from_numpy(self.data_x[self.current_idx]).float()
    
    def step(self, action: int) -> Tuple[torch.Tensor, float, bool, Dict]:
        """
        Take action in environment.
        
        Args:
            action: 0 (sell/close), 1 (hold), 2 (buy/open)
            
        Returns:
            next_state, reward, done, info
        """
        if self.current_idx >= len(self.data_x) - 1:
            return None, 0.0, True, {}
        
        # Get current price (from direction label, we infer price movement)
        # This is simplified - in real trading, use actual price data
        direction = self.data_y['direction'][self.current_idx]
        
        # Simulate price change (simplified)
        if direction == 1:  # Up
            price_change = 0.01
        elif direction == 0:  # Down
            price_change = -0.01
        else:  # Neutral
            price_change = 0.0
        
        reward = 0.0
        
        # Execute action
        old_position = self.position
        
        if action == 0:  # Sell/Close
            if self.position > 0:  # Close long
                profit = price_change * self.position
                self.balance += profit - self.transaction_cost
                reward += self.profit_scale * profit
                self.position = 0
                self.total_trades += 1
        
        elif action == 1:  # Hold
            if self.position != 0:
                # Apply holding penalty
                reward -= self.holding_penalty
                # Accumulate unrealized P&L
                unrealized_pnl = price_change * self.position
                reward += self.profit_scale * unrealized_pnl * 0.1  # Partial credit
        
        elif action == 2:  # Buy/Open
            if self.position == 0:  # Open long
                self.position = self.max_position
                self.entry_price = 1.0  # Normalized
                self.balance -= self.transaction_cost
                self.total_trades += 1
        
        # Turnover penalty
        if old_position != self.position:
            reward -= self.turnover_penalty
        
        # Drawdown penalty
        if self.balance < self.max_balance:
            drawdown = (self.max_balance - self.balance) / self.max_balance
            reward -= self.drawdown_penalty * drawdown
        else:
            self.max_balance = self.balance
        
        # Move to next step
        self.current_idx += 1
        next_state = self._get_state()
        done = next_state is None
        
        info = {
            'balance': self.balance,
            'position': self.position,
            'trades': self.total_trades
        }
        
        return next_state, reward, done, info


class PPOAgent:
    """
    Proximal Policy Optimization Agent.
    """
    
    def __init__(
        self,
        model: NatronTransformer,
        config: Dict,
        device: str = 'cuda'
    ):
        self.device = device
        self.config = config
        self.model = model.to(device)
        
        # PPO policy head (3 actions: sell, hold, buy)
        self.policy_head = nn.Sequential(
            nn.Linear(model.d_model, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 3)
        ).to(device)
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(model.d_model, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        ).to(device)
        
        # Optimizer
        params = list(self.model.parameters()) + \
                 list(self.policy_head.parameters()) + \
                 list(self.value_head.parameters())
        
        self.optimizer = torch.optim.Adam(
            params,
            lr=config['rl']['learning_rate']
        )
        
        # PPO parameters
        self.gamma = config['rl']['gamma']
        self.gae_lambda = config['rl']['gae_lambda']
        self.clip_epsilon = config['rl']['clip_epsilon']
        self.entropy_coef = config['rl']['entropy_coef']
        self.value_loss_coef = config['rl']['value_loss_coef']
        self.max_grad_norm = config['rl']['max_grad_norm']
        
        # Checkpoint directory
        self.checkpoint_dir = Path(config['rl']['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def get_action(self, state: torch.Tensor) -> Tuple[int, float, float]:
        """
        Get action from policy.
        
        Returns:
            action, log_prob, value
        """
        state = state.unsqueeze(0).to(self.device)  # Add batch dimension
        
        with torch.no_grad():
            embeddings = self.model.get_embeddings(state)
            logits = self.policy_head(embeddings)
            value = self.value_head(embeddings).squeeze(-1)
            
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item(), value.item()
    
    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation.
        """
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        returns = [adv + val for adv, val in zip(advantages, values)]
        
        return np.array(advantages), np.array(returns)
    
    def update(self, rollout_buffer: Dict):
        """
        Update policy using PPO.
        """
        states = torch.stack(rollout_buffer['states']).to(self.device)
        actions = torch.tensor(rollout_buffer['actions']).to(self.device)
        old_log_probs = torch.tensor(rollout_buffer['log_probs']).to(self.device)
        advantages = torch.tensor(rollout_buffer['advantages']).float().to(self.device)
        returns = torch.tensor(rollout_buffer['returns']).float().to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO epochs
        for _ in range(self.config['rl']['ppo_epochs']):
            # Forward pass
            embeddings = self.model.get_embeddings(states)
            logits = self.policy_head(embeddings)
            values = self.value_head(embeddings).squeeze(-1)
            
            # Policy loss
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(values, returns)
            
            # Total loss
            loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item()
        }
    
    def train(self, env: TradingEnvironment, n_episodes: int):
        """
        Train PPO agent.
        """
        print("\n" + "="*60)
        print("🚀 Phase 3: Reinforcement Learning (PPO)")
        print("="*60)
        print(f"Episodes: {n_episodes}")
        print(f"Device: {self.device}")
        print("="*60 + "\n")
        
        best_reward = -float('inf')
        
        for episode in range(1, n_episodes + 1):
            # Collect rollout
            rollout_buffer = {
                'states': [],
                'actions': [],
                'log_probs': [],
                'rewards': [],
                'values': [],
                'dones': []
            }
            
            state = env.reset()
            episode_reward = 0
            
            while state is not None:
                action, log_prob, value = self.get_action(state)
                next_state, reward, done, info = env.step(action)
                
                rollout_buffer['states'].append(state)
                rollout_buffer['actions'].append(action)
                rollout_buffer['log_probs'].append(log_prob)
                rollout_buffer['rewards'].append(reward)
                rollout_buffer['values'].append(value)
                rollout_buffer['dones'].append(done)
                
                episode_reward += reward
                state = next_state
            
            # Compute advantages
            advantages, returns = self.compute_gae(
                rollout_buffer['rewards'],
                rollout_buffer['values'],
                rollout_buffer['dones']
            )
            
            rollout_buffer['advantages'] = advantages
            rollout_buffer['returns'] = returns
            
            # Update policy
            update_info = self.update(rollout_buffer)
            
            # Logging
            if episode % 10 == 0:
                print(f"Episode {episode}/{n_episodes}")
                print(f"   Reward: {episode_reward:.2f}")
                print(f"   Balance: {info['balance']:.2f}")
                print(f"   Trades: {info['trades']}")
                print(f"   Policy Loss: {update_info['policy_loss']:.4f}")
            
            # Save best model
            if episode_reward > best_reward:
                best_reward = episode_reward
                self.save_checkpoint(episode, episode_reward, is_best=True)
        
        print("\n✅ RL training complete!")
    
    def save_checkpoint(self, episode: int, reward: float, is_best: bool = False):
        """Save checkpoint"""
        checkpoint = {
            'episode': episode,
            'model_state_dict': self.model.state_dict(),
            'policy_head_state_dict': self.policy_head.state_dict(),
            'value_head_state_dict': self.value_head.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'reward': reward
        }
        
        if is_best:
            path = self.checkpoint_dir / 'rl_best.pt'
            torch.save(checkpoint, path)
            print(f"💾 Saved best RL checkpoint: {path}")


if __name__ == "__main__":
    print("Phase 3: Reinforcement Learning Module")
    print("Run main.py for full training pipeline")
