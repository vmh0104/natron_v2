"""
Natron Phase 3: Reinforcement Learning (Optional)
Uses PPO for trading policy optimization
Reward: profit - α × turnover - β × drawdown
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Dict, Tuple, List
import numpy as np
from tqdm import tqdm
import os


class TradingEnvironment:
    """
    Simulated trading environment for RL training
    """
    
    def __init__(
        self,
        data: np.ndarray,
        initial_balance: float = 10000.0,
        transaction_cost: float = 0.001
    ):
        """
        Args:
            data: (N, seq_len, features) - trading data
            initial_balance: Starting capital
            transaction_cost: Commission per trade
        """
        self.data = data
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        
        self.n_samples = len(data)
        self.reset()
    
    def reset(self):
        """Reset environment to initial state"""
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0  # 0: neutral, 1: long, -1: short
        self.entry_price = 0
        self.total_profit = 0
        self.num_trades = 0
        self.max_balance = self.initial_balance
        self.max_drawdown = 0
        
        return self.data[0]
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute action in environment
        
        Args:
            action: 0=hold, 1=buy, 2=sell
            
        Returns:
            observation, reward, done, info
        """
        # Get current and next price (simplified - use close price)
        current_obs = self.data[self.current_step]
        
        # Simulate price change (for reward calculation)
        price_change = np.random.randn() * 0.001  # Placeholder
        
        reward = 0
        info = {}
        
        # Execute action
        if action == 1 and self.position == 0:  # Buy
            self.position = 1
            self.entry_price = 1.0  # Normalized price
            self.num_trades += 1
            reward -= self.transaction_cost
            
        elif action == 2 and self.position == 1:  # Sell
            # Calculate profit
            profit = (1.0 - self.entry_price) * self.balance
            self.balance += profit
            self.total_profit += profit
            self.position = 0
            self.num_trades += 1
            reward += profit - self.transaction_cost
        
        # Update drawdown
        self.max_balance = max(self.max_balance, self.balance)
        drawdown = (self.max_balance - self.balance) / self.max_balance
        self.max_drawdown = max(self.max_drawdown, drawdown)
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= self.n_samples - 1
        
        next_obs = self.data[self.current_step] if not done else current_obs
        
        info = {
            'balance': self.balance,
            'position': self.position,
            'profit': self.total_profit,
            'num_trades': self.num_trades,
            'drawdown': self.max_drawdown
        }
        
        return next_obs, reward, done, info


class PPOMemory:
    """Experience replay buffer for PPO"""
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def store(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool
    ):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def get(self):
        return (
            self.states,
            self.actions,
            self.rewards,
            self.values,
            self.log_probs,
            self.dones
        )


class ActorCritic(nn.Module):
    """Actor-Critic network for PPO"""
    
    def __init__(self, model, n_actions: int = 3):
        super().__init__()
        self.model = model
        d_model = model.d_model
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, n_actions)
        )
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )
    
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (batch, seq, features)
            
        Returns:
            action_probs, value
        """
        # Get features from model
        features = self.model.get_encoder_features(x)
        
        # Pool features
        pooled = features.mean(dim=1)
        
        # Actor: action probabilities
        action_logits = self.actor(pooled)
        action_probs = F.softmax(action_logits, dim=-1)
        
        # Critic: state value
        value = self.critic(pooled)
        
        return action_probs, value


class RLTrainer:
    """Phase 3: Reinforcement Learning with PPO"""
    
    def __init__(
        self,
        model,
        config: Dict,
        device: str = 'cuda'
    ):
        self.config = config
        self.device = device
        
        rl_config = config.get('training', {}).get('rl', {})
        self.episodes = rl_config.get('episodes', 1000)
        self.gamma = rl_config.get('gamma', 0.99)
        self.reward_alpha = rl_config.get('reward_alpha', 0.01)
        self.reward_beta = rl_config.get('reward_beta', 0.05)
        
        # Create actor-critic
        self.actor_critic = ActorCritic(model).to(device)
        
        # Optimizer
        lr = rl_config.get('lr', 3e-4)
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=lr)
        
        # PPO parameters
        self.clip_epsilon = 0.2
        self.ppo_epochs = 4
        
        # Memory
        self.memory = PPOMemory()
        
        # Tracking
        self.episode_rewards = []
        self.best_reward = float('-inf')
    
    def select_action(
        self,
        state: torch.Tensor
    ) -> Tuple[int, float, float]:
        """
        Select action using policy
        
        Returns:
            action, log_prob, value
        """
        with torch.no_grad():
            action_probs, value = self.actor_critic(state.unsqueeze(0))
            
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item(), value.item()
    
    def compute_returns(
        self,
        rewards: List[float],
        dones: List[bool],
        values: List[float]
    ) -> List[float]:
        """Compute discounted returns"""
        returns = []
        R = 0
        
        for reward, done, value in zip(
            reversed(rewards),
            reversed(dones),
            reversed(values)
        ):
            if done:
                R = 0
            R = reward + self.gamma * R
            returns.insert(0, R)
        
        return returns
    
    def ppo_update(self):
        """Update policy using PPO"""
        states, actions, rewards, values, old_log_probs, dones = self.memory.get()
        
        # Compute returns
        returns = self.compute_returns(rewards, dones, values)
        
        # Convert to tensors
        states = torch.stack(states).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        old_log_probs = torch.tensor(old_log_probs).to(self.device)
        returns = torch.tensor(returns).to(self.device)
        values = torch.tensor(values).to(self.device)
        
        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Advantages
        advantages = returns - values
        
        # PPO update
        for _ in range(self.ppo_epochs):
            # Forward pass
            action_probs, new_values = self.actor_critic(states)
            
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy()
            
            # Ratio
            ratio = (new_log_probs - old_log_probs).exp()
            
            # Clipped surrogate
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            
            # Loss
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(new_values.squeeze(), returns)
            entropy_loss = -entropy.mean()
            
            loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss
            
            # Update
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
            self.optimizer.step()
    
    def train(
        self,
        train_data: np.ndarray,
        checkpoint_dir: str = 'model'
    ):
        """Full RL training loop"""
        print("\n" + "="*60)
        print("🎮 PHASE 3: REINFORCEMENT LEARNING (Optional)")
        print("="*60)
        print(f"Episodes: {self.episodes}")
        print(f"Gamma: {self.gamma}")
        print(f"Reward penalties - α: {self.reward_alpha}, β: {self.reward_beta}")
        print("="*60 + "\n")
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Create environment
        env = TradingEnvironment(train_data)
        
        for episode in range(self.episodes):
            state = env.reset()
            state_tensor = torch.FloatTensor(state).to(self.device)
            
            episode_reward = 0
            done = False
            
            while not done:
                # Select action
                action, log_prob, value = self.select_action(state_tensor)
                
                # Take step
                next_state, reward, done, info = env.step(action)
                
                # Apply reward shaping
                shaped_reward = reward
                shaped_reward -= self.reward_alpha * (info['num_trades'] / 100)
                shaped_reward -= self.reward_beta * info['drawdown']
                
                # Store in memory
                self.memory.store(
                    state_tensor,
                    action,
                    shaped_reward,
                    value,
                    log_prob,
                    done
                )
                
                episode_reward += reward
                state_tensor = torch.FloatTensor(next_state).to(self.device)
            
            # Update policy
            self.ppo_update()
            self.memory.clear()
            
            # Track rewards
            self.episode_rewards.append(episode_reward)
            
            # Print progress
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                print(f"Episode {episode+1}/{self.episodes} | "
                      f"Avg Reward: {avg_reward:.2f} | "
                      f"Profit: {info['profit']:.2f} | "
                      f"Trades: {info['num_trades']}")
                
                # Save best model
                if avg_reward > self.best_reward:
                    self.best_reward = avg_reward
                    checkpoint_path = f"{checkpoint_dir}/natron_v2_rl_best.pt"
                    torch.save({
                        'episode': episode,
                        'model_state_dict': self.actor_critic.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'best_reward': self.best_reward,
                        'config': self.config
                    }, checkpoint_path)
        
        print("\n" + "="*60)
        print("✅ RL TRAINING COMPLETE")
        print(f"Best average reward: {self.best_reward:.2f}")
        print("="*60 + "\n")


# Note: RL training is optional and experimental
# For production use, supervised learning is recommended
