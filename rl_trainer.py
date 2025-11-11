"""
Natron Transformer - Phase 3: Reinforcement Learning
PPO-based trading agent for profit optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import os
from tqdm import tqdm
from typing import Dict, Tuple, List
import wandb

from model import NatronTransformer, create_model


class TradingEnvironment:
    """
    Trading environment for RL training
    Simulates trading based on model predictions
    """
    
    def __init__(self, features: np.ndarray, prices: np.ndarray, 
                 sequence_length: int = 96):
        """
        Args:
            features: (N, n_features) normalized features
            prices: (N,) close prices
            sequence_length: Length of observation window
        """
        self.features = features
        self.prices = prices
        self.sequence_length = sequence_length
        self.n_steps = len(features) - sequence_length
        
        # State
        self.current_step = 0
        self.position = 0  # -1: short, 0: flat, 1: long
        self.entry_price = 0.0
        self.portfolio_value = 10000.0  # Starting capital
        self.cash = 10000.0
        self.shares = 0
        
        # Tracking
        self.trades = []
        self.portfolio_history = [self.portfolio_value]
    
    def reset(self) -> np.ndarray:
        """Reset environment"""
        self.current_step = 0
        self.position = 0
        self.entry_price = 0.0
        self.portfolio_value = 10000.0
        self.cash = 10000.0
        self.shares = 0
        self.trades = []
        self.portfolio_history = [self.portfolio_value]
        
        return self._get_observation()
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation (sequence of features)"""
        idx = self.current_step
        return self.features[idx:idx + self.sequence_length].copy()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take action in environment
        
        Args:
            action: 0=hold, 1=buy, 2=sell
        
        Returns:
            observation, reward, done, info
        """
        current_price = self.prices[self.current_step + self.sequence_length]
        
        # Execute action
        reward = 0.0
        info = {'action': action, 'position': self.position}
        
        if action == 1:  # Buy
            if self.position <= 0:  # Close short or go long
                if self.position == -1:  # Close short
                    profit = (self.entry_price - current_price) * abs(self.shares)
                    self.cash += profit
                    reward += profit / self.portfolio_value
                
                # Go long
                self.shares = self.cash / current_price
                self.cash = 0
                self.position = 1
                self.entry_price = current_price
                self.trades.append(('BUY', current_price, self.current_step))
        
        elif action == 2:  # Sell
            if self.position >= 0:  # Close long or go short
                if self.position == 1:  # Close long
                    profit = (current_price - self.entry_price) * self.shares
                    self.cash += self.shares * current_price
                    self.shares = 0
                    reward += profit / self.portfolio_value
                
                # Go short
                self.shares = -self.cash / current_price
                self.position = -1
                self.entry_price = current_price
                self.trades.append(('SELL', current_price, self.current_step))
        
        # Calculate portfolio value
        if self.position == 1:  # Long
            self.portfolio_value = self.shares * current_price
        elif self.position == -1:  # Short
            self.portfolio_value = self.cash + (self.entry_price - current_price) * abs(self.shares)
        else:  # Flat
            self.portfolio_value = self.cash
        
        self.portfolio_history.append(self.portfolio_value)
        
        # Penalties
        turnover_penalty = 0.001 if action != 0 else 0  # Transaction cost
        reward -= turnover_penalty
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= self.n_steps - 1
        
        if done:
            # Close any open position
            if self.position != 0:
                if self.position == 1:
                    self.cash = self.shares * current_price
                    self.shares = 0
                elif self.position == -1:
                    profit = (self.entry_price - current_price) * abs(self.shares)
                    self.cash += profit
                self.position = 0
            
            self.portfolio_value = self.cash
            self.portfolio_history.append(self.portfolio_value)
            
            # Calculate final reward metrics
            total_return = (self.portfolio_value - 10000.0) / 10000.0
            info['total_return'] = total_return
            info['final_value'] = self.portfolio_value
            info['n_trades'] = len(self.trades)
            
            # Sharpe ratio (simplified)
            if len(self.portfolio_history) > 1:
                returns = np.diff(self.portfolio_history) / np.array(self.portfolio_history[:-1])
                sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
                info['sharpe'] = sharpe
            
            # Max drawdown
            cummax = np.maximum.accumulate(self.portfolio_history)
            drawdown = (np.array(self.portfolio_history) - cummax) / cummax
            max_dd = np.min(drawdown)
            info['max_drawdown'] = max_dd
            
            # Final reward incorporates return and risk
            reward = total_return - 0.5 * abs(max_dd)
        
        observation = self._get_observation() if not done else np.zeros_like(self._get_observation())
        
        return observation, reward, done, info


class PPOAgent:
    """PPO agent for trading"""
    
    def __init__(self, model: NatronTransformer, config, device: str = 'cuda'):
        self.model = model
        self.config = config
        self.device = device
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(config.model.d_model, 128),
            nn.Tanh(),
            nn.Linear(128, 3)  # 3 actions: hold, buy, sell
        ).to(device)
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(config.model.d_model, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        ).to(device)
        
        # Optimizer
        params = list(self.actor.parameters()) + list(self.critic.parameters())
        self.optimizer = torch.optim.Adam(params, lr=config.rl.learning_rate)
        
        # Hyperparameters
        self.gamma = config.rl.gamma
        self.gae_lambda = config.rl.gae_lambda
        self.clip_epsilon = config.rl.clip_epsilon
        self.entropy_coef = config.rl.entropy_coef
        self.value_loss_coef = config.rl.value_loss_coef
    
    def get_action_and_value(self, observation: torch.Tensor) -> Tuple[int, float, float]:
        """
        Get action from policy and value estimate
        
        Args:
            observation: (seq_len, n_features)
        
        Returns:
            action, log_prob, value
        """
        with torch.no_grad():
            obs = observation.unsqueeze(0).to(self.device)  # (1, seq_len, n_features)
            
            # Get embeddings from model
            outputs = self.model(obs, return_embeddings=True)
            embeddings = outputs['embeddings']  # (1, d_model)
            
            # Get action probabilities
            logits = self.actor(embeddings)  # (1, 3)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            # Get value
            value = self.critic(embeddings)
            
            return action.item(), log_prob.item(), value.item()
    
    def compute_gae(self, rewards: List[float], values: List[float], 
                    dones: List[bool]) -> Tuple[List[float], List[float]]:
        """Compute Generalized Advantage Estimation"""
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
        
        return advantages, returns
    
    def update(self, observations: torch.Tensor, actions: torch.Tensor,
               old_log_probs: torch.Tensor, returns: torch.Tensor,
               advantages: torch.Tensor) -> Dict[str, float]:
        """PPO update"""
        # Get current policy and value
        obs = observations.to(self.device)
        
        # Forward pass
        outputs = self.model(obs, return_embeddings=True)
        embeddings = outputs['embeddings']
        
        # Actor loss
        logits = self.actor(embeddings)
        dist = Categorical(logits=logits)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        
        # Critic loss
        values = self.critic(embeddings).squeeze()
        critic_loss = F.mse_loss(values, returns)
        
        # Total loss
        loss = actor_loss + self.value_loss_coef * critic_loss - self.entropy_coef * entropy
        
        # Update
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.actor.parameters()) + list(self.critic.parameters()), 
            0.5
        )
        self.optimizer.step()
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy': entropy.item(),
            'total_loss': loss.item()
        }


def train_rl(config, model_path: str):
    """
    Run Phase 3 RL training
    
    Args:
        config: Configuration
        model_path: Path to pretrained supervised model
    """
    print("\n🚀 Starting Phase 3: Reinforcement Learning")
    
    if not config.rl.enabled:
        print("⚠️  RL training is disabled in config")
        return
    
    # Load supervised model
    print(f"📂 Loading supervised model from {model_path}")
    model = create_model(config)
    checkpoint = torch.load(model_path, map_location=config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Freeze encoder
    
    # Freeze encoder weights
    for param in model.parameters():
        param.requires_grad = False
    
    # Load data
    from feature_engine import load_and_prepare_data
    from dataset import load_scaler
    
    raw_df, features_df = load_and_prepare_data(config.data.csv_path)
    
    # Load scaler and normalize
    scaler_path = os.path.join(config.output_dir, 'scaler.pkl')
    scaler = load_scaler(scaler_path)
    features_normalized = scaler.transform(features_df.values)
    
    # Create environment
    prices = raw_df['close'].values
    env = TradingEnvironment(features_normalized, prices, config.data.sequence_length)
    
    # Create PPO agent
    agent = PPOAgent(model, config, config.device)
    
    print(f"   Episodes: {config.rl.episodes}")
    print(f"   Algorithm: {config.rl.algorithm.upper()}")
    
    # Initialize wandb
    if config.wandb_enabled:
        wandb.init(
            project=config.wandb_project,
            name='rl_training',
            config=config.__dict__
        )
    
    # Training loop
    os.makedirs(config.rl.checkpoint_dir, exist_ok=True)
    best_return = -float('inf')
    
    for episode in range(1, config.rl.episodes + 1):
        observation = env.reset()
        
        observations = []
        actions = []
        log_probs = []
        values = []
        rewards = []
        dones = []
        
        # Collect episode
        done = False
        step = 0
        while not done and step < config.rl.steps_per_episode:
            obs_tensor = torch.from_numpy(observation).float()
            action, log_prob, value = agent.get_action_and_value(obs_tensor)
            
            next_obs, reward, done, info = env.step(action)
            
            observations.append(obs_tensor)
            actions.append(action)
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            dones.append(done)
            
            observation = next_obs
            step += 1
        
        # Compute advantages and returns
        advantages, returns = agent.compute_gae(rewards, values, dones)
        
        # Convert to tensors
        observations_t = torch.stack(observations)
        actions_t = torch.tensor(actions, dtype=torch.long).to(config.device)
        log_probs_t = torch.tensor(log_probs, dtype=torch.float32).to(config.device)
        returns_t = torch.tensor(returns, dtype=torch.float32).to(config.device)
        advantages_t = torch.tensor(advantages, dtype=torch.float32).to(config.device)
        
        # Normalize advantages
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
        
        # Update policy
        update_metrics = agent.update(observations_t, actions_t, log_probs_t, 
                                       returns_t, advantages_t)
        
        # Log
        total_return = info.get('total_return', 0)
        if episode % 10 == 0:
            print(f"\n📍 Episode {episode}/{config.rl.episodes}")
            print(f"   Return: {total_return:.4f}")
            print(f"   Final Value: ${info.get('final_value', 0):.2f}")
            print(f"   Trades: {info.get('n_trades', 0)}")
            print(f"   Sharpe: {info.get('sharpe', 0):.2f}")
            print(f"   Max DD: {info.get('max_drawdown', 0):.2%}")
            print(f"   Actor Loss: {update_metrics['actor_loss']:.4f}")
        
        if config.wandb_enabled:
            wandb.log({
                'rl/return': total_return,
                'rl/final_value': info.get('final_value', 0),
                'rl/n_trades': info.get('n_trades', 0),
                'rl/sharpe': info.get('sharpe', 0),
                'rl/max_drawdown': info.get('max_drawdown', 0),
                'rl/actor_loss': update_metrics['actor_loss'],
                'rl/critic_loss': update_metrics['critic_loss'],
                'rl/entropy': update_metrics['entropy'],
                'rl/episode': episode
            })
        
        # Save best policy
        if total_return > best_return:
            best_return = total_return
            checkpoint = {
                'episode': episode,
                'actor_state_dict': agent.actor.state_dict(),
                'critic_state_dict': agent.critic.state_dict(),
                'return': total_return,
                'config': config
            }
            path = os.path.join(config.rl.checkpoint_dir, 'best_rl_policy.pt')
            torch.save(checkpoint, path)
            print(f"   💾 Saved best RL policy (return: {total_return:.4f})")
    
    print(f"\n✅ RL training completed!")
    print(f"   Best return: {best_return:.4f}")
    
    if config.wandb_enabled:
        wandb.finish()


if __name__ == "__main__":
    from config import load_config
    import sys
    
    config = load_config()
    
    if len(sys.argv) > 1:
        config.data.csv_path = sys.argv[1]
    
    if len(sys.argv) > 2:
        model_path = sys.argv[2]
    else:
        model_path = os.path.join(config.model_dir, 'natron_v2.pt')
    
    train_rl(config, model_path)
