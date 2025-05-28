import torch
import torch.optim as optim
from torch.distributions import Categorical
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import numpy as np
import os

from Gnn import GNN
from PowerGridEnv import PowerGridEnv


def obs_to_data(obs, device=torch.device("cpu")):
    # Convert environment observation to PyG Data object
    x = torch.tensor(obs["bus_features"], dtype=torch.float32, device=device)
    edge_index = torch.tensor(obs["topology"].T, dtype=torch.long, device=device)
    edge_attr = torch.tensor(obs["line_features"], dtype=torch.float32, device=device)
    # Single graph, so batch is all zeros
    batch = torch.zeros(x.size(0), dtype=torch.long, device=device)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)


def plot_learning_curves(steps, rewards, losses=None, entropies=None):
    """Plot and save learning curves."""
    os.makedirs("plots", exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    
    # Plot rewards
    plt.subplot(2, 1, 1)
    plt.plot(steps, rewards, 'b-', label='Average Reward (10 episodes)')
    plt.xlabel('Training Steps')
    plt.ylabel('Average Reward')
    plt.title('Learning Curve - Rewards')
    plt.grid(True)
    plt.legend()
    
    # Plot loss and entropy if available
    if losses and entropies:
        # Create x-axis for update steps
        update_steps = np.linspace(0, steps[-1], len(losses)) if steps else []
        
        plt.subplot(2, 1, 2)
        plt.plot(update_steps, losses, 'r-', label='Loss')
        plt.plot(update_steps, entropies, 'g-', label='Entropy')
        plt.xlabel('Training Steps')
        plt.ylabel('Value')
        plt.title('Loss and Entropy')
        plt.grid(True)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"plots/learning_curve_{len(steps)}.png")
    plt.close()

    

class PPOAgent:
    def __init__(
        self,
        model,
        lr=3e-4,
        gamma=0.99,
        lam=0.95,
        clip_eps=0.2,
        c1=0.5,
        c2=0.05,
        device=None,
    ):
        self.model = model
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.c1 = c1
        self.c2 = c2
        self.device = device or torch.device("cpu")
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.model.to(self.device)

    def select_action(self, obs):
        data = obs_to_data(obs, device=self.device)
        with torch.no_grad():  # No need to track gradients during inference
            logits, value = self.model(data)

            mask = torch.tensor(
                obs["action_mask"], dtype=torch.bool, device=self.device
            )
            # Mask invalid actions
            masked_logits = logits.masked_fill(~mask, -1e8)
            dist = Categorical(logits=masked_logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

        return action.item(), log_prob, entropy, value

    def compute_gae(self, rewards, values, dones, last_value):
        if torch.is_tensor(values):
            values = values.squeeze().tolist()
        if torch.is_tensor(last_value):
            last_value = last_value.item()

        returns = []
        gae = 0
        next_val = last_value

        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * next_val * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.lam * (1 - dones[i]) * gae
            ret = gae + values[i]
            returns.insert(0, ret)
            next_val = values[i]
        return torch.tensor(returns, dtype=torch.float32, device=self.device)

    def update(self, trajectories, epochs=4, batch_size=32):
        # trajectories: list of dicts with keys: obs, action, logp, value, reward, done
        obs_list = [t["obs"] for t in trajectories]
        actions = torch.tensor(
            [t["action"] for t in trajectories], dtype=torch.long, device=self.device
        )
        old_logps = torch.stack([t["logp"] for t in trajectories]).detach()
        values = torch.stack([t["value"] for t in trajectories]).detach()
        rewards = [t["reward"] for t in trajectories]
        dones = [t["done"] for t in trajectories]

        # Compute last value for bootstrapping
        with torch.no_grad():
            last_data = obs_to_data(trajectories[-1]["obs"], device=self.device)
            _, last_value = self.model(last_data)
            last_value = last_value.detach()

        # Compute returns using GAE
        returns = self.compute_gae(rewards, values, dones, last_value)
        advantages = returns - values.squeeze(-1)  # Ensure dimensions match

        data_list = [obs_to_data(o, device=self.device) for o in obs_list]

        mask_list = [
            torch.tensor(o["action_mask"], dtype=torch.bool, device=self.device)
            for o in obs_list
        ]

        dataset = list(
            zip(data_list, actions, old_logps, returns, advantages, mask_list)
        )

        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        update_count = 0

        for _ in range(epochs):
            indices = torch.randperm(len(dataset))

            for start_idx in range(0, len(dataset), batch_size):
                mb_indices = indices[start_idx : start_idx + batch_size]
                mb_data = [data_list[i] for i in mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_logps = old_logps[mb_indices]
                mb_returns = returns[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_masks = [mask_list[i] for i in mb_indices]

                logits_list = []
                values_pred = []
                log_probs = []
                entropies = []

                for i, data in enumerate(mb_data):
                    logits, value = self.model(data)

                    masked_logits = logits.masked_fill(~mb_masks[i], -1e8)
                    dist = Categorical(logits=masked_logits)

                    log_prob = dist.log_prob(mb_actions[i])
                    entropy = dist.entropy()

                    logits_list.append(logits)
                    values_pred.append(value.squeeze(0))
                    log_probs.append(log_prob)
                    entropies.append(entropy)

                # Stack tensors
                log_probs = torch.stack(log_probs)
                values_pred = torch.stack(values_pred)
                entropies = torch.stack(entropies)

                # Compute PPO policy loss
                ratio = torch.exp(log_probs - mb_old_logps)
                surr1 = ratio * mb_advantages
                surr2 = (
                    torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                    * mb_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = self.c1 * ((mb_returns - values_pred) ** 2).mean()
                entropy_bonus = -self.c2 * entropies.mean()

                loss = policy_loss + value_loss + entropy_bonus

                # Update model
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Update metrics
                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropies.mean().item()
                update_count += 1

        return {
            "loss": total_loss / update_count,
            "policy_loss": total_policy_loss / update_count,
            "value_loss": total_value_loss / update_count,
            "entropy": total_entropy / update_count,
        }


if __name__ == "__main__":
    device = torch.device("cuda")
    env = PowerGridEnv(k=5)

    model = GNN(node_feat_dim=4, edge_feat_dim=5, hidden_dim=64, n_layers=3)
    agent = PPOAgent(model, lr=1e-3, gamma=0.99, lam=0.95, clip_eps=0.2)

    max_steps = 30000
    update_every = 256
    obs, _ = env.reset()
    trajectories = []
    total_steps = 0
    best_reward = float("-inf")
    episode_rewards = []
    current_episode_reward = 0
    
    # Data for plotting
    training_steps = []
    avg_rewards = []
    step_losses = []
    step_entropies = []

    print(f"Starting training on device: {device}")

    while total_steps < max_steps:
        # Collect rollout
        steps_this_batch = 0
        while steps_this_batch < update_every:
            action, logp, entropy, value = agent.select_action(obs)
            next_obs, reward, done, _, info = env.step(action)

            # Track episode rewards
            current_episode_reward += reward

            trajectories.append(
                {
                    "obs": obs,
                    "action": action,
                    "logp": logp,
                    "value": value,
                    "reward": reward,
                    "done": done,
                }
            )

            obs = next_obs
            total_steps += 1
            steps_this_batch += 1

            if done:
                # Track episode metrics
                episode_rewards.append(current_episode_reward)
                if current_episode_reward >= best_reward:
                    best_reward = current_episode_reward

                # Store data for plotting
                training_steps.append(total_steps)
                if len(episode_rewards) >= 10:
                    avg_rewards.append(sum(episode_rewards[-10:]) / 10)
                else:
                    avg_rewards.append(sum(episode_rewards) / len(episode_rewards))
                
                # Reset for new episode
                obs, _ = env.reset()
                current_episode_reward = 0


        # Update policy
        if trajectories:  # Make sure we collected some data
            metrics = agent.update(trajectories)
            trajectories = []
            
            # Store metrics for plotting
            step_losses.append(metrics['loss'])
            step_entropies.append(metrics['entropy'])
            

    # Save final model
    torch.save(model.state_dict(), "final_power_grid_model.pt")
    
    # Final plot
    plot_learning_curves(training_steps, avg_rewards, step_losses, step_entropies)


