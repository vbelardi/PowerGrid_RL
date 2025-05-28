import torch
import numpy as np
from collections import defaultdict

from PowerGridEnv import PowerGridEnv
from Gnn import GNN
from Train import PPOAgent, obs_to_data


def evaluate(agent, env, episodes=100, deterministic=True, render=False):
    rewards = []
    steps_per_episode = []
    collapse_count = 0
    violation_stats = defaultdict(list)

    # Track removed lines
    all_removed_lines = []
    lines_per_episode = []

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        # Track lines removed in this episode
        episode_lines = []

        while not done:
            # Get action based on policy
            if deterministic:
                # For deterministic evaluation, use the highest probability action
                data = obs_to_data(
                    obs, device=agent.device
                )  # Use the standalone function
                with torch.no_grad():
                    logits, _ = agent.model(data)
                    mask = torch.tensor(
                        obs["action_mask"], dtype=torch.bool, device=agent.device
                    )
                    masked_logits = logits.masked_fill(~mask, -1e8)
                    action = torch.argmax(masked_logits).item()
            else:
                # Use the stochastic policy from the agent
                action, _, _, _ = agent.select_action(obs)

            # Get the actual line ID that was removed
            line_id = env.all_line_ids[action]
            episode_lines.append(line_id)
            all_removed_lines.append(line_id)

            # Take action in environment
            next_obs, reward, done, truncated, info = env.step(action)
            done = done or truncated

            # Track stats
            total_reward += reward
            steps += 1

            # Track voltage and loading violations
            if not info.get("collapsed", False):
                violation_stats["bus_violation"].append(info["bus_violation_ratio"])
                violation_stats["line_violation"].append(info["line_violation_ratio"])

            if render:
                env.render()

            obs = next_obs

        # Episode complete
        rewards.append(total_reward)
        steps_per_episode.append(steps)
        if info.get("collapsed", False):
            collapse_count += 1

        # Store lines removed in this episode
        lines_per_episode.append(episode_lines)

        # Progress update
        print(
            f"Episode {ep + 1}/{episodes} reward: {total_reward:.2f}, steps: {steps}, collapsed: {info.get('collapsed', False)}"
        )
        print(f"  Lines removed: {episode_lines}")

    # Calculate success rate (in this case, causing grid collapse)
    success_rate = collapse_count / episodes

    # Get unique removed lines
    unique_lines = sorted(list(set(all_removed_lines)))

    # Count frequency of each line being removed
    line_counts = defaultdict(int)
    for line_id in all_removed_lines:
        line_counts[line_id] += 1

    # Sort by frequency
    sorted_lines = sorted(line_counts.items(), key=lambda x: x[1], reverse=True)

    return {
        "rewards": rewards,
        "steps_per_episode": steps_per_episode,
        "success_rate": success_rate,
        "collapse_count": collapse_count,
        "mean_bus_violation": np.mean(violation_stats["bus_violation"])
        if violation_stats["bus_violation"]
        else 0,
        "mean_line_violation": np.mean(violation_stats["line_violation"])
        if violation_stats["line_violation"]
        else 0,
        "unique_lines": unique_lines,
        "line_frequencies": sorted_lines,
        "lines_per_episode": lines_per_episode,
    }


if __name__ == "__main__":
    # Set device
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Create environment and agent
    env = PowerGridEnv(k=5)
    print(env.case_name)
    model = GNN(node_feat_dim=4, edge_feat_dim=5, hidden_dim=64, n_layers=3)
    model.load_state_dict(torch.load("best_power_grid_model.pt", map_location=device))
    model.to(device)
    agent = PPOAgent(model=model, device=device)

    # Set model to evaluation mode
    agent.model.eval()

    # Run stochastic evaluation
    print("\n=== Stochastic Evaluation ===")
    num_episodes = 10
    stochastic_results = evaluate(
        agent, env, episodes=num_episodes, deterministic=False
    )

    num_episodes = 1
    print("\n=== Deterministic Evaluation ===")
    deterministic_results = evaluate(
        agent, env, episodes=num_episodes, deterministic=True
    )

    # Print summary statistics
    print("\n=== Evaluation Results ===")

    print("Stochastic Policy:")
    mean_reward = np.mean(stochastic_results["rewards"])
    std_reward = np.std(stochastic_results["rewards"])
    mean_steps = np.mean(stochastic_results["steps_per_episode"])
    print(f"- Mean Reward: {mean_reward:.2f}, Std: {std_reward:.2f}")
    print(f"- Mean Steps per Episode: {mean_steps:.2f}")
    print(f"- Collapse Success Rate: {stochastic_results['success_rate']:.2%}")
    print(f"- Mean Bus Violation Ratio: {stochastic_results['mean_bus_violation']:.4f}")
    print(
        f"- Mean Line Violation Ratio: {stochastic_results['mean_line_violation']:.4f}"
    )

    # Print removed lines for stochastic policy
    print("\n- Unique Lines Removed (Stochastic):")
    for line_id in stochastic_results["unique_lines"]:
        print(f"  Line {line_id}")

    print("\n- Line Removal Frequency (Stochastic):")
    for line_id, count in stochastic_results["line_frequencies"]:
        print(
            f"  Line {line_id}: {count} times ({count / sum(stochastic_results['steps_per_episode']) * 100:.2f}%)"
        )

    print("\nDeterministic Policy:")
    mean_reward = np.mean(deterministic_results["rewards"])
    std_reward = np.std(deterministic_results["rewards"])
    mean_steps = np.mean(deterministic_results["steps_per_episode"])
    print(f"- Mean Reward: {mean_reward:.2f}, Std: {std_reward:.2f}")
    print(f"- Mean Steps per Episode: {mean_steps:.2f}")
    print(f"- Collapse Success Rate: {deterministic_results['success_rate']:.2%}")
    print(
        f"- Mean Bus Violation Ratio: {deterministic_results['mean_bus_violation']:.4f}"
    )
    print(
        f"- Mean Line Violation Ratio: {deterministic_results['mean_line_violation']:.4f}"
    )

    # Print removed lines for deterministic policy
    print("\n- Unique Lines Removed (Deterministic):")
    for line_id in deterministic_results["unique_lines"]:
        print(f"  Line {line_id}")

    print("\n- Line Removal Frequency (Deterministic):")
    for line_id, count in deterministic_results["line_frequencies"]:
        print(
            f"  Line {line_id}: {count} times ({count / sum(deterministic_results['steps_per_episode']) * 100:.2f}%)"
        )

    # Save line removal data to files
    with open("stochastic_line_removals.txt", "w") as f:
        f.write("Unique lines removed by stochastic policy:\n")
        for line_id in stochastic_results["unique_lines"]:
            f.write(f"Line {line_id}\n")

        f.write("\nLine removal frequencies:\n")
        for line_id, count in stochastic_results["line_frequencies"]:
            f.write(
                f"Line {line_id}: {count} times ({count / sum(stochastic_results['steps_per_episode']) * 100:.2f}%)\n"
            )

    with open("deterministic_line_removals.txt", "w") as f:
        f.write("Unique lines removed by deterministic policy:\n")
        for line_id in deterministic_results["unique_lines"]:
            f.write(f"Line {line_id}\n")

        f.write("\nLine removal frequencies:\n")
        for line_id, count in deterministic_results["line_frequencies"]:
            f.write(
                f"Line {line_id}: {count} times ({count / sum(deterministic_results['steps_per_episode']) * 100:.2f}%)\n"
            )

    print(
        "\nLine removal data saved to stochastic_line_removals.txt and deterministic_line_removals.txt"
    )
