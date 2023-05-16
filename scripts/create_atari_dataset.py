import os

from dopamine.replay_memory import circular_replay_buffer
import numpy as np

from src.dataset import CustomDataset


def load_buffer(data_dir, game, buffer_idx):
    replay_buffer = circular_replay_buffer.OutOfGraphReplayBuffer(
        observation_shape=(84, 84),
        stack_size=4,
        update_horizon=1,
        gamma=0.99,
        observation_dtype=np.uint8,
        batch_size=32,
        replay_capacity=100000,
    )
    replay_buffer.load(os.path.join(data_dir, game, "1", "replay_logs"), buffer_idx)
    return replay_buffer


def create_dataset(data_dir, game, num_samples):
    obs, acts, rewards, dones = [], [], [], []

    buffer_idx, depleted = -1, True
    while len(obs) < num_samples:
        if depleted:
            buffer_idx, depleted = buffer_idx + 1, False
            buffer, i = load_buffer(data_dir, game, buffer_idx), 0

        (
            s,
            a,
            r,
            _,
            _,
            _,
            terminal,
            _,
        ) = buffer.sample_transition_batch(batch_size=1, indices=[i])

        obs.append(s[0])
        acts.append(a[0])
        rewards.append(r[0])
        dones.append(terminal[0])

        i += 1
        if i == buffer._replay_capacity:
            depleted = True

    return CustomDataset(
        level_group="",
        level_name="",
        missions=np.array([]),
        direction_observations=np.array([]),
        agent_positions=np.array([]),
        oracle_views=np.array([]),
        dataset_name="",
        algorithm_name="",
        environment_name="",
        environment_stack="",
        seed_used=0,
        code_permalink="",
        author="",
        author_email="",
        observations=obs,
        actions=acts,
        rewards=rewards,
        terminations=dones,
        truncations=np.zeros_like(dones),
        episode_terminals=None,
        discrete_action=True,
    )


def main():
    data_dir = "/home/s2227283/projects/feedback-DT/data/dqn_replay"
    game = "Breakout"
    num_samples = 1000

    dataset = create_dataset(data_dir, game, num_samples)

    print(dataset.observations.shape)
    print(dataset.actions.shape)
    print(dataset.rewards.shape)
    print(dataset.terminations.shape)
    print(dataset.truncations.shape)


if __name__ == "__main__":
    main()
