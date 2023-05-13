from itertools import product

mission_actions = [
    "go to the {color} {type} {location}",
    "pick up a/the {color} {type} {location}",
    "open the {color} door {location}",
    "put the {color} {type} {location} next to the {color} {type} {location}",
]
sequence_concatenators = [", then ", " after you "]
mission_space = [
    m[0] + " and " + m[1] for m in product(mission_actions, mission_actions)
]
sequence_mission_space = [
    s[0][0] + s[0][1] + s[1]
    for s in product(
        product(mission_space + mission_actions, sequence_concatenators),
        mission_space + mission_actions,
    )
]

print(sequence_mission_space)

# # with action resulting in current state

# pi = args["policy"] or env.action_space.sample

# total_steps = 0
# for episode in args["num_episodes"]:
#     episode_step, terminated, truncated = 0, False, False
#     env.reset(seed=args["seed"])

#     while not (terminated or truncated):
#         action = pi()
#         observation, reward, terminated, truncated, _ = env.step(action)

#         replay_buffer["episode"][total_steps] = np.array(episode)
#         replay_buffer["action"][total_steps] = np.array(action)
#         replay_buffer["reward"][total_steps] = np.array(reward)
#         replay_buffer["terminated"][total_steps] = np.array(terminated)
#         replay_buffer["truncated"][total_steps] = np.array(truncated)

#         episode_step, total_steps = episode_step + 1, total_steps + 1

# env.close()

# # with action taken on current state

# pi = args["policy"] or env.action_space.sample

# total_steps = 0
# for episode in args["num_episodes"]:
#     terminated, truncated = False, False
#     observation, reward, _ = env.reset(seed=args["seed"])

#     while not (terminated or truncated):
#         replay_buffer["episode"][total_steps] = np.array(episode)
#         replay_buffer["reward"][total_steps] = np.array(reward)
#         replay_buffer["observation"][total_steps] = np.array(observation)

#         action = pi(observation)
#         replay_buffer["action"][total_steps] = np.array(action)

#         observation, reward, terminated, truncated, _ = env.step(action)

#         total_steps += 1

# env.close()
