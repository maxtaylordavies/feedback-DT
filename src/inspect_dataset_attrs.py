from get_datasets import load_dataset, list_local_datasets

print(list_local_datasets())
dataset = load_dataset("BabyAI-GoToRedBallGrey-v0_10-eps_incl-timeout")
print("\n")
print("Dataset attributes")
print("*"*10)
attrs = dataset.__dir__()
for a in attrs:
    print(a)

print("\n")
print("Episode attributes")
print("*"*10)
attrs = dataset.episodes[0].__dir__()
for a in attrs:
    print(a)
    
print("\n")
print("Examples (dataset) attributes")
print("*"*10)

dataset_attr_dict = {"goal_positions": dataset.goal_positions, "agent_positions": dataset.agent_positions, "direction_observations": dataset.direction_observations, "observations": dataset.observations, "actions": dataset.actions, "rewards": dataset.rewards, "terminations": dataset.terminations, "truncations": dataset.truncations, "episode_terminals": dataset.episode_terminals, "episodes": dataset.episodes}
for name, attr in dataset_attr_dict.items():
    try:
        print(f"Shape of {name} (from dataset): {attr.shape}\n")
    except:
        print(f"Length of {name} (from dataset): {len(attr)}\n")

print(dataset.goal_positions)

print("\n")
print("Examples (episode) attributes")
print("*"*10)
print("Episode length")
for e in dataset.episodes:
    print(len(e))
print()
episodes_attr_dict = {"observation_shape": dataset.episodes[0].observation_shape, "action_size": dataset.episodes[0].action_size, "observations": dataset.episodes[0].observations, "actions": dataset.episodes[0].actions, "rewards": dataset.episodes[0].rewards, "termination": dataset.episodes[0].termination, "truncation": dataset.episodes[0].truncation, "transitions": dataset.episodes[0].transitions}
for name, attr in episodes_attr_dict.items():
    try:
        print(f"Shape of {name} (from episode): {attr.shape}\n")
    except:
        try:
            print(f"Length of {name} (from episode): {len(attr)}\n")    
        except:
            print(f"{name} (from episode): {attr}\n")   

