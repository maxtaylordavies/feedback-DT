import gymnasium as gym
import json

# List of BabyAI environments used in experiments
envs = [
    "BabyAI-GoToRedBallGrey-v0", 
    "BabyAI-GoToRedBall-v0",
    "BabyAI-GoToRedBallNoDists-v0",
    "BabyAI-GoToRedBlueBall-v0",
    "BabyAI-GoToObj-v0",
    "BabyAI-GoToObjS4-v0",
    "BabyAI-GoToObjS6-v0"
]

envs_dict = dict()
for e in envs:
    env = gym.make(e)
    observation, _ = env.reset()
    envs_dict[e] = {"max_steps": env.max_steps, "mission_string": observation["mission"]}
    
with open("env_metadata.json", "w") as outfile:
    json.dump(envs_dict, outfile)

print("Success! Saved metadata for environments")
