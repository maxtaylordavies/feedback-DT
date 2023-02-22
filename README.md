# feedback-DT

## Creating a BabyAI dataset
Currently, this can be run with three different parameters. The plan is to also add an RGB parameter.
```src/datsets.py --env-name --num-episodes --include-timeout```

## Passing different environment name strings
The names of BabyAI environments can be found on this page https://minigrid.farama.org/environments/babyai/
under "Registered Configurations". 
A list of environments for the single-room ```goto``` tasks (to be used for prelimary experiments as per project proposal) can be found in ```envs.json```.

### Passing different numbers of episodes
Note that when passing any ```--num-episodes``` with ```--include-timeout=False```, this will likely result in significantly fewer episodes than passed (due to only successful episodes being registered).

### Recording episodes that timed out / were truncated
By default, the ```--include-timeout``` parameter will be parsed with ```True```, resulting in episodes that ended because they timed out (reached ```max_steps```) before being terminated by the agent. To only record successful episodes, pass ```False```

### Example
```src/datsets.py --env-name "BabyAI-GoToRedBallGrey-v0" --num-episodes 1000 --include-timeout False```