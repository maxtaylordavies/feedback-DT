# feedback-DT

## Creating a BabyAI dataset
Currently, this can be run with three different parameters. The plan is to also add an RGB parameter.
```src/_datsets.py --env_name --num_episodes --include_timeout```

## Passing different environment name strings
The names of BabyAI environments can be found on this page https://minigrid.farama.org/environments/babyai/
under "Registered Configurations". 
A list of environments for the single-room ```goto``` tasks (to be used for prelimary experiments as per project proposal) can be found in ```envs.json```.

### Passing different numbers of episodes
Note that when passing any ```--num_episodes``` with ```--include_timeout=False```, this will likely result in significantly fewer episodes than passed (due to only successful episodes being registered).

### Recording episodes that timed out / were truncated
By default, the ```--include_timeout``` parameter will be parsed with ```True```, resulting in episodes that ended because they timed out (reached ```max_steps```) before being terminated by the agent. To only record successful episodes, pass ```False```

### Example
```sh
python src/get_datasets.py --env_name "BabyAI-GoToRedBallGrey-v0" --num_episodes 1000 --include_timeout False
```

## Generating language feedback for an existing BabyAI dataset

### Specifying the dataset
Rather than providing the name of the dataset, simply provide the same argument values as when you created the dataset (or in case the dataset doesn't exist yet, it will be created now), e.g. ``--env-name "BabyAI-GoToRedBallGrey-v0" --num_episodes 1000 --include_timeout False``.

### Specifying the type of feedback
Possible feedback types for ablations include "direction", "distance", "adjacency" and "action". So far, only direction feedback has been implemented fully.

### Specifying the mode of the feedback
This refers to whether there are multiple, expressive variants (generated with the help of ChatGPT) to choose from ("verbose"). The 'simple' mode retrieves a simple base variant.

### Specifying the frequency of providing feedback
You can specify after how many steps to provide feedback. Use a sensible number based on the ``max_steps``, which for most environments is 64 - except ``BabyAI-GoToObjS4-v0`` and ``BabyAI-GoToObjS6-v0``, where ``max_steps`` is 16 - we want to provide feedback at least once, ideally significantly more often. Additionally, you can specify whether to give feedback exactly every ``feedback_freq_steps`` or average (using a poisson distribution) every ``feedback_freq_steps``. Note that both for 'exact' and 'poisson', a ``feedback_freq_steps`` of at least 2 is enforced (so that we're never providing feedback at every step).

### Example

```sh
python src/get_feedback.py --feedback_type "direction" --feedback_mode "simple" --feedback_freq_steps 3 --feedback_freq_type "exact" --env_name "BabyAI-GoToRedBallGrey-v0" --num_episodes 10 --include_timeout False
```

### Where to find feedback
Feedback is store in json files in the folder ``feedback_data``. You will find all feedback variations for a given dataset in the same json file, organised hierarchically by the feedback hyperparameters (type > mode > frequency). You can retrieved it using the appropriate combination of keys.
