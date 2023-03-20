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
python src/_datasets.py --env_name "BabyAI-GoToRedBallGrey-v0" --num_episodes 1000 --include_timeout False
```

## Generating language feedback for an existing BabyAI dataset

### Specifying the dataset
Rather than providing the name of the dataset, simply provide the same argument values as when you created the dataset (or in case the dataset doesn't exist yet, it will be created now), e.g. ``--env-name "BabyAI-GoToRedBallGrey-v0" --num_episodes 1000 --include_timeout False --seed 42``.

### Specifying the type of feedback
For ablations, the feedback ``--type`` can be set to ``direction``, ``distance``, ``adjacency`` or ``action``.

### Specifying the mode of the feedback
This refers to whether feedback should be chosen at random from multiple, expressive variants (generated with the help of ChatGPT), in whcih cae ``verbose`` should be passed for ``--mode``. The ``simple`` mode instead retrieves a single simple base variant.

### Specifying the frequency of providing feedback
You can specify after how many steps to provide feedback (``--feedback_freq_steps``). Use a sensible number based on the ``max_steps``, which for most environments is 64 - except ``BabyAI-GoToObjS4-v0`` and ``BabyAI-GoToObjS6-v0``, where ``max_steps`` is 16. For training, feedback can be provided as often as every step, for testing, it should not be provided at every step.

Additionally, you can specify whether to give feedback exactly every ``feedback_freq_steps`` for all episodes in the run (``exact``), or on average (using a poisson distribution) every ``feedback_freq_steps`` across the episodes, with this being constant on a per-episode basis (``poisson``).

### Example

```sh
python src/_feedback.py --feedback_type "direction" --feedback_mode "simple" --feedback_freq_steps 1 --feedback_freq_type "exact" --env_name "BabyAI-GoToRedBallGrey-v0" --num_episodes 10 --include_timeout False --seed 42
```

If the dataset specified doesn't exist yet locally, it will be generated when the ``_feedback.py`` script is run.

### Where to find feedback
Feedback is store in json files in the folder ``feedback_data``. You will find all feedback variations for a given dataset in the same json file, organised hierarchically by the feedback hyperparameters (type > mode > frequency). You can retrieved it using the appropriate combination of keys.
