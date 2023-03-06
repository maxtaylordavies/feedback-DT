# feedback-DT

## Creating a BabyAI dataset
Currently, this can be run with three different parameters. The plan is to also add an RGB parameter.
```src/get_datsets.py --env-name --num-episodes --include-timeout```

## Passing different environment name strings
The names of BabyAI environments can be found on this page https://minigrid.farama.org/environments/babyai/
under "Registered Configurations". 
A list of environments for the single-room ```goto``` tasks (to be used for prelimary experiments as per project proposal) can be found in ```envs.json```.

### Passing different numbers of episodes
Note that when passing any ```--num-episodes``` with ```--include-timeout=False```, this will likely result in significantly fewer episodes than passed (due to only successful episodes being registered).

### Recording episodes that timed out / were truncated
By default, the ```--include-timeout``` parameter will be parsed with ```True```, resulting in episodes that ended because they timed out (reached ```max_steps```) before being terminated by the agent. To only record successful episodes, pass ```False```

### Example
```sh
python src/get_datasets.py --env-name "BabyAI-GoToRedBallGrey-v0" --num-episodes 1000 --include-timeout False
```

## Generating language feedback for an existing BabyAI dataset

### Specifying the dataset
Rather than providing the name of the dataset, simply provide the same argument values as when you created the dataset, e.g. ``--env-name "BabyAI-GoToRedBallGrey-v0" --num-episodes 10 --include-timeout incl-timeout``. Feedback can only be generated for existing datasets, so you may have to create a dataset first.

### Specifying the type of feedback
Possible feedback types for ablations include "direction", "distance", "adjacency" and "action". So far, only direction feedback has been implemented.

### Specifying the mode of the feedback
This refers to whether there are multiple, expressive variants (generated with the help of ChatGPT) to choose from ("verbose"). The 'simple' mode retrieves a simple base variant.

### Specifying the frequency of providing feedback
You can specify after how many steps to provide feedback. Use a sensible number based on the ``max_steps``, which for most environments is 64 - except ``BabyAI-GoToObjS4-v0`` and ``BabyAI-GoToObjS6-v0``, where ``max_steps`` is 16 - we want to provide feedback at least once, ideally significantly more often. Additionally, you can specify whether to give feedback exactly every ``n-steps`` or average (using a poisson distribution) every ``n-steps``. Note that both for 'exact' and 'poisson', an ``n-steps`` of at least 2 is enforced (so that we're never providing feedback at every step).

### Example

```sh
python src/get_feedback.py --type "direction" --mode "simple" --n-steps 3 --freq-type "exact" --env-name "BabyAI-GoToRedBallGrey-v0" --num-episodes 10 --include-timeout incl-timeout
```

### Where to find feedback
Feedback is store in json files in the folder ``feedback_data``. You will find all feedback variations for a given dataset in the same json file, organised hierarchically by the feedback hyperparameters (type > mode > frequency). You can retrieved it using the appropriate combination of keys.
