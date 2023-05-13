import json
import os

from .generate_datasets import get_dataset
from src.utils.argparsing import get_args
from src.utils.utils import log, name_dataset
from src.constants import FEEDBACK_DIR
from src.dataset.feedback import (
    DirectionFeedback,
    DistanceFeedback,
    ActionFeedback,
    AdjacencyFeedback,
)


def _feedback_contains_config(feedback, args):
    if args["feedback_type"] in feedback:
        if args["feedback_mode"] in feedback[args["feedback_type"]]:
            if (
                f"{args['feedback_freq_type']}_{args['feedback_freq_steps']}"
                in feedback[args["feedback_type"]][args["feedback_mode"]]
            ):
                return True
    return False


def _get_feedback_with_config(feedback, args):
    return feedback[args["feedback_type"]][args["feedback_mode"]][
        f"{args['feedback_freq_type']}_{args['feedback_freq_steps']}"
    ]


def get_feedback(args, dataset):
    generator = {
        "direction": DirectionFeedback,
        "distance": DistanceFeedback,
        "action": ActionFeedback,
        "adjacency": AdjacencyFeedback,
    }[args["feedback_type"]](args, dataset)

    fn = os.path.join(FEEDBACK_DIR, f"{name_dataset(args)}.json")
    should_generate = True

    # if we don't already have feedback for this dataset, generate it
    if os.path.exists(fn):
        log("found existing feedback file for this dataset, loading...")
        with open(fn) as f:
            feedback = json.load(f)
            should_generate = not _feedback_contains_config(feedback, args)

    if should_generate:
        log("generating feedback...")
        feedback = generator.generate_feedback()
        generator.save_feedback()

    return _get_feedback_with_config(feedback, args)


if __name__ == "__main__":
    args = get_args()
    dataset = get_dataset(args)
    feedback = get_feedback(args, dataset)
