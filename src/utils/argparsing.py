import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Decision transformer training")

    parser.add_argument(
        "--run_name",
        type=str,
        default="",
        help="name of the run (default: current date and time)",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=5 * 10**6,
        help="the number of episodes to collect for the environment",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="ppo",
        help="the policy to use for training; can be either 'ppo' or 'random'",
    )
    parser.add_argument(
        "--include_timeout",
        type=bool,
        default=True,
        help="whether to include episodes terminated by timeout / truncated episodes",
    )
    parser.add_argument(
        "--fully_obs",
        type=bool,
        default=False,
        help="whether to use fully-observed environment",
    )
    parser.add_argument(
        "--rgb_obs",
        type=bool,
        default=True,
        help="whether to use rgb oberservations of the environment",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="number of epochs to train",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="per-device batch size for training",
    )
    parser.add_argument(
        "--context_length",
        type=int,
        default=32,
        help="context length in timesteps",
    )
    parser.add_argument("--randomise_starts", type=bool, default=False)
    parser.add_argument(
        "--lr",
        type=float,
        default=5 * 1e-4,
        help="learning rate (default: 5 * 1e-4)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        help="Learning rate step gamma (default: 0.7)",
    )
    parser.add_argument(
        "--no_gpu", action="store_true", default=False, help="disables GPU training"
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=10,
        help="how many training steps between logging output (for PPO)",
    )
    parser.add_argument(
        "--record_video",
        type=bool,
        default=False,
        help="Whether to record videos of evaluation episodes",
    )
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default="disabled",
        help="wandb mode - can be online, offline, or disabled",
    )
    parser.add_argument(
        "--checkpoint", type=str, default="", help="path to pytorch checkpoint file"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="/disk/scratch/feedback-DT/output",
        help="Path to the directory to write output to",
    )
    parser.add_argument(
        "--del_all",
        type=bool,
        default=False,
        help="Whether to delete all local datasets",
    )
    parser.add_argument(
        "--use_feedback",
        type=bool,
        default=True,
        help="whether to use feedback during training",
    )
    parser.add_argument(
        "--use_mission",
        type=bool,
        default=True,
        help="whether to use mission during training",
    )
    parser.add_argument(
        "--demo",
        type=str,
        default="from_default_hard",
        help="the type of demo to make, either from a predefined action sequence corresponding to an easy or a hard task, or from an action sequence corresponding to an episode of the actual training data for a given environment and seed'",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="demos",
        help="the directory to save output - such as demo videos - to",
    )
    parser.add_argument(
        "--demo_episode",
        type=int,
        default=0,
        help="the index of the episode to make a demo video of",
    )
    parser.add_argument(
        "--feedback_mode",
        type=str,
        default="all",
        help="which type of feedback to use during training; can be either 'all', 'rule_only', 'task_only', 'random', 'random_lorem_ipsum, or 'numerical_reward'",
    )
    parser.add_argument(
        "--level",
        type=str,
        default="GoToRedBallGrey",
        help="the name of the level to train on",
    )
    parser.add_argument(
        "--train_mode",
        type=str,
        default="single_task",
        help="the training mode to use; can be either 'single_task', 'round_robin', 'curriculum_default', 'curriculum_custom', or 'anti_curriculum",
    )
    parser.add_argument(
        "--use_pretrained",
        type=bool,
        default=True,
        help="whether to use the pretrained GPT-2 model",
    )
    parser.add_argument(
        "--sample_interval",
        type=int,
        default=128,
        help="after how many samples to evaluate the sample efficiency of the model; ideally this should be multiples of the chosen batch size.",
    )
    parser.add_argument(
        "--target_return",
        type=int,
        default=1,
        help="the target return to condition on",
    )
    parser.add_argument(
        "--num_repeats",
        type=int,
        default=128,
        help="number of seeds to evaluate over (for validation, this will be 1 / 4)",
    )
    parser.add_argument(
        "--custom_order",
        type=str,
        default="",
        help="the custom order of tasks to use for the curriculum training mode, in the format 'level1,level9,level5,...'",
    )
    parser.add_argument(
        "--predict_feedback",
        type=bool,
        default=False,
        help="whether to also predict feedback during training (besides the action)",
    )
    parser.add_argument(
        "--load_existing_dataset",
        type=bool,
        default=False,
        help="whether to load the dataset if it already exists",
    )
    parser.add_argument(
        "--eps_per_shard",
        type=int,
        default=10,
        help="the number of episodes to collect per dataset shard",
    )
    parser.add_argument(
        "--use_full_ep",
        type=bool,
        default=False,
        help="whether to use the full episode, rather than a given context length",
    )
    parser.add_argument(
        "--ep_dist",
        type=str,
        default="inverse_length",
        help="the distribution from which to sample episodes",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="the number of samples - sub-episodes or full episodes - to train on. If 0, will use all available samples",
    )
    parser.add_argument(
        "--model_seed",
        type=int,
        default=1234567890,
        help="the seed used for seeding different instantiations of the model",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=1,
        help="how many training steps between logging output (for HF Trainer)",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=15,
        help="how many steps to wait for improvements in the evaluation metric before stopping training",
    )
    parser.add_argument(
        "--early_stopping_threshold",
        type=float,
        default=0.02,
        help="the threshold by which improvements in the evaluation metric have to exceed the previous best performance for early stopping",
    )
    return vars(parser.parse_args())
