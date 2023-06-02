import gymnasium as gym
import matplotlib.pyplot as plt
from minigrid.wrappers import RGBImgObsWrapper
from src.dataset.custom_feedback_verifier import RuleFeedback, TaskFeedback
import cv2
import os

DEFAULT_HARD_ACTIONS = [
    2,
    1,
    2,
    2,
    2,
    2,
    # try to move forward into the wall
    2,
    # try to open the wall
    5,
    1,
    2,
    2,
    0,
    2,
    # try to pick up the door
    3,
    5,
    2,
    2,
    2,
    2,
    2,
    2,
    0,
    2,
    # try to open empty cell
    5,
    2,
    5,
    2,
    2,
    2,
    0,
    2,
    2,
    2,
    # try to move forward into obstacle
    2,
    # try to open obstacle
    5,
    3,
    2,
    # try to open door with wrong key
    5,
    1,
    1,
    2,
    2,
    2,
    2,
    1,
    2,
    # try to open an already open door
    5,
    5,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    1,
    2,
    2,
    # try to pickup while carrying
    3,
    # try to drop on top of another object
    0,
    # try to drop on wall
    4,
    1,
    1,
    4,
    0,
    0,
    3,
    # try to pickup wall
    1,
    3,
    0,
    0,
    2,
    2,
    0,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    0,
    2,
    2,
    2,
    2,
    5,
    4,
    1,
    1,
    2,
    2,
    2,
    2,
    1,
    2,
    2,
    2,
    2,
    2,
    1,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    0,
    2,
    1,
    2,
    # try to drop on top of box
    4,
    5,
    # try to open an open box
    5,
    4,
    0,
    0,
    2,
    0,
    2,
    3,
    1,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    1,
    4,
    3,
    2,
    4,
    0,
    2,
    0,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    0,
    2,
    2,
    1,
    0,
    2,
]
DEFAULT_EASY_ACTIONS = [2, 2, 2, 2, 2, 2, 5, 0, 3, 1, 1, 2, 5, 3, 2, 2, 1, 0, 2, 1]

DEFAULT_HARD_CONFIG = "BabyAI-BossLevel-v0"
DEFAULT_EASY_CONFIG = "BabyAI-GoToRedBall-v0"

DEFAULT_HARD_SEED = 103
DEFAULT_EASY_SEED = 0

ACTION_DICT = {
    0: "left",
    1: "right",
    2: "forward",
    3: "pickup",
    4: "drop",
    5: "toggle (=open)",
}


class DemoVideo:
    def __init__(self, config, seed, actions, demo_mode, output_dir):
        self.config = config
        self.seed = seed
        self.actions = actions
        self.demo_mode = demo_mode
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        self.env = self._instantiate_rgb_env(config, seed)
        self.mission = None

    def _format_mission(self):
        if ", then " in self.mission:
            mission_split = self.mission.split(", then ")
            return f"{mission_split[0]},\nthen {mission_split[1]}"
        if " after you " in self.mission:
            mission_split = self.mission.split(" after you ")
            return f"{mission_split[0]}\nafter you {mission_split[1]}"

    def _format_feedback(self, feedback, action):
        rule_feedback, task_feedback = feedback
        return (
            f"action: {ACTION_DICT[action]}\n"
            + f"RF: {rule_feedback if rule_feedback else '--'}\n"
            + f"TF: {task_feedback if task_feedback else '--'}"
        )

    def _plot_and_save_obs(self, obs, i=None, feedback=None):
        _, ax = plt.subplots(1)
        _ = plt.imshow(obs["image"])
        mission_formatted = self._format_mission()
        feedback_formatted = self._format_feedback(feedback, self.actions[i])
        title = feedback_formatted if feedback else None
        ax.set(title=title, xlabel=mission_formatted, xticks=[], yticks=[])
        image_name = (
            f"step_{i+1 if i >= 99 else (f'0{i+1}' if i >= 9 else f'00{i+1}')}.png"
            if i
            else "000.png"
        )
        plt.savefig(os.path.join(self.output_dir, image_name))
        plt.close()

    def _instantiate_rgb_env(self, config, seed):
        env = RGBImgObsWrapper(gym.make(config))
        obs, _ = env.reset(seed=seed)
        self.mission = env.instrs.surface(env)
        self._plot_and_save_obs(obs)
        return env

    def interact_with_env(self):
        rule_feedback_generator = RuleFeedback()
        task_feedback_generator = TaskFeedback(self.env)
        for i, action in enumerate(self.actions):
            rule_feedback = rule_feedback_generator.verify_feedback(self.env, action)
            output = self.env.step(action)
            task_feedback = task_feedback_generator.verify_feedback(self.env, action)
            self._plot_and_save_obs(
                output[0], feedback=(rule_feedback, task_feedback), i=i
            )

    def make_demo_video_from_images(self):
        video_name = "demo.mp4"
        image_list = [
            image for image in os.listdir(self.output_dir) if image.endswith(".png")
        ]
        image_list.sort()
        frame = cv2.imread(os.path.join(self.output_dir, image_list[0]))
        height, width, _ = frame.shape
        video = cv2.VideoWriter(
            os.path.join(self.output_dir, video_name),
            cv2.VideoWriter_fourcc(*"mp4v"),
            1,
            (width, height),
        )

        for image in image_list:
            video.write(cv2.imread(os.path.join(self.output_dir, image)))

        cv2.destroyAllWindows()
        video.release()

    def make_demo_video(self):
        self.interact_with_env()
        self.make_demo_video_from_images()
