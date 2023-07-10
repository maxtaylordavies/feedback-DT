import os
import sys
import time

# import tensorboardX
import torch_ac

import external_rl.utils as utils
from external_rl.model import ACModel
from external_rl.utils import device

os.environ["PROJECT_STORAGE"] = os.path.join(os.getcwd(), "external_rl/storage")


class PPOAgent:
    """
    PPOAgent is a wrapper around the PPO algorithm implementation from
    https://github.com/lcswillems/rl-starter-files for MinGrid and BabyAI environments.
    """

    def __init__(self, env_name, seed, n_frames):
        self.args = {
            "algo": "ppo",
            "env": env_name,
            "model": None,
            "seed": seed,
            "log_interval": 1,
            "save_interval": 10,
            "procs": 16,
            "frames": n_frames,
            "epochs": 4,
            "batch_size_ppo": 256,
            "frames_per_proc": 128,
            "discount": 0.99,
            "lr": 0.001,
            "gae_lambda": 0.95,
            "entropy_coef": 0.01,
            "value_loss_coef": 0.5,
            "max_grad_norm": 0.5,
            "optim_eps": 1e-8,
            "optim_alpha": 0.99,
            "clip_eps": 0.2,
            "recurrence": 4
            if env_name in ["BabyAI-OpenTwoDoors-v0", "BabyAI-OpenRedBlueDoors-v0"]
            else 1,
            "text": True,
            "argmax": True,
        }
        self.args["mem"] = self.args["recurrence"] > 1
        self.env = utils.make_env(self.args["env"], self.args["seed"])
        self.env.reset()
        self.model_dir = self._get_model_dir()
        self.model = self._get_model()

    def _get_model_dir(self):
        """
        Returns the path to the directory where the model weights are saved.
        """
        default_model_name = f"{self.args['env']}_{self.args['algo']}_seed{self.args['seed']}_frames{self.args['frames']}"
        model_name = self.args["model"] or default_model_name
        return os.path.join("external_rl", utils.get_model_dir(model_name))

    def _get_model(self):
        """
        Returns the model instance for the env and trained weights.
        """
        if not os.path.exists(os.path.join(self.model_dir, "status.pt")):
            self._train_agent()

        return utils.Agent(
            self.env.observation_space,
            self.env.action_space,
            self.model_dir,
            argmax=False,
            num_envs=1,
            use_memory=self.args["mem"],
            use_text=self.args["text"],
        )

    def _train_agent(self):
        """
        Trains the agent for the specified number of frames.
        This corresponds to the train.py script in the original implementation.
        """
        # Load loggers and Tensorboard writer
        txt_logger = utils.get_txt_logger(self.model_dir)
        csv_file, csv_logger = utils.get_csv_logger(self.model_dir)
        # tb_writer = tensorboardX.SummaryWriter(self.model_dir)

        # Log command and all script arguments
        txt_logger.info(f"{' '.join(sys.argv)}\n")
        txt_logger.info(f"{self.args}\n")

        # Set seed for all randomness sources
        utils.seed(self.args["seed"])

        # Set device
        txt_logger.info(f"Device: {device}\n")

        # Load environments
        envs = []
        for i in range(self.args["procs"]):
            envs.append(utils.make_env(self.args["env"], self.args["seed"] + 10000 * i))
        txt_logger.info("Environments loaded\n")

        # Load training status
        try:
            status = utils.get_status(self.model_dir)
        except OSError:
            status = {"num_frames": 0, "update": 0}
        txt_logger.info("Training status loaded\n")

        # Load observations preprocessor
        obs_space, preprocess_obss = utils.get_obss_preprocessor(
            envs[0].observation_space
        )
        if "vocab" in status:
            preprocess_obss.vocab.load_vocab(status["vocab"])
        txt_logger.info("Observations preprocessor loaded")

        # Load model
        acmodel = ACModel(
            obs_space, envs[0].action_space, self.args["mem"], self.args["text"]
        )
        if "model_state" in status:
            acmodel.load_state_dict(status["model_state"])
        acmodel.to(device)
        txt_logger.info("Model loaded\n")
        txt_logger.info(f"{acmodel}\n")

        # Load algo
        algo = torch_ac.PPOAlgo(
            envs,
            acmodel,
            device,
            self.args["frames_per_proc"],
            self.args["discount"],
            self.args["lr"],
            self.args["gae_lambda"],
            self.args["entropy_coef"],
            self.args["value_loss_coef"],
            self.args["max_grad_norm"],
            self.args["recurrence"],
            self.args["optim_eps"],
            self.args["clip_eps"],
            self.args["epochs"],
            self.args["batch_size_ppo"],
            preprocess_obss,
        )

        if "optimizer_state" in status:
            algo.optimizer.load_state_dict(status["optimizer_state"])
        txt_logger.info("Optimizer loaded\n")

        # Train model
        num_frames = status["num_frames"]
        update = status["update"]
        start_time = time.time()

        while num_frames < self.args["frames"]:
            # Update model parameters
            update_start_time = time.time()
            exps, logs1 = algo.collect_experiences()
            logs2 = algo.update_parameters(exps)
            logs = {**logs1, **logs2}
            update_end_time = time.time()

            num_frames += logs["num_frames"]
            update += 1

            # Print logs
            if update % self.args["log_interval"] == 0:
                fps = logs["num_frames"] / (update_end_time - update_start_time)
                duration = int(time.time() - start_time)
                return_per_episode = utils.synthesize(logs["return_per_episode"])
                rreturn_per_episode = utils.synthesize(
                    logs["reshaped_return_per_episode"]
                )
                num_frames_per_episode = utils.synthesize(
                    logs["num_frames_per_episode"]
                )

                header = ["update", "frames", "FPS", "duration"]
                data = [update, num_frames, fps, duration]
                header += ["rreturn_" + key for key in rreturn_per_episode.keys()]
                data += rreturn_per_episode.values()
                header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
                data += num_frames_per_episode.values()
                header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm"]
                data += [
                    logs["entropy"],
                    logs["value"],
                    logs["policy_loss"],
                    logs["value_loss"],
                    logs["grad_norm"],
                ]

                txt_logger.info(
                    "U {} | F {:06} | FPS {:04.0f} | D {} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f}".format(
                        *data
                    )
                )

                header += ["return_" + key for key in return_per_episode.keys()]
                data += return_per_episode.values()

                if status["num_frames"] == 0:
                    csv_logger.writerow(header)
                csv_logger.writerow(data)
                csv_file.flush()

                # for field, value in zip(header, data):
                #     tb_writer.add_scalar(field, value, num_frames)

            # Save status
            if (
                self.args["save_interval"] > 0
                and update % self.args["save_interval"] == 0
            ):
                status = {
                    "num_frames": num_frames,
                    "update": update,
                    "model_state": acmodel.state_dict(),
                    "optimizer_state": algo.optimizer.state_dict(),
                }
                if hasattr(preprocess_obss, "vocab"):
                    status["vocab"] = preprocess_obss.vocab.vocab
                utils.save_status(status, self.model_dir)
                txt_logger.info("Status saved")
