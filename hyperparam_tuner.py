from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor


class HyperparamTuner:

    def __init__(self, env):
        # Environment
        self.env = env

        # Define hyperparameter ranges
        self.learning_rates = [1e-4, 3e-4, 1e-3]
        self.gammas = [0.95, 0.99, 0.999]
        self.clip_ranges = [0.1, 0.2, 0.3]
        self.n_steps_options = [1024, 2048, 4096]
        self.batch_sizes = [32, 64, 128]
        self.n_epochs_options = [3, 5, 10]

        # Record results
        self.results = []

    def tune(self):
        trial = 1

        for lr in self.learning_rates:
            for gamma in self.gammas:
                for clip_range in self.clip_ranges:
                    for n_steps in self.n_steps_options:
                        for batch_size in self.batch_sizes:
                            for n_epochs in self.n_epochs_options:
                                print(f"Starting trial {trial}...")

                                # Create environment and wrap with Monitor
                                env = TimeLimit(self.env, max_episode_steps=10000)
                                env = Monitor(env)

                                # Create PPO model with current hyperparameters
                                model = PPO('MlpPolicy', env, learning_rate=lr, gamma=gamma,
                                            n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs,
                                            clip_range=clip_range, verbose=0)

                                # Train the model
                                model.learn(total_timesteps=10000)

                                # Evaluate the model
                                mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)

                                # Record the result
                                result = {
                                    'learning_rate': lr,
                                    'gamma': gamma,
                                    'clip_range': clip_range,
                                    'n_steps': n_steps,
                                    'batch_size': batch_size,
                                    'n_epochs': n_epochs,
                                    'mean_reward': mean_reward
                                }
                                self.results.append(result)
                                print(f"Evaluated: {result}")

                                trial += 1

        # Find the best hyperparameters
        best_result = max(self.results, key=lambda x: x['mean_reward'])
        print('Best hyperparameters:', best_result)

        return best_result
