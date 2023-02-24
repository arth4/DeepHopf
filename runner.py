from tensorforce import Agent, Runner
from Enviroment import HopfieldEnvironment
from NudgeEnviroment import HopfieldNudgeEnvironment
from utils import plot_multiple, ParseKwargs, ProgressBar
import sys
import pickle as pkl
import time
import pandas as pd
import numpy as np
from os import path, mkdir
def run(environment, agent, n_episodes, max_step_per_episode, combination=1, test=False):
    """
    Train agent for n_episodes
    """
    environment._set_max_actions(max_step_per_episode)

    disable_progress = not sys.stdout.isatty()
    progress_bar = ProgressBar(desc="doing it", total=n_episodes, disable=disable_progress)

    # Loop over episodes
    for i in range(n_episodes):
        # Initialize episode
        episode_length = 0
        states = environment.reset()
        terminal = False
        while not terminal:
            # Run episode
            episode_length += 1
            actions = agent.act(states=states)
            #print("actions", actions)
            #print("states", states)
            states, terminal, reward = environment.execute(actions=actions)
            #print("reward", reward)
            agent.observe(terminal=terminal, reward=reward)
        #print("episode_final_length", episode_length)
        progress_bar.update()
    progress_bar.close()
    return environment.last_reward


def my_runner(
    environment,
    agent,
    max_step_per_episode,
    n_episodes,
    n_episodes_test=1,
    combination=1,
):
    # Train agent
    result_vec = [] #initialize the result list
    for i in range(round(n_episodes / 100)): #Divide the number of episodes into batches of 100 episodes
        if result_vec:
            print("batch", i, "Best result", result_vec[-1]) #Show the results for the current batch
        # Train Agent for 100 episode
        run(environment, agent, 100, max_step_per_episode, combination=combination) 
        # Test Agent for this batch
        test_results = run(
                environment,
                agent,
                n_episodes_test,
                max_step_per_episode,
                combination=combination,
                test=True
            )
        # Append the results for this batch
        result_vec.append(test_results)
    with open(f"DeepQResults.pkl", "wb") as f:
        pkl.dump(result_vec, f) 
    # Plot the evolution of the agent over the batches
    plot_multiple(
        Series=[result_vec],
        labels = ["Reward"],
        xlabel = "episodes",
        ylabel = "Reward",
        title = "Reward vs episodes",
        save_fig=True,
        path="env",
        folder=str(combination),
        time=False,
    )
    #Terminate the agent and the environment
    agent.close()
    environment.close


    

def main(agent_params={}, env_type="hopf", enviroment_params={}, n_episodes=30, n_batches=3, n_test=1, max_step_per_episode=1000, repeat=-1, output_dir=None):
    assert output_dir is not None
    if repeat in [-1]:
        assert not path.exists(output_dir)
        mkdir(output_dir)
    
    if repeat > -1:
        try:
            mkdir(output_dir)
        except FileExistsError:
            pass
        output_dir = path.join(output_dir, f"repeat_{repeat}")
        try:
            mkdir(output_dir)
        except FileExistsError:
            pass
    eval_record = []
    np.random.seed(repeat if repeat > -1 else 0)

    def eval_callback(runner):
        mean = np.mean(runner.evaluation_returns[-0:])
        print(f"mean={mean}  <- {runner.evaluation_returns}")
        eval_record.append(mean)
        return float(mean)

    df = pd.DataFrame(columns=["reward", "new_best"])
    def save_callback(runner, parrallel=False):
        reward = runner.environments[0].last_reward
        new_best = len(df) < 1 or reward > df["reward"].max()
        df.loc[save_callback.i] = [reward, new_best]
        if new_best:
            runner.environments[0].save(path.join(output_dir, f"env_{save_callback.i}"))

        save_callback.i += 1
        return True

    save_callback.i = 0
    output_fn = output_dir

    """
    default_agent_params = dict(
    agent='ppo', environment=HopfieldEnvironment, max_episode_timesteps=1000,
    batch_size=10, update_frequency=10, learning_rate=1e-3, subsampling_fraction=0.2,
    #optimization_steps=100,
    ## Network
    network=dict(type='auto', size=64, depth=8),
    ## Reward estimation
    #likelihood_ratio_clipping=0.2, discount=1.0, estimate_terminal=False,
    ## Critic
    #critic_network='auto',
    #critic_optimizer=dict(optimizer='adam', multi_step=30, learning_rate=1e-3),
    ## Preprocessing
    #preprocessing=None,
    ## Exploration
    #exploration=0.01, variable_noise=0.0,
    ## Regularization
    #l2_regularization=0.1, entropy_regularization=0.1,
    ## TensorFlow etc
    #name='agent', device=None, parallel_interactions=1, seed=None, execution=None, saver=None,
    #summarizer=None, recorder=None,
    )
    """
    default_agent_params = dict(
    agent='ppo', environment=HopfieldEnvironment, max_episode_timesteps=1000,
    # Automatically configured network
    network='auto',
    # PPO optimization parameters
    batch_size=50, update_frequency=2, learning_rate=3e-4, multi_step=10,
    subsampling_fraction=0.33,
    # Reward estimation
    likelihood_ratio_clipping=0.2, discount=0.99, predict_terminal_values=False,
    reward_processing=None,
    # Baseline network and optimizer
    baseline=dict(type='auto', size=32, depth=4),
    baseline_optimizer=dict(optimizer='adam', learning_rate=1e-3, multi_step=10),
    # Regularization
    l2_regularization=0.0, entropy_regularization=0.0,
    # Preprocessing
    state_preprocessing='linear_normalization',
    # Exploration
    exploration=0.05, variable_noise=0.0,
    # Default additional config values
    config=None,

    # Do not record agent-environment interaction trace
    recorder=None
    )
    if env_type == "hopf":
        environment = HopfieldEnvironment(**enviroment_params)
    elif env_type == "hopf-nudge":
        environment = HopfieldNudgeEnvironment(**enviroment_params)
    else:
        raise ValueError(f"Unknown environment type: {(env_type)}")
    
    agent_params = {**default_agent_params, **agent_params}
    agent_params["environment"] = environment
    agent = Agent.create(**agent_params)
    environment._set_max_actions(max_step_per_episode)

    runner = Runner(agent=agent, environment=environment)
    for b in range(n_batches):
        print(f"Batch {b}")
        runner.run(num_episodes=n_episodes, callback=save_callback)
        print(f"Batch {b} test")
        runner.run(num_episodes=n_test, evaluation=True, evaluation_callback=eval_callback)
        df.to_csv(path.join(output_dir, "results.csv"))
    runner.close()
    np.save(output_fn + "_eval.npy", eval_record)
    """
    my_runner(environment, agent, max_step_per_episode, n_episodes)
    """

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("Hopfield Spins")
    parser.add_argument("-ne", "--n_episodes", type=int, default=10000)
    parser.add_argument("-nb", "--n_batches", type=int, default=3)
    parser.add_argument("-nt", "--n_test", type=int, default=1)
    parser.add_argument("-mspe", "--max_step_per_episode", type=int, default=1000)
    parser.add_argument("--env_type", type=str, choices=["hopf", "hopf-nudge"], default="hopf")
    parser.add_argument("-a", "--agent-param", action=ParseKwargs, default={})
    parser.add_argument("-e", "--env-param", action=ParseKwargs, default={})
    parser.add_argument("-r", "--repeat", type=int, default=-1)
    parser.add_argument("-o", "--output", type=str, default=None)
    args = parser.parse_args()
    print(args)
    main(args.agent_param, args.env_type, args.env_param, args.n_episodes, args.n_batches, args.n_test, args.max_step_per_episode, args.repeat, args.output)


