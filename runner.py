from tensorforce import Agent
from Enviroment import HopfieldEnvironment
from NudgeEnviroment import HopfieldNudgeEnvironment
from utils import plot_multiple, ParseKwargs, ProgressBar
import sys
import pickle as pkl
import time
import pandas as pd
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


def runner(
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

def main(agent_params={}, env_type="hopf", enviroment_params={}, n_episodes=10000, max_step_per_episode=1000, repeats=1):
    for rep in range(repeats):
        print(f"Repeat {rep}")
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
        if env_type == "hopf":
            environment = HopfieldEnvironment(**enviroment_params)
        elif env_type == "hopf-nudge":
            environment = HopfieldNudgeEnvironment(**enviroment_params)
        else:
            raise ValueError(f"Unknown environment type: {(env_type)}")
        
        agent_params = {**default_agent_params, **agent_params}
        agent_params["environment"] = environment
        agent = Agent.create(**agent_params)
        runner(environment, agent, max_step_per_episode, n_episodes)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("Hopfield Spins")
    parser.add_argument("-n", "--n_episodes", type=int, default=10000)
    parser.add_argument("-mspe", "--max_step_per_episode", type=int, default=1000)
    parser.add_argument("--env_type", type=str, choices=["hopf", "hopf-nudge"], default="hopf")
    parser.add_argument("-a", "--agent-param", action=ParseKwargs, default={})
    parser.add_argument("-e", "--env-param", action=ParseKwargs, default={})
    parser.add_argument("-r", "--repeats", type=int, default=1)
    args = parser.parse_args()
    print(args)
    main(args.agent_param, args.env_type, args.env_param, args.n_episodes, args.max_step_per_episode, args.repeats)


