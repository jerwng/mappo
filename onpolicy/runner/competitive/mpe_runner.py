import time
import numpy as np
import torch
from onpolicy.runner.competitive.base_runner import Runner
import wandb
import imageio
import subprocess

def _t2n(x):
    return x.detach().cpu().numpy()

class MPERunner(Runner):
    """Runner class to perform training, evaluation. and data collection for the MPEs. See parent class for details."""
    def __init__(self, config):
        super(MPERunner, self).__init__(config)

    def run(self):
        self.warmup()   

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        # If false, train good agent
        should_train_adversary = True
        interval_episode = 0

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.train_adversary.policy.lr_decay(episode, episodes)
                self.train_good_agent.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                values_ad, actions_ad, action_log_probs_ad, rnn_states_ad, rnn_states_critic_ad, actions_env_ad = self.collect_adversary(step)
                values_ga, actions_ga, action_log_probs_ga, rnn_states_ga, rnn_states_critic_ga, actions_env_ga = self.collect_good_agent(step)

                actions_env = np.concatenate((actions_env_ad, actions_env_ga), axis=1)
                    
                # Obser reward and next obs
                obs_ad, rewards_ad, dones_ad, infos_ad = self.envs.step(actions_env)
                obs_ga, rewards_ga, dones_ga, infos_ga = self.envs.step(actions_env)

                data_ad = obs_ad, rewards_ad, dones_ad, infos_ad, values_ad, actions_ad, action_log_probs_ad, rnn_states_ad, rnn_states_critic_ad
                data_ga = obs_ga, rewards_ga, dones_ga, infos_ga, values_ga, actions_ga, action_log_probs_ga, rnn_states_ga, rnn_states_critic_ga

                # insert data into buffer
                self.insert_adversary(data_ad)
                self.insert_good_agent(data_ga)

            # compute return and update network
            if should_train_adversary:
                self.compute_adversary()
                train_infos_ad = self.train_adversary()            
            else:
                self.compute_good_agent()
                train_infos_ga = self.train_good_agent()
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()

                subprocess.run(["echo", "\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                        .format(self.all_args.scenario_name,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start)))])

                if self.env_name == "MPE":
                    env_infos = {}
                    for agent_id in range(self.num_agents):
                        idv_rews = []

                        infos = infos_ad if agent_id < self.good_agent_id else infos_ga

                        for info in infos:
                            if 'individual_reward' in info[agent_id].keys():
                                idv_rews.append(info[agent_id]['individual_reward'])
                        agent_k = 'agent%i/individual_rewards' % agent_id
                        env_infos[agent_k] = idv_rews

                if should_train_adversary:
                    train_infos_ad["average_episode_rewards"] = np.mean(self.buffer_adversary.rewards) * self.episode_length
                    train_infos_ad["average_episode_rewards"] = np.mean(self.buffer_adversary.rewards) * self.episode_length
                    subprocess.run(["echo", "average adversary episode rewards is {}".format(train_infos_ad["average_episode_rewards"])])
                    self.log_train(train_infos_ad, total_num_steps)
                else:
                    train_infos_ga["average_episode_rewards"] = np.mean(self.buffer_good_agent.rewards) * self.episode_length
                    subprocess.run(["echo", "average good agent episode rewards is {}".format(train_infos_ga["average_episode_rewards"])])
                    self.log_train(train_infos_ga, total_num_steps)

                self.log_env(env_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)
            
            interval_episode += 1
            
            # Flip training between adversary and good agent
            if (should_train_adversary and interval_episode == self.adversary_training_interval) or (not should_train_adversary and interval_episode == self.good_agent_training_interval):
                subprocess.run(["echo", f"episode: {episode}, flipping should_train_adversary from {should_train_adversary} to {not should_train_adversary}"])
                should_train_adversary = not should_train_adversary
                interval_episode = 0

    def warmup(self):
        # reset env
        obs = self.envs.reset()

        obs_adversary = obs[:, :self.good_agent_id]
        obs_good_agent = obs[:, self.good_agent_id:]

        # replay buffer
        if self.use_centralized_V:
            share_obs_adversary = obs_adversary.reshape(self.n_rollout_threads, -1)
            share_obs_adversary = np.expand_dims(share_obs_adversary, 1).repeat(self.num_adversaries, axis=1)
            
            share_obs_good_agent = obs_good_agent.reshape(self.n_rollout_threads, -1)
            share_obs_good_agent = np.expand_dims(share_obs_good_agent, 1).repeat(self.num_good_agents, axis=1)
        else:
            share_obs_adversary = obs_adversary
            share_obs_good_agent = obs_good_agent

        self.buffer_adversary.share_obs[0] = share_obs_adversary.copy()
        self.buffer_adversary.obs[0] = obs_adversary.copy()

        self.buffer_good_agent.share_obs[0] = share_obs_good_agent.copy()
        self.buffer_good_agent.obs[0] = obs_good_agent.copy()

    @torch.no_grad()
    def collect_adversary(self, step):
        self.trainer_adversary.prep_rollout()
        value, action, action_log_prob, rnn_states, rnn_states_critic \
            = self.trainer_adversary.policy.get_actions(np.concatenate(self.buffer_adversary.share_obs[step]),
                            np.concatenate(self.buffer_adversary.obs[step]),
                            np.concatenate(self.buffer_adversary.rnn_states[step]),
                            np.concatenate(self.buffer_adversary.rnn_states_critic[step]),
                            np.concatenate(self.buffer_adversary.masks[step]))
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))
        # rearrange action
        if self.envs.action_space[self.adversary_id].__class__.__name__ == 'MultiDiscrete':
            for i in range(self.envs.action_space[self.adversary_id].shape):
                uc_actions_env = np.eye(self.envs.action_space[self.adversary_id].high[i] + 1)[actions[:, :, i]]
                if i == 0:
                    actions_env = uc_actions_env
                else:
                    actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
        elif self.envs.action_space[self.adversary_id].__class__.__name__ == 'Discrete':
            actions_env = np.squeeze(np.eye(self.envs.action_space[self.adversary_id].n)[actions], 2)
        else:
            raise NotImplementedError

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

    @torch.no_grad()
    def collect_good_agent(self, step):
        self.trainer_good_agent.prep_rollout()
        value, action, action_log_prob, rnn_states, rnn_states_critic \
            = self.trainer_good_agent.policy.get_actions(np.concatenate(self.buffer_good_agent.share_obs[step]),
                            np.concatenate(self.buffer_good_agent.obs[step]),
                            np.concatenate(self.buffer_good_agent.rnn_states[step]),
                            np.concatenate(self.buffer_good_agent.rnn_states_critic[step]),
                            np.concatenate(self.buffer_good_agent.masks[step]))
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))
        # rearrange action
        if self.envs.action_space[self.good_agent_id].__class__.__name__ == 'MultiDiscrete':
            for i in range(self.envs.action_space[self.good_agent_id].shape):
                uc_actions_env = np.eye(self.envs.action_space[self.good_agent_id].high[i] + 1)[actions[:, :, i]]
                if i == 0:
                    actions_env = uc_actions_env
                else:
                    actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
        elif self.envs.action_space[self.good_agent_id].__class__.__name__ == 'Discrete':
            actions_env = np.squeeze(np.eye(self.envs.action_space[self.good_agent_id].n)[actions], 2)
        else:
            raise NotImplementedError

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

    def insert_adversary(self, data):
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        dones = dones[:, :self.good_agent_id]
        obs = obs[:, :self.good_agent_id]
        rewards = rewards[:, :self.good_agent_id]


        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), *self.buffer_adversary.rnn_states_critic.shape[3:]), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_adversaries, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)


        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_adversaries, axis=1)
        else:
            share_obs = obs

        self.buffer_adversary.insert(share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs, values, rewards, masks)

    def insert_good_agent(self, data):
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        dones = dones[:, self.good_agent_id:]
        obs = obs[:, self.good_agent_id:]
        rewards = rewards[:, self.good_agent_id:]

        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), *self.buffer_good_agent.rnn_states_critic.shape[3:]), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_good_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_good_agents, axis=1)
        else:
            share_obs = obs

        self.buffer_good_agent.insert(share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs, values, rewards, masks)

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_rewards = []
        eval_obs = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        for eval_step in range(self.episode_length):
            self.trainer.prep_rollout()
            eval_action, eval_rnn_states = self.trainer.policy.act(np.concatenate(eval_obs),
                                                np.concatenate(eval_rnn_states),
                                                np.concatenate(eval_masks),
                                                deterministic=True)
            eval_actions = np.array(np.split(_t2n(eval_action), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))
            
            if self.eval_envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
                for i in range(self.eval_envs.action_space[0].shape):
                    eval_uc_actions_env = np.eye(self.eval_envs.action_space[0].high[i]+1)[eval_actions[:, :, i]]
                    if i == 0:
                        eval_actions_env = eval_uc_actions_env
                    else:
                        eval_actions_env = np.concatenate((eval_actions_env, eval_uc_actions_env), axis=2)
            elif self.eval_envs.action_space[0].__class__.__name__ == 'Discrete':
                eval_actions_env = np.squeeze(np.eye(self.eval_envs.action_space[0].n)[eval_actions], 2)
            else:
                raise NotImplementedError

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards)
        eval_env_infos = {}
        eval_env_infos['eval_average_episode_rewards'] = np.sum(np.array(eval_episode_rewards), axis=0)
        eval_average_episode_rewards = np.mean(eval_env_infos['eval_average_episode_rewards'])
        print("eval average episode rewards of agent: " + str(eval_average_episode_rewards))
        self.log_env(eval_env_infos, total_num_steps)

    @torch.no_grad()
    def render(self):
        """Visualize the env."""
        envs = self.envs
        
        all_frames = []
        for episode in range(self.all_args.render_episodes):
            obs = envs.reset()
            if self.all_args.save_gifs:
                image = envs.render('rgb_array')[0][0]
                all_frames.append(image)
            else:
                envs.render('human')

            rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
            
            episode_rewards = []
            
            for step in range(self.episode_length):
                calc_start = time.time()

                self.trainer.prep_rollout()
                action, rnn_states = self.trainer.policy.act(np.concatenate(obs),
                                                    np.concatenate(rnn_states),
                                                    np.concatenate(masks),
                                                    deterministic=True)
                actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
                rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))

                if envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
                    for i in range(envs.action_space[0].shape):
                        uc_actions_env = np.eye(envs.action_space[0].high[i]+1)[actions[:, :, i]]
                        if i == 0:
                            actions_env = uc_actions_env
                        else:
                            actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
                elif envs.action_space[0].__class__.__name__ == 'Discrete':
                    actions_env = np.squeeze(np.eye(envs.action_space[0].n)[actions], 2)
                else:
                    raise NotImplementedError

                # Obser reward and next obs
                obs, rewards, dones, infos = envs.step(actions_env)
                episode_rewards.append(rewards)

                rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                if self.all_args.save_gifs:
                    image = envs.render('rgb_array')[0][0]
                    all_frames.append(image)
                    calc_end = time.time()
                    elapsed = calc_end - calc_start
                    if elapsed < self.all_args.ifi:
                        time.sleep(self.all_args.ifi - elapsed)
                else:
                    envs.render('human')

            print("average episode rewards is: " + str(np.mean(np.sum(np.array(episode_rewards), axis=0))))

        if self.all_args.save_gifs:
            imageio.mimsave(str(self.gif_dir) + '/render.gif', all_frames, duration=self.all_args.ifi)
