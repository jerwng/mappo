import wandb
import os
import numpy as np
import torch
from tensorboardX import SummaryWriter
from onpolicy.utils.shared_buffer import SharedReplayBuffer

def _t2n(x):
    """Convert torch tensor to a numpy array."""
    return x.detach().cpu().numpy()

class Runner(object):
    """
    Base class for training recurrent policies.
    :param config: (dict) Config dictionary containing parameters for training.
    """
    def __init__(self, config):

        self.all_args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.num_agents = config['num_agents']
        self.num_good_agents = config['num_good_agents']
        self.num_adversaries = config['num_adversaries']
        if config.__contains__("render_envs"):
            self.render_envs = config['render_envs']       

        # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.n_render_rollout_threads = self.all_args.n_render_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.use_wandb = self.all_args.use_wandb
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N

        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        # dir
        self.model_dir_adversary = self.all_args.model_dir_adversary
        self.model_dir_good_agent = self.all_args.model_dir_good_agent

        if self.use_wandb:
            self.save_dir = str(wandb.run.dir)
            self.run_dir = str(wandb.run.dir)
        else:
            self.run_dir = config["run_dir"]
            self.log_dir = str(self.run_dir / 'logs')
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            self.writter = SummaryWriter(self.log_dir)
            self.save_dir_adversary = str(self.run_dir / 'models_adversary')
            self.save_dir_good_agent = str(self.run_dir / 'models_good_agent')
            if not os.path.exists(self.save_dir_adversary):
                os.makedirs(self.save_dir_adversary)

            if not os.path.exists(self.save_dir_good_agent):
                os.makedirs(self.save_dir_good_agent)

        if self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":
            from onpolicy.algorithms.mat.mat_trainer import MATTrainer as TrainAlgo
            from onpolicy.algorithms.mat.algorithm.transformer_policy import TransformerPolicy as Policy
        else:
            from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO as TrainAlgo
            from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy

        self.adversary_id = 0
        self.good_agent_id = self.num_adversaries # Good agents come after adversary agents

        share_observation_space_adversary = self.envs.share_observation_space[self.adversary_id] if self.use_centralized_V else self.envs.observation_space[self.adversary_id]
        share_observation_space_good_agent = self.envs.share_observation_space[self.good_agent_id] if self.use_centralized_V else self.envs.observation_space[self.good_agent_id]
        
        print("obs_space: ", self.envs.observation_space)
        print("share_obs_space: ", self.envs.share_observation_space)
        print("act_space: ", self.envs.action_space)
        
        # policy network
        if self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":
            self.policy_adversary = Policy(self.all_args, self.envs.observation_space[self.adversary_id], share_observation_space_adversary, self.envs.action_space[self.adversary_id], self.num_adversaries, device = self.device)
            self.policy_good_agent = Policy(self.all_args, self.envs.observation_space[self.good_agent_id], share_observation_space_good_agent, self.envs.action_space[self.good_agent_id], self.num_good_agents, device = self.device)
        else:
            self.policy_adversary = Policy(self.all_args, self.envs.observation_space[self.adversary_id], share_observation_space_adversary, self.envs.action_space[self.adversary_id], device = self.device)
            self.policy_good_agent = Policy(self.all_args, self.envs.observation_space[self.good_agent_id], share_observation_space_good_agent, self.envs.action_space[self.good_agent_id], device = self.device)

        if self.model_dir_adversary is not None and self.model_dir_good_agent is not None:
            self.restore(self.model_dir_adversary, self.model_dir_good_agent)

        # algorithm
        if self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":
            self.trainer_adversary = TrainAlgo(self.all_args, self.policy_adversary, self.num_adversaries, device = self.device)
            self.trainer_good_agent = TrainAlgo(self.all_args, self.policy_good_agent, self.num_good_agents, device = self.device)
        else:
            self.trainer_adversary = TrainAlgo(self.all_args, self.policy_adversary, device = self.device)
            self.trainer_good_agent = TrainAlgo(self.all_args, self.policy_good_agent, device = self.device)
        
        # buffer
        self.buffer_adversary = SharedReplayBuffer(self.all_args,
                                        self.num_adversaries,
                                        self.envs.observation_space[self.adversary_id],
                                        share_observation_space_adversary,
                                        self.envs.action_space[self.adversary_id])

        self.buffer_good_agent = SharedReplayBuffer(self.all_args,
                                        self.num_good_agents,
                                        self.envs.observation_space[self.good_agent_id],
                                        share_observation_space_good_agent,
                                        self.envs.action_space[self.good_agent_id])

    def run(self):
        """Collect training data, perform training updates, and evaluate policy."""
        raise NotImplementedError

    def warmup(self):
        """Collect warmup pre-training data."""
        raise NotImplementedError

    def collect(self, step):
        """Collect rollouts for training."""
        raise NotImplementedError

    def insert(self, data):
        """
        Insert data into buffer.
        :param data: (Tuple) data to insert into training buffer.
        """
        raise NotImplementedError
    
    @torch.no_grad()
    def compute_adversary(self):
        """Calculate returns for the collected data."""
        self.trainer_adversary.prep_rollout()
        if self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":
            next_values = self.trainer_adversary.policy.get_values(np.concatenate(self.buffer_adversary.share_obs[-1]),
                                                        np.concatenate(self.buffer_adversary.obs[-1]),
                                                        np.concatenate(self.buffer_adversary.rnn_states_critic[-1]),
                                                        np.concatenate(self.buffer_adversary.masks[-1]))
        else:
            next_values = self.trainer_adversary.policy.get_values(np.concatenate(self.buffer_adversary.share_obs[-1]),
                                                        np.concatenate(self.buffer_adversary.rnn_states_critic[-1]),
                                                        np.concatenate(self.buffer_adversary.masks[-1]))
        next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
        self.buffer_adversary.compute_returns(next_values, self.trainer_adversary.value_normalizer)

    @torch.no_grad()
    def compute_good_agent(self):
        """Calculate returns for the collected data."""
        self.trainer_good_agent.prep_rollout()
        if self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":
            next_values = self.trainer_good_agent.policy.get_values(np.concatenate(self.buffer_good_agent.share_obs[-1]),
                                                        np.concatenate(self.buffer_good_agent.obs[-1]),
                                                        np.concatenate(self.buffer_good_agent.rnn_states_critic[-1]),
                                                        np.concatenate(self.buffer_good_agent.masks[-1]))
        else:
            next_values = self.trainer_good_agent.policy.get_values(np.concatenate(self.buffer_good_agent.share_obs[-1]),
                                                        np.concatenate(self.buffer_good_agent.rnn_states_critic[-1]),
                                                        np.concatenate(self.buffer_good_agent.masks[-1]))
        next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
        self.buffer_good_agent.compute_returns(next_values, self.trainer_good_agent.value_normalizer)
    
    def train_adversary(self):
        """Train adversary policies with data in buffer. """
        self.trainer_adversary.prep_training()
        train_infos = self.trainer_adversary.train(self.buffer_adversary)      
        self.buffer_adversary.after_update()
        return train_infos

    def train_good_agent(self):
        """Train good agent policies with data in buffer. """
        self.trainer_good_agent.prep_training()
        train_infos = self.trainer_good_agent.train(self.buffer_good_agent)      
        self.buffer_good_agent.after_update()
        return train_infos

    def save(self, episode=0):
        """Save policy's actor and critic networks."""
        if self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":
            self.policy_adversary.save(self.save_dir_adversary, episode)
            self.policy_good_agent.save(self.save_dir_good_agent, episode)
        else:
            policy_actor_adversary = self.trainer_adversary.policy.actor
            torch.save(policy_actor_adversary.state_dict(), str(self.save_dir_adversary) + "/actor.pt")
            policy_actor_good_agent = self.trainer_good_agent.policy.actor
            torch.save(policy_actor_good_agent.state_dict(), str(self.save_dir_good_agent) + "/actor.pt")

            policy_critic_adversary = self.trainer_adversary.policy.critic
            torch.save(policy_critic_adversary.state_dict(), str(self.save_dir_adversary) + "/critic.pt")
            policy_critic_good_agent = self.trainer_good_agent.policy.critic
            torch.save(policy_critic_good_agent.state_dict(), str(self.save_dir_good_agent) + "/critic.pt")

    def restore(self, model_dir_adversary, model_dir_good_agent):
        """Restore policy's networks from a saved model."""
        if self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":
            self.policy_adversary.restore(model_dir_adversary)
            self.policy_good_agent.restore(model_dir_good_agent)
        else:
            policy_actor_state_dict_adversary = torch.load(str(self.model_dir_adversary) + '/actor.pt')
            self.policy_adversary.actor.load_state_dict(policy_actor_state_dict_adversary)
            if not self.all_args.use_render:
                policy_critic_state_dict_adversary = torch.load(str(self.model_dir_adversary) + '/critic.pt')
                self.policy_adversary.critic.load_state_dict(policy_critic_state_dict_adversary)

            policy_actor_state_dict_good_agent = torch.load(str(self.model_dir_good_agent) + '/actor.pt')
            self.policy_good_agent.actor.load_state_dict(policy_actor_state_dict_good_agent)
            if not self.all_args.use_render:
                policy_critic_state_dict_good_agent = torch.load(str(self.model_dir_good_agent) + '/critic.pt')
                self.policy_good_agent.critic.load_state_dict(policy_critic_state_dict_good_agent)

    def log_train(self, train_infos, total_num_steps):
        """
        Log training info.
        :param train_infos: (dict) information about training update.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, {k: v}, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        """
        Log env info.
        :param env_infos: (dict) information about env state.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in env_infos.items():
            if len(v)>0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)
