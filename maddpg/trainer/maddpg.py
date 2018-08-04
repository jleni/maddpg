import numpy as np
import random
import tensorflow as tf
import maddpg.common.tf_util as U

from maddpg.common.distributions import make_pdtype
from maddpg import AgentTrainer
from maddpg.trainer.replay_buffer import ReplayBuffer


def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma * r
        r = r * (1. - done)
        discounted.append(r)
    return discounted[::-1]


def make_update_exp(vals, target_vals):
    polyak = 1.0 - 1e-2
    expression = []
    for var, var_target in zip(sorted(vals, key=lambda v: v.name), sorted(target_vals, key=lambda v: v.name)):
        expression.append(var_target.assign(polyak * var_target + (1.0 - polyak) * var))
    expression = tf.group(*expression)
    return U.function([], [], updates=[expression])


class MADDPGAgentTrainer(AgentTrainer):
    def __init__(self, name, model, obs_shape_n, act_space_n, agent_index, args, local_q_func=False):
        self.name = name
        self.n = len(obs_shape_n)
        self.agent_index = agent_index
        self.args = args

        obs_ph_n = []
        for i in range(self.n):
            obs_ph_n.append(U.BatchInput(obs_shape_n[i], name="obs{}".format(i)).get())

        q_func = model
        q_optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)
        q_grad_norm_clipping = 0.5

        p_func = model
        p_optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)
        p_grad_norm_clipping = 0.5

        act = None
        p_train_func = None
        update_target_p = None
        p_debug = None

        num_units = args.num_units

        with tf.variable_scope(self.name, reuse=None):
            ##################
            # create distributions
            act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]
            act_space_len = len(act_space_n)

            ##################
            # set up placeholders
            act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action{}".format(i)) for i in range(act_space_len)]
            target_ph = tf.placeholder(tf.float32, [None], name="target")

            ##################
            # input
            with tf.variable_scope("obs_and_act", reuse=None):
                q_input = tf.concat(obs_ph_n + act_ph_n, 1)
            p_input = obs_ph_n[agent_index]
            if local_q_func:
                q_input = tf.concat([obs_ph_n[agent_index], act_ph_n[agent_index]], 1)

            ##################
            # q_func / p_func
            tmp = q_func(q_input, 1, scope="q_func", num_units=num_units)
            with tf.variable_scope("q", reuse=None):
                q = tmp[:, 0]

            p = p_func(p_input, int(act_pdtype_n[agent_index].param_shape()[0]), scope="p_func", num_units=num_units)

            q_func_vars = U.scope_vars(U.absolute_scope_name("q_func"))
            p_func_vars = U.scope_vars(U.absolute_scope_name("p_func"))

            with tf.variable_scope("q_loss", reuse=None):
                q_loss = tf.reduce_mean(tf.square(q - target_ph))

                # viscosity solution to Bellman differential equation in place of an initial condition
                q_reg = tf.reduce_mean(tf.square(q))

            loss = q_loss  # + 1e-3 * q_reg

            q_optimize_expr = U.minimize_and_clip(q_optimizer, loss, q_func_vars, q_grad_norm_clipping)

            # Create callable functions
            q_train_func = U.function(inputs=obs_ph_n + act_ph_n + [target_ph], outputs=loss, updates=[q_optimize_expr])
            q_values = U.function(obs_ph_n + act_ph_n, q)

            # target network
            tmp = q_func(q_input, 1, scope="target_q_func", num_units=num_units)
            with tf.variable_scope("target_q", reuse=None):
                target_q = tmp[:, 0]
            target_q_func_vars = U.scope_vars(U.absolute_scope_name("target_q_func"))

            with tf.variable_scope("q_update", reuse=None):
                update_target_q = make_update_exp(q_func_vars, target_q_func_vars)

            target_q_values = U.function(obs_ph_n + act_ph_n, target_q)

            q_debug = {'q_values': q_values, 'target_q_values': target_q_values}

            ###############################################################################################
            ###############################################################################################
            ###############################################################################################

            # wrap parameters in distribution
            act_pd = act_pdtype_n[agent_index].pdfromflat(p)

            with tf.variable_scope("act_sample", reuse=None):
                act_sample = act_pd.sample()

            with tf.variable_scope("act_n_sample", reuse=None):
                act_input_n = act_ph_n + []
                act_input_n[agent_index] = act_pd.sample()

            with tf.variable_scope("p_loss", reuse=None):
                pg_loss = -tf.reduce_mean(q)
                with tf.variable_scope("p_reg", reuse=None):
                    p_reg = tf.reduce_mean(tf.square(act_pd.flatparam()))
                p_loss = pg_loss + p_reg * 1e-3

            p_optimize_expr = U.minimize_and_clip(p_optimizer, p_loss, p_func_vars, p_grad_norm_clipping)

            # Create callable functions
            p_train_func = U.function(inputs=obs_ph_n + act_ph_n, outputs=p_loss, updates=[p_optimize_expr])
            act = U.function(inputs=[obs_ph_n[agent_index]], outputs=act_sample)
            p_values = U.function([obs_ph_n[agent_index]], p)

            # target network
            target_p = p_func(p_input, int(act_pdtype_n[agent_index].param_shape()[0]),
                              scope="target_p_func",
                              num_units=num_units)
            target_p_func_vars = U.scope_vars(U.absolute_scope_name("target_p_func"))

            with tf.variable_scope("p_update", reuse=None):
                update_target_p = make_update_exp(p_func_vars, target_p_func_vars)

            with tf.variable_scope("target_act_sample", reuse=None):
                target_act_sample = act_pdtype_n[agent_index].pdfromflat(target_p).sample()
            target_act = U.function(inputs=[obs_ph_n[agent_index]], outputs=target_act_sample)

            p_debug = {'p_values': p_values, 'target_act': target_act}

        self.q_train = q_train_func
        self.q_update = update_target_q
        self.q_debug = q_debug

        self.act = act
        self.p_train = p_train_func
        self.p_update = update_target_p
        self.p_debug = p_debug

        # Create experience buffer
        self.replay_buffer = ReplayBuffer(1e6)
        self.max_replay_buffer_len = args.batch_size * args.max_episode_len
        self.replay_sample_index = None

    def action(self, obs):
        return self.act(obs[None])[0]

    def experience(self, obs, act, rew, new_obs, done, terminal):
        # Store transition in the replay buffer.
        self.replay_buffer.add(obs, act, rew, new_obs, float(done))

    def preupdate(self):
        self.replay_sample_index = None

    def update(self, agents, t):
        if len(self.replay_buffer) < self.max_replay_buffer_len:  # replay buffer is not large enough
            return
        if not t % 100 == 0:  # only update every 100 steps
            return

        self.replay_sample_index = self.replay_buffer.make_index(self.args.batch_size)
        # collect replay sample from all agents
        obs_n = []
        obs_next_n = []
        act_n = []
        index = self.replay_sample_index
        for i in range(self.n):
            obs, act, rew, obs_next, done = agents[i].replay_buffer.sample_index(index)
            obs_n.append(obs)
            obs_next_n.append(obs_next)
            act_n.append(act)
        obs, act, rew, obs_next, done = self.replay_buffer.sample_index(index)

        # train q network
        num_sample = 1
        target_q = 0.0
        for i in range(num_sample):
            target_act_next_n = [agents[i].p_debug['target_act'](obs_next_n[i]) for i in range(self.n)]
            target_q_next = self.q_debug['target_q_values'](*(obs_next_n + target_act_next_n))
            target_q += rew + self.args.gamma * (1.0 - done) * target_q_next
        target_q /= num_sample

        try:
            q_loss = self.q_train(*(obs_n + act_n + [target_q]))
        except Exception as e:
            pass

        # train p network
        p_loss = self.p_train(*(obs_n + act_n))

        self.p_update()
        self.q_update()

        return [q_loss, p_loss, np.mean(target_q), np.mean(rew), np.mean(target_q_next), np.std(target_q)]
