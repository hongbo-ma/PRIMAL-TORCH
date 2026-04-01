import numpy as np
import copy
import torch
import imageio
from Env_Builder import *
from Map_Generator import maze_generator
from parameters import *


def discount(x, gamma):
    import scipy.signal as signal
    return signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


class Worker:
    def __init__(self, metaAgentID, workerID, workers_per_metaAgent,
                 env, local_model, device, groupLock, learningAgent):
        self.metaAgentID    = metaAgentID
        self.agentID        = workerID
        self.name           = "worker_" + str(workerID)
        self.num_workers    = workers_per_metaAgent
        self.env            = env
        self.local_model    = local_model
        self.device         = device
        self.groupLock      = groupLock
        self.learningAgent  = learningAgent
        self.allGradients   = []
        self.loss_metrics   = None
        self.perf_metrics   = None

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _obs_to_tensor(self, obs):
        """obs[0]: (C,H,W) ndarray  →  (1,C,H,W) float tensor"""
        return torch.FloatTensor(obs[0]).unsqueeze(0).to(self.device)

    def _goal_to_tensor(self, obs):
        """obs[1]: [dx,dy,mag]  →  (1,3) float tensor"""
        return torch.FloatTensor(obs[1]).unsqueeze(0).to(self.device)

    # ------------------------------------------------------------------
    # gradient computation
    # ------------------------------------------------------------------
    def calculateImitationGradient(self, rollout, episode_count):
        """
        rollout row: [obs, goal_vec, optimal_action, train_imitation_flag]
        """
        rollout = np.array(rollout, dtype=object)
        obs_stack = np.stack(rollout[:, 0]).astype(np.float32, copy=True)
        if obs_stack.ndim != 4:
            raise RuntimeError(f"obs_batch shape error: {obs_stack.shape}, expected (B, C, H, W)")
        obs_batch    = torch.from_numpy(obs_stack).to(self.device)
        goal_batch   = torch.from_numpy(np.stack(rollout[:, 1]).astype(np.float32, copy=True)).to(self.device)
        opt_actions  = torch.LongTensor(np.stack(rollout[:, 2])).to(self.device)
        train_flags  = torch.FloatTensor(rollout[:, 3].astype(np.float32)).to(self.device)

        hx, cx = self.local_model.get_init_hidden(batch_size=len(rollout), device=self.device)

        self.local_model.zero_grad()
        policy, _, _, _, _ = self.local_model(obs_batch, goal_batch, hx, cx)

        # cross-entropy imitation loss, masked by train_flags
        ce = F.cross_entropy(torch.log(policy.clamp(1e-10, 1.0)), opt_actions, reduction='none')
        imitation_loss = (train_flags * ce).mean()

        imitation_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), GRAD_CLIP)

        grads = [p.grad.clone() if p.grad is not None else torch.zeros_like(p)
                 for p in self.local_model.parameters()]
        return [imitation_loss.item()], grads

    def calculateGradient(self, rollout, bootstrap_value, episode_count, rnn_state0):
        rollout = np.array(rollout, dtype=object)
        observations = np.stack(rollout[:, 0])
        goals        = np.stack(rollout[:, -3])
        actions      = rollout[:, 1].astype(int)
        rewards      = rollout[:, 2].astype(float)
        values       = rollout[:, 4].astype(float)
        valids_mask  = np.stack(rollout[:, 5])   # (T, a_size)
        train_value  = rollout[:, -2].astype(float)
        train_policy = rollout[:, -1].astype(float)

        # GAE
        rewards_plus   = np.append(rewards, bootstrap_value)
        disc_returns   = discount(rewards_plus, gamma)[:-1]
        value_plus     = np.append(values, bootstrap_value)
        advantages     = rewards + gamma * value_plus[1:] - value_plus[:-1]
        advantages     = discount(advantages, gamma)

        # to tensors
        obs_t    = torch.FloatTensor(observations).to(self.device)
        goal_t   = torch.FloatTensor(goals).to(self.device)
        act_t    = torch.LongTensor(actions).to(self.device)
        ret_t    = torch.FloatTensor(disc_returns).to(self.device)
        adv_t    = torch.FloatTensor(advantages).to(self.device)
        tv_t     = torch.FloatTensor(train_value).to(self.device)
        tp_t     = torch.FloatTensor(train_policy).to(self.device)
        valid_t  = torch.FloatTensor(valids_mask).to(self.device)

        hx, cx = rnn_state0
        self.local_model.zero_grad()
        policy, value, valids, _, _ = self.local_model(obs_t, goal_t, hx, cx)

        # losses (matching original formulation)
        value_loss = 0.1 * (tv_t * (ret_t - value.squeeze(-1)) ** 2).mean()

        resp = policy.gather(1, act_t.unsqueeze(1)).squeeze(1)
        policy_loss = -0.5 * (tp_t * torch.log(resp.clamp(1e-15, 1.0)) * adv_t).mean()

        valid_loss = -16.0 * (
            tp_t.unsqueeze(1) * (
                torch.log(valids.clamp(1e-10, 1.0)) * valid_t +
                torch.log((1 - valids).clamp(1e-10, 1.0)) * (1 - valid_t)
            )
        ).mean()

        entropy = -(policy * torch.log(policy.clamp(1e-10, 1.0))).mean()

        loss = value_loss + policy_loss + valid_loss - 0.01 * entropy
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), GRAD_CLIP)

        grads = [p.grad.clone() if p.grad is not None else torch.zeros_like(p)
                 for p in self.local_model.parameters()]

        metrics = [value_loss.item(), policy_loss.item(), valid_loss.item(),
                   entropy.item(),
                   float(np.sqrt(sum(g.norm().item()**2 for g in grads))),
                   float(np.sqrt(sum(p.norm().item()**2 for p in self.local_model.parameters())))]
        return metrics, grads

    # ------------------------------------------------------------------
    # imitation learning episode
    # ------------------------------------------------------------------
    def imitation_learning_only(self, episode_count):
        self.env._reset()
        rollouts, targets_done = self.parse_path(episode_count)
        if rollouts is None:
            return None, 0

        gradients, losses = [], []
        for i in range(self.num_workers):
            train_buffer = rollouts[i]
            imitation_loss, grads = self.calculateImitationGradient(train_buffer, episode_count)
            gradients.append(grads)
            losses.append(imitation_loss)
        return gradients, losses

    # ------------------------------------------------------------------
    # RL episode (multi-threaded, called from Runner)
    # ------------------------------------------------------------------
    def run_episode_multithreaded(self, episode_count, coord):
        episode_buffer, episode_values = [], []
        episode_reward = episode_step_count = episode_inv_count = 0
        targets_done = episode_stop_count = 0

        if self.agentID == 1:
            self.env._reset()
            joint_observations[self.metaAgentID] = self.env._observe()

        self.synchronize()

        validActions = self.env.listValidActions(
            self.agentID, joint_observations[self.metaAgentID][self.agentID])
        s = joint_observations[self.metaAgentID][self.agentID]

        hx, cx = self.local_model.get_init_hidden(batch_size=1, device=self.device)
        rnn_state0 = (hx.clone(), cx.clone())

        self.synchronize()
        swarm_reward[self.metaAgentID]  = 0
        swarm_targets[self.metaAgentID] = 0

        self.env.finished = False
        while not self.env.finished:
            obs_t  = self._obs_to_tensor(s)
            goal_t = self._goal_to_tensor(s)

            with torch.no_grad():
                policy, value, _, hx, cx = self.local_model(obs_t, goal_t, hx, cx)

            a_dist = policy.cpu().numpy().flatten()
            v      = value.cpu().item()

            train_policy = train_val = 1
            if np.argmax(a_dist) not in validActions:
                episode_inv_count += 1
                train_val = 0

            train_valid = np.zeros(a_size)
            train_valid[validActions] = 1

            valid_dist = a_dist[validActions]
            valid_dist /= valid_dist.sum()
            a = validActions[np.random.choice(len(validActions), p=valid_dist)]
            joint_actions[self.metaAgentID][self.agentID] = a
            if a == 0:
                episode_stop_count += 1

            self.synchronize()

            if self.agentID == 1:
                all_obs, all_rewards = self.env.step_all(joint_actions[self.metaAgentID])
                for i in range(1, self.num_workers + 1):
                    joint_observations[self.metaAgentID][i] = all_obs[i]
                    joint_rewards[self.metaAgentID][i]      = all_rewards[i]
                    joint_done[self.metaAgentID][i]         = (self.env.world.agents[i].status == 1)

            self.synchronize()

            s1          = joint_observations[self.metaAgentID][self.agentID]
            r           = copy.deepcopy(joint_rewards[self.metaAgentID][self.agentID])
            validActions = self.env.listValidActions(self.agentID, s1)

            self.synchronize()

            episode_buffer.append([s[0], a, r, s1, v, train_valid, s[1], train_val, train_policy])
            episode_values.append(v)
            episode_reward      += r
            episode_step_count  += 1
            s = s1

            buf_full  = len(episode_buffer) % EXPERIENCE_BUFFER_SIZE == 0
            agent_done = joint_done[self.metaAgentID][self.agentID]
            ep_end    = episode_step_count == max_episode_length

            if len(episode_buffer) > 1 and (buf_full or agent_done or ep_end):
                train_buffer = (episode_buffer[-EXPERIENCE_BUFFER_SIZE:]
                                if len(episode_buffer) >= EXPERIENCE_BUFFER_SIZE
                                else episode_buffer[:])

                if agent_done:
                    s1_value = 0.0
                    episode_buffer = []
                    joint_done[self.metaAgentID][self.agentID] = False
                    targets_done += 1
                else:
                    obs_t2  = self._obs_to_tensor(s)
                    goal_t2 = self._goal_to_tensor(s)
                    with torch.no_grad():
                        _, v2, _, _, _ = self.local_model(obs_t2, goal_t2, hx, cx)
                    s1_value = v2.cpu().item()

                self.loss_metrics, grads = self.calculateGradient(
                    train_buffer, s1_value, episode_count, rnn_state0)
                self.allGradients.append(grads)
                rnn_state0 = (hx.clone(), cx.clone())

            self.synchronize()

            if episode_step_count >= max_episode_length:
                break

        swarm_reward[self.metaAgentID]  += episode_reward
        swarm_targets[self.metaAgentID] += targets_done

        self.perf_metrics = np.array([
            episode_step_count,
            np.nanmean(episode_values),
            episode_inv_count,
            episode_stop_count,
            episode_reward,
            targets_done
        ])
        return self.perf_metrics

    def synchronize(self):
        if not hasattr(self, "lock_bool"):
            self.lock_bool = False
        self.groupLock.release(int(self.lock_bool), self.name)
        self.groupLock.acquire(int(not self.lock_bool), self.name)
        self.lock_bool = not self.lock_bool

    def work(self, currEpisode, coord, allVariables=None):
        self.currEpisode = currEpisode
        if COMPUTE_TYPE == COMPUTE_OPTIONS.multiThreaded:
            self.perf_metrics = self.run_episode_multithreaded(currEpisode, coord)
        else:
            raise NotImplementedError

    # ------------------------------------------------------------------
    # imitation: parse M* path into rollout
    # ------------------------------------------------------------------
    def parse_path(self, episode_count):
        result         = [[] for _ in range(self.num_workers)]
        actions        = {}
        o              = {}
        train_imitation = {}
        targets_done   = 0
        single_done    = False
        new_call       = False
        new_MSTAR_call = False

        all_obs = self.env._observe()
        for agentID in range(1, self.num_workers + 1):
            o[agentID]              = all_obs[agentID]
            train_imitation[agentID] = 1

        step_count = 0
        while step_count <= IL_MAX_EP_LENGTH:
            path = self.env.expert_until_first_goal()
            if path is None:
                if step_count != 0:
                    return result, targets_done
                return None, 0

            none_on_goal = True
            path_step    = 1
            while none_on_goal and step_count <= IL_MAX_EP_LENGTH:
                completed_agents = []
                start_positions  = []
                goals            = []
                for i in range(self.num_workers):
                    agent_id = i + 1
                    next_pos = path[path_step][i]
                    diff     = tuple_minus(next_pos, self.env.world.getPos(agent_id))
                    actions[agent_id] = dir2action(diff)

                all_obs, _ = self.env.step_all(actions)
                for i in range(self.num_workers):
                    agent_id = i + 1
                    result[i].append([o[agent_id][0], o[agent_id][1],
                                      actions[agent_id], train_imitation[agent_id]])
                    if self.env.world.agents[agent_id].status == 1:
                        completed_agents.append(i)
                        targets_done += 1
                        single_done   = True
                        if targets_done % MSTAR_CALL_FREQUENCY == 0:
                            new_MSTAR_call = True
                        else:
                            new_call = True

                if single_done and new_MSTAR_call:
                    path = self.env.expert_until_first_goal()
                    if path is None:
                        return result, targets_done
                    path_step = 0
                elif single_done and new_call:
                    path = path[path_step:]
                    path = [list(state) for state in path]
                    for finished_agent in completed_agents:
                        path = merge_plans(path, [None] * len(path), finished_agent)
                    try:
                        while path[-1] == path[-2]:
                            path = path[:-1]
                    except Exception:
                        assert len(path) <= 2
                    start_positions_dir = self.env.getPositions()
                    goals_dir           = self.env.getGoals()
                    for i in range(1, self.env.world.num_agents + 1):
                        start_positions.append(start_positions_dir[i])
                        goals.append(goals_dir[i])
                    world = self.env.getObstacleMap()
                    try:
                        path = priority_planner(world, tuple(start_positions), tuple(goals), path)
                    except Exception:
                        path = self.env.expert_until_first_goal()
                        if path is None:
                            return result, targets_done
                    path_step = 0

                o          = all_obs
                step_count += 1
                path_step  += 1
                new_call       = False
                new_MSTAR_call = False

        return result, targets_done

    def shouldRun(self, coord, episode_count=None):
        if TRAINING:
            return not coord.should_stop()
        return False
