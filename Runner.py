import numpy as np
import threading
import torch
import ray

from ACNet import ACNet
import GroupLock
from Primal2Env import Primal2Env
from Primal2Observer import Primal2Observer
from Map_Generator import maze_generator
from Worker import Worker
from parameters import *


class SimpleCoordinator:
    """Drop-in replacement for tf.train.Coordinator."""

    def __init__(self):
        self._stop_event = threading.Event()

    def should_stop(self):
        return self._stop_event.is_set()

    def request_stop(self):
        self._stop_event.set()

    def join(self, threads, stop_grace_period_secs=120):
        for t in threads:
            t.join(timeout=stop_grace_period_secs)


class Runner:
    def __init__(self, metaAgentID):
        import os
        os.environ["TORCH_DISABLE_ONEDNN"] = "1"
        os.environ["ONEDNN_PRIMITIVE_CACHE_CAPACITY"] = "0"
        self.metaAgentID = metaAgentID

        # device: IL agents run on CPU, RL agents on GPU if available
        if metaAgentID < NUM_IL_META_AGENTS:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.env = Primal2Env(
            num_agents=NUM_THREADS,
            observer=Primal2Observer(observation_size=OBS_SIZE,
                                     num_future_steps=NUM_FUTURE_STEPS),
            map_generator=maze_generator(
                env_size=ENVIRONMENT_SIZE,
                wall_components=WALL_COMPONENTS,
                obstacle_density=OBSTACLE_DENSITY),
            IsDiagonal=DIAG_MVMT,
            isOneShot=False)

        self.local_model = ACNet(NUM_CHANNEL, OBS_SIZE, a_size).to(self.device)
        self.local_model.train()

    # ------------------------------------------------------------------
    # weight sync
    # ------------------------------------------------------------------
    def set_weights(self, weights):
        """weights: list of numpy arrays (one per parameter)"""
        with torch.no_grad():
            for param, w in zip(self.local_model.parameters(), weights):
                param.copy_(torch.FloatTensor(np.array(w)).to(self.device))

    def get_weights(self):
        return [p.cpu().numpy() for p in self.local_model.parameters()]

    # ------------------------------------------------------------------
    # RL multi-threaded job
    # ------------------------------------------------------------------
    def multiThreadedJob(self, episodeNumber):
        workers       = []
        worker_threads = []
        workerNames   = ["worker_" + str(i + 1) for i in range(NUM_THREADS)]
        groupLock     = GroupLock.GroupLock([workerNames, workerNames])

        coord = SimpleCoordinator()

        for a in range(NUM_THREADS):
            agentID = a + 1
            workers.append(Worker(
                self.metaAgentID, agentID, NUM_THREADS,
                self.env, self.local_model, self.device,
                groupLock, learningAgent=True))

        for w in workers:
            groupLock.acquire(0, w.name)
            t = threading.Thread(target=lambda ww=w: ww.work(episodeNumber, coord))
            t.start()
            worker_threads.append(t)

        coord.join(worker_threads)

        jobResults   = []
        loss_metrics = []
        perf_metrics = []
        for w in workers:
            if w.learningAgent:
                jobResults += w.allGradients
            if w.loss_metrics is not None:
                loss_metrics.append(w.loss_metrics)
            if w.perf_metrics is not None:
                perf_metrics.append(w.perf_metrics)

        avg_loss = list(np.mean(np.array(loss_metrics), axis=0)) if loss_metrics else [0.0] * 6

        if perf_metrics:
            pm = np.array(perf_metrics)
            avg_perf = list(np.mean(pm[:, :4], axis=0))
            avg_perf += [np.sum(pm[:, 4]), np.sum(pm[:, 5])]
            all_metrics = avg_loss + avg_perf
        else:
            all_metrics = avg_loss

        return jobResults, all_metrics, False   # is_imitation=False

    # ------------------------------------------------------------------
    # IL job
    # ------------------------------------------------------------------
    def imitationLearningJob(self, episodeNumber):
        worker = Worker(
            self.metaAgentID, None, NUM_THREADS,
            self.env, self.local_model, self.device,
            None, learningAgent=True)

        gradients, losses = worker.imitation_learning_only(episodeNumber)
        mean_loss = [np.mean(losses)] if losses else [0.0]
        return gradients, mean_loss, True   # is_imitation=True

    # ------------------------------------------------------------------
    # main entry point called by driver
    # ------------------------------------------------------------------
    def job(self, global_weights, episodeNumber):
        print(f"episode {episodeNumber} | metaAgent {self.metaAgentID}")
        self.set_weights(global_weights)

        if self.metaAgentID < NUM_IL_META_AGENTS:
            jobResults, metrics, is_imitation = self.imitationLearningJob(episodeNumber)
        elif COMPUTE_TYPE == COMPUTE_OPTIONS.multiThreaded:
            jobResults, metrics, is_imitation = self.multiThreadedJob(episodeNumber)
        else:
            raise NotImplementedError

        info = {
            "id":             self.metaAgentID,
            "episode_number": episodeNumber,
            "is_imitation":   is_imitation,
        }
        return jobResults, metrics, info


@ray.remote(num_cpus=3, num_gpus=1.0 / max(1, NUM_META_AGENTS - NUM_IL_META_AGENTS))
class RLRunner(Runner):
    def __init__(self, metaAgentID):
        super().__init__(metaAgentID)


@ray.remote(num_cpus=1, num_gpus=0)
class imitationRunner(Runner):
    def __init__(self, metaAgentID):
        super().__init__(metaAgentID)
