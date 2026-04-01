import os
os.environ["TORCH_DISABLE_ONEDNN"] = "1"

import numpy as np
import threading
import torch

from ACNet import ACNet
import GroupLock
from Primal2Env import Primal2Env
from Primal2Observer import Primal2Observer
from Map_Generator import maze_generator
from Worker import Worker
from parameters import *


class SimpleCoordinator:
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
    """
    Runs in a subprocess. Holds a local model copy, syncs weights from
    global_model (shared memory), computes gradients, returns them via queue.
    """

    def __init__(self, metaAgentID, global_model, lock):
        self.metaAgentID  = metaAgentID
        self.global_model = global_model
        self.lock         = lock

        if metaAgentID < NUM_IL_META_AGENTS:
            self.device = torch.device('cpu')
        else:
            if torch.cuda.is_available():
                gpu_id = (metaAgentID - NUM_IL_META_AGENTS) % torch.cuda.device_count()
                self.device = torch.device(f'cuda:{gpu_id}')
            else:
                self.device = torch.device('cpu')

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

    def set_weights(self, weights):
        with torch.no_grad():
            for param, w in zip(self.local_model.parameters(), weights):
                param.copy_(torch.FloatTensor(np.ascontiguousarray(w)).to(self.device))

    def multiThreadedJob(self, episodeNumber):
        workers        = []
        worker_threads = []
        workerNames    = ["worker_" + str(i + 1) for i in range(NUM_THREADS)]
        groupLock      = GroupLock.GroupLock([workerNames, workerNames])
        coord          = SimpleCoordinator()

        for a in range(NUM_THREADS):
            workers.append(Worker(
                self.metaAgentID, a + 1, NUM_THREADS,
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
            pm       = np.array(perf_metrics)
            avg_perf = list(np.mean(pm[:, :4], axis=0))
            avg_perf += [np.sum(pm[:, 4]), np.sum(pm[:, 5])]
            all_metrics = avg_loss + avg_perf
        else:
            all_metrics = avg_loss

        return jobResults, all_metrics, False

    def imitationLearningJob(self, episodeNumber):
        worker = Worker(
            self.metaAgentID, None, NUM_THREADS,
            self.env, self.local_model, self.device,
            None, learningAgent=True)

        gradients, losses = worker.imitation_learning_only(episodeNumber)
        mean_loss = [np.mean(losses)] if losses else [0.0]
        return gradients, mean_loss, True

    def job(self, global_weights, episodeNumber):
        print(f"episode {episodeNumber} | metaAgent {self.metaAgentID}", flush=True)
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
