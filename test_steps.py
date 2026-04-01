"""
PRIMAL2-TORCH unit tests — run each step in order.
Usage: python test_steps.py
"""
import os
os.environ["TORCH_DISABLE_ONEDNN"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
import traceback
import threading
import numpy as np
import torch


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def ok(msg):
    print(f"  [PASS] {msg}")


def fail(msg, e):
    print(f"  [FAIL] {msg}")
    traceback.print_exc()
    sys.exit(1)


# ─────────────────────────────────────────────
# Step 1: GPU basic
# ─────────────────────────────────────────────
section("Step 1: GPU availability")
try:
    import torch
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    print(f"  GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        mem_gb = props.total_memory / 1024**3
        print(f"  GPU {i}: {props.name}, {mem_gb:.1f} GB")
    ok("GPU check done")
except Exception as e:
    fail("GPU check", e)


# ─────────────────────────────────────────────
# Step 2: ACNet forward + backward on each GPU
# ─────────────────────────────────────────────
section("Step 2: ACNet forward + backward")
try:
    sys.path.insert(0, os.path.dirname(__file__))
    from ACNet import ACNet
    from parameters import NUM_CHANNEL, OBS_SIZE, a_size

    print(f"  NUM_CHANNEL={NUM_CHANNEL}, OBS_SIZE={OBS_SIZE}, a_size={a_size}")

    n_gpus = max(1, torch.cuda.device_count())
    for gpu_id in range(n_gpus):
        device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
        m = ACNet(NUM_CHANNEL, OBS_SIZE, a_size).to(device)
        obs  = torch.zeros(4, NUM_CHANNEL, OBS_SIZE, OBS_SIZE, device=device)
        goal = torch.zeros(4, 3, device=device)
        hx, cx = m.get_init_hidden(batch_size=4, device=device)
        policy, value, valids, hx2, cx2 = m(obs, goal, hx, cx)
        loss = policy.sum() + value.sum()
        loss.backward()
        ok(f"GPU {gpu_id}: forward+backward OK, policy={tuple(policy.shape)}")
except Exception as e:
    fail("ACNet forward/backward", e)


# ─────────────────────────────────────────────
# Step 3: Environment reset + observe
# ─────────────────────────────────────────────
section("Step 3: Environment reset + observe")
try:
    from Primal2Env import Primal2Env
    from Primal2Observer import Primal2Observer
    from Map_Generator import maze_generator
    from parameters import (OBS_SIZE, NUM_FUTURE_STEPS, ENVIRONMENT_SIZE,
                             WALL_COMPONENTS, OBSTACLE_DENSITY, DIAG_MVMT,
                             NUM_THREADS)

    env = Primal2Env(
        num_agents=NUM_THREADS,
        observer=Primal2Observer(observation_size=OBS_SIZE,
                                 num_future_steps=NUM_FUTURE_STEPS),
        map_generator=maze_generator(
            env_size=ENVIRONMENT_SIZE,
            wall_components=WALL_COMPONENTS,
            obstacle_density=OBSTACLE_DENSITY),
        IsDiagonal=DIAG_MVMT,
        isOneShot=False)

    env._reset()
    obs = env._observe()
    agent1_obs = obs[1]
    print(f"  obs[1][0] shape: {agent1_obs[0].shape}")
    print(f"  obs[1][1] (goal vec): {agent1_obs[1]}")
    ok("Environment reset + observe OK")
except Exception as e:
    fail("Environment", e)


# ─────────────────────────────────────────────
# Step 4: IL episode (parse_path)
# ─────────────────────────────────────────────
section("Step 4: Imitation learning episode")
try:
    from Worker import Worker
    from parameters import NUM_THREADS

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    local_model = ACNet(NUM_CHANNEL, OBS_SIZE, a_size).to(device)

    worker = Worker(
        metaAgentID=0,
        workerID=None,
        workers_per_metaAgent=NUM_THREADS,
        env=env,
        local_model=local_model,
        device=device,
        groupLock=None,
        learningAgent=True)

    env._reset()
    rollouts, targets_done = worker.parse_path(episode_count=0)
    if rollouts is None:
        print("  parse_path returned None (M* failed), retrying...")
        for _ in range(5):
            env._reset()
            rollouts, targets_done = worker.parse_path(episode_count=0)
            if rollouts is not None:
                break

    if rollouts is None:
        print("  WARNING: M* consistently fails, skipping IL gradient test")
    else:
        print(f"  rollouts: {len(rollouts)} agents, steps per agent: {len(rollouts[0])}")
        print(f"  targets_done: {targets_done}")
        ok("parse_path OK")

        # test gradient computation on first agent's rollout
        if len(rollouts[0]) > 0:
            loss, grads = worker.calculateImitationGradient(rollouts[0], 0)
            print(f"  IL loss: {loss}")
            print(f"  num grad tensors: {len(grads)}")
            ok("calculateImitationGradient OK")
except Exception as e:
    fail("IL episode", e)


# ─────────────────────────────────────────────
# Step 5: Multi-thread IL (2 agents)
# ─────────────────────────────────────────────
section("Step 5: Multi-thread IL (2 agents)")
try:
    results = {}
    errors  = {}

    def run_il(agent_id):
        try:
            dev = torch.device(f'cuda:{agent_id % max(1, torch.cuda.device_count())}'
                               if torch.cuda.is_available() else 'cpu')
            from Primal2Env import Primal2Env
            from Primal2Observer import Primal2Observer
            from Map_Generator import maze_generator

            env_i = Primal2Env(
                num_agents=NUM_THREADS,
                observer=Primal2Observer(observation_size=OBS_SIZE,
                                         num_future_steps=NUM_FUTURE_STEPS),
                map_generator=maze_generator(
                    env_size=ENVIRONMENT_SIZE,
                    wall_components=WALL_COMPONENTS,
                    obstacle_density=OBSTACLE_DENSITY),
                IsDiagonal=DIAG_MVMT,
                isOneShot=False)

            model_i = ACNet(NUM_CHANNEL, OBS_SIZE, a_size).to(dev)
            w = Worker(agent_id, None, NUM_THREADS, env_i, model_i, dev, None, True)
            grads, losses = w.imitation_learning_only(0)
            results[agent_id] = (grads, losses)
        except Exception as ex:
            errors[agent_id] = ex
            traceback.print_exc()

    threads = [threading.Thread(target=run_il, args=(i,)) for i in range(2)]
    for t in threads: t.start()
    for t in threads: t.join()

    for i in range(2):
        if i in errors:
            print(f"  Agent {i} FAILED: {errors[i]}")
        else:
            print(f"  Agent {i} OK, losses={results[i][1]}")
    if not errors:
        ok("Multi-thread IL OK")
    else:
        print("  Some agents failed — see above")
except Exception as e:
    fail("Multi-thread IL", e)


# ─────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────
section("All steps completed")
print("  Ready to run full training: python driver.py")
