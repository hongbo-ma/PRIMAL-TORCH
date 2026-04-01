import os
os.environ["TORCH_DISABLE_ONEDNN"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import torch
import torch.optim as optim
import threading
import queue
from torch.utils.tensorboard import SummaryWriter

from ACNet import ACNet, GRAD_CLIP
from Runner import Runner
from parameters import *


def get_lr(step):
    if ADAPT_LR:
        return LR_Q / np.sqrt(ADAPT_COEFF * step + 1.0)
    return LR_Q


def apply_gradients(global_model, optimizer, gradient_list, step):
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.zero_grad()
    for grads in gradient_list:
        for param, g in zip(global_model.parameters(), grads):
            g_t = torch.FloatTensor(np.ascontiguousarray(g))
            if param.grad is None:
                param.grad = g_t.clone()
            else:
                param.grad += g_t
    torch.nn.utils.clip_grad_norm_(global_model.parameters(), GRAD_CLIP)
    optimizer.step()


def agent_thread(runner, result_queue, stop_event):
    """Each agent runs episodes in a loop, puts results into queue."""
    episode = 0
    while not stop_event.is_set():
        try:
            weights = runner.get_global_weights()
            job_results, metrics, info = runner.job(weights, episode)
            result_queue.put((job_results, metrics, info))
            episode += 1
        except Exception as e:
            print(f"Agent {runner.metaAgentID} error: {e}", flush=True)
            import traceback
            traceback.print_exc()
            break


def main():
    for path in [model_path, gifs_path, train_path]:
        os.makedirs(path, exist_ok=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    global_model = ACNet(NUM_CHANNEL, OBS_SIZE, a_size).to(device)
    global_model.train()

    optimizer = optim.NAdam(global_model.parameters(), lr=LR_Q)
    writer    = SummaryWriter(log_dir=train_path)

    step = 0
    if load_model:
        ckpt = torch.load(os.path.join(model_path, 'model_latest.pt'), map_location=device)
        global_model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        step = ckpt.get('step', 0)
        print(f"Loaded model at step {step}")

    # create runners — all share the same global_model reference
    runners = [Runner(i, global_model) for i in range(NUM_META_AGENTS)]

    result_queue = queue.Queue(maxsize=NUM_META_AGENTS * 4)
    stop_event   = threading.Event()

    threads = []
    for runner in runners:
        t = threading.Thread(target=agent_thread,
                             args=(runner, result_queue, stop_event),
                             daemon=True)
        t.start()
        threads.append(t)

    tensorboard_data = []
    num_il_episodes  = 0
    num_rl_episodes  = 0

    try:
        while True:
            job_results, metrics, info = result_queue.get(timeout=300)

            if info['is_imitation']:
                if job_results:
                    writer.add_scalar('Losses/Imitation Loss', metrics[0], step)
                    num_il_episodes += 1
            else:
                if job_results:
                    tensorboard_data.append(metrics)
                    num_rl_episodes += 1

            total = num_il_episodes + num_rl_episodes
            if total > 0:
                writer.add_scalar('Perf/RL_IL_ratio', num_rl_episodes / total, step)
            writer.add_scalar('Perf/Learning_Rate', get_lr(step), step)

            if job_results:
                apply_gradients(global_model, optimizer, job_results, step)

            if len(tensorboard_data) >= SUMMARY_WINDOW:
                data = np.mean(np.array(tensorboard_data), axis=0)
                tags = ['Losses/Value', 'Losses/Policy', 'Losses/Valid',
                        'Losses/Entropy', 'Losses/GradNorm', 'Losses/VarNorm',
                        'Perf/Length', 'Perf/MeanValue', 'Perf/InvalidRate',
                        'Perf/StopRate', 'Perf/Reward', 'Perf/TargetsDone']
                for tag, val in zip(tags, data):
                    writer.add_scalar(tag, val, step)
                tensorboard_data = []

            step += 1

            if step % 100 == 0:
                ckpt = {'model': global_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'step': step}
                torch.save(ckpt, os.path.join(model_path, f'model_{step}.pt'))
                torch.save(ckpt, os.path.join(model_path, 'model_latest.pt'))
                print(f"Saved checkpoint at step {step}", flush=True)

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        stop_event.set()
        writer.close()


if __name__ == '__main__':
    main()
