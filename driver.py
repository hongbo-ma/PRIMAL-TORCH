import os
import numpy as np
import torch
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

from ACNet import ACNet, GRAD_CLIP
from parameters import *


def get_lr(step):
    if ADAPT_LR:
        return LR_Q / np.sqrt(ADAPT_COEFF * step + 1.0)
    return LR_Q


def apply_gradients(global_model, optimizer, gradient_list, lock, step):
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    with lock:
        optimizer.zero_grad()
        for grads in gradient_list:
            for param, g in zip(global_model.parameters(), grads):
                g_tensor = torch.FloatTensor(np.ascontiguousarray(g))
                if param.grad is None:
                    param.grad = g_tensor.clone()
                else:
                    param.grad += g_tensor
        torch.nn.utils.clip_grad_norm_(global_model.parameters(), GRAD_CLIP)
        optimizer.step()


def worker_process(metaAgentID, global_model, lock, result_queue, episode_counter):
    """Each subprocess runs this function."""
    # must set before importing torch ops
    os.environ["TORCH_DISABLE_ONEDNN"] = "1"

    from Runner import Runner

    runner = Runner(metaAgentID, global_model, lock)

    while True:
        episode = episode_counter.value
        with episode_counter.get_lock():
            episode_counter.value += 1

        weights = [p.data.cpu().numpy().copy() for p in global_model.parameters()]
        job_results, metrics, info = runner.job(weights, episode)
        result_queue.put((job_results, metrics, info))


def main():
    mp.set_start_method('spawn', force=True)

    for path in [model_path, gifs_path, train_path]:
        os.makedirs(path, exist_ok=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    global_model = ACNet(NUM_CHANNEL, OBS_SIZE, a_size).to(device)
    global_model.share_memory()

    optimizer = optim.NAdam(global_model.parameters(), lr=LR_Q)
    lock      = mp.Lock()
    writer    = SummaryWriter(log_dir=train_path)

    curr_episode = mp.Value('i', 0)

    if load_model:
        ckpt = torch.load(os.path.join(model_path, 'model_latest.pt'), map_location=device)
        global_model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        curr_episode.value = ckpt['episode']
        print(f"Loaded model at episode {curr_episode.value}")

    result_queue = mp.Queue(maxsize=NUM_META_AGENTS * 2)

    processes = []
    for i in range(NUM_META_AGENTS):
        p = mp.Process(
            target=worker_process,
            args=(i, global_model, lock, result_queue, curr_episode),
            daemon=True)
        p.start()
        processes.append(p)

    tensorboard_data = []
    num_il_episodes  = 0
    num_rl_episodes  = 0
    step             = 0

    try:
        while True:
            job_results, metrics, info = result_queue.get()

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
                writer.add_scalar('Perf/ RL IL ratio', num_rl_episodes / total, step)
            writer.add_scalar('Perf/Learning Rate', get_lr(step), step)

            if job_results:
                apply_gradients(global_model, optimizer, job_results, lock, step)

            if len(tensorboard_data) >= SUMMARY_WINDOW:
                data = np.mean(np.array(tensorboard_data), axis=0)
                tags = ['Losses/Value Loss', 'Losses/Policy Loss', 'Losses/Valid Loss',
                        'Losses/Entropy Loss', 'Losses/Grad Norm', 'Losses/Var Norm',
                        'Perf/Length', 'Perf/Mean Value', 'Perf/Invalid Rate',
                        'Perf/Stop Rate', 'Perf/Reward', 'Perf/Targets Done']
                for tag, val in zip(tags, data):
                    writer.add_scalar(tag, val, step)
                tensorboard_data = []

            step += 1

            if step % 100 == 0:
                torch.save({
                    'model':     global_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'episode':   curr_episode.value,
                }, os.path.join(model_path, f'model_{step}.pt'))
                torch.save({
                    'model':     global_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'episode':   curr_episode.value,
                }, os.path.join(model_path, 'model_latest.pt'))
                print(f"Saved checkpoint at step {step}")

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        for p in processes:
            p.terminate()
        writer.close()


if __name__ == '__main__':
    main()
