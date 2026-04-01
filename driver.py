import os
os.environ["TORCH_DISABLE_ONEDNN"] = "1"
os.environ["ONEDNN_PRIMITIVE_CACHE_CAPACITY"] = "0"
import numpy as np
import torch
import torch.optim as optim
import ray
from torch.utils.tensorboard import SummaryWriter

from ACNet import ACNet, GRAD_CLIP
from Runner import imitationRunner, RLRunner
from parameters import *


# ---------------------------------------------------------------------------
# learning-rate schedule:  LR_Q / sqrt(ADAPT_COEFF * step + 1)
# ---------------------------------------------------------------------------
def get_lr(step):
    if ADAPT_LR:
        return LR_Q / np.sqrt(ADAPT_COEFF * step + 1.0)
    return LR_Q


# ---------------------------------------------------------------------------
# apply a list of gradient lists to the global model
# ---------------------------------------------------------------------------
def apply_gradients(global_model, optimizer, gradient_list, step):
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    optimizer.zero_grad()
    params = list(global_model.parameters())

    # accumulate gradients from all workers
    for grads in gradient_list:
        for param, g in zip(params, grads):
            g_tensor = torch.FloatTensor(g)
            if param.grad is None:
                param.grad = g_tensor.clone()
            else:
                param.grad += g_tensor

    # clip and step
    torch.nn.utils.clip_grad_norm_(global_model.parameters(), GRAD_CLIP)
    optimizer.step()


# ---------------------------------------------------------------------------
# tensorboard helpers
# ---------------------------------------------------------------------------
def write_rl_metrics(writer, tensorboard_data, curr_episode):
    data = np.mean(np.array(tensorboard_data), axis=0)
    (value_loss, policy_loss, valid_loss, entropy_loss, grad_norm, var_norm,
     mean_length, mean_value, mean_invalid, mean_stop, mean_reward, mean_finishes) = data

    writer.add_scalar('Losses/Value Loss',   value_loss,   curr_episode)
    writer.add_scalar('Losses/Policy Loss',  policy_loss,  curr_episode)
    writer.add_scalar('Losses/Valid Loss',   valid_loss,   curr_episode)
    writer.add_scalar('Losses/Entropy Loss', entropy_loss, curr_episode)
    writer.add_scalar('Losses/Grad Norm',    grad_norm,    curr_episode)
    writer.add_scalar('Losses/Var Norm',     var_norm,     curr_episode)
    writer.add_scalar('Perf/Reward',         mean_reward,  curr_episode)
    writer.add_scalar('Perf/Targets Done',   mean_finishes, curr_episode)
    writer.add_scalar('Perf/Length',         mean_length,  curr_episode)
    if mean_length > 0:
        writer.add_scalar('Perf/Valid Rate',
                          (mean_length - mean_invalid) / mean_length, curr_episode)
        writer.add_scalar('Perf/Stop Rate',
                          mean_stop / mean_length, curr_episode)


def write_il_metrics(writer, metrics, curr_episode):
    writer.add_scalar('Losses/Imitation Loss', metrics[0], curr_episode)


def write_episode_ratio(writer, num_il, num_rl, curr_episode, step):
    total = num_il + num_rl
    if total > 0:
        writer.add_scalar('Perf/ RL IL ratio Ep.', num_rl / total, curr_episode)
    writer.add_scalar('Perf/Num IL Ep.', num_il, curr_episode)
    writer.add_scalar('Perf/Num RL Ep.', num_rl,  curr_episode)
    writer.add_scalar('Perf/Learning Rate', get_lr(step), curr_episode)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    # directories
    for path in [model_path, gifs_path, train_path]:
        os.makedirs(path, exist_ok=True)

    # Ray: use all available GPUs
    ray.init(
        num_gpus=torch.cuda.device_count(),
        object_store_memory=8 * 1024 ** 3,
        _memory=64 * 1024 ** 3,
        _system_config={"automatic_object_spilling_enabled": True},
    )

    # global model + optimizer (on GPU 0 if available)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    global_model = ACNet(NUM_CHANNEL, OBS_SIZE, a_size).to(device)
    global_model.train()

    optimizer = optim.NAdam(global_model.parameters(), lr=LR_Q)
    writer    = SummaryWriter(log_dir=train_path)

    curr_episode = 0
    if load_model:
        ckpt = torch.load(os.path.join(model_path, 'model_latest.pt'), map_location=device)
        global_model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        curr_episode = ckpt['episode']
        print(f"Loaded model at episode {curr_episode}")

    # current weights as numpy (sent to remote actors)
    def get_weights():
        return [p.detach().cpu().numpy() for p in global_model.parameters()]

    # launch remote actors
    il_agents  = [imitationRunner.remote(i) for i in range(NUM_IL_META_AGENTS)]
    rl_agents  = [RLRunner.remote(i) for i in range(NUM_IL_META_AGENTS, NUM_META_AGENTS)]
    meta_agents = il_agents + rl_agents

    weights  = get_weights()
    job_list = []
    for i, agent in enumerate(meta_agents):
        job_list.append(agent.job.remote(weights, curr_episode))
        curr_episode += 1

    tensorboard_data   = []
    num_il_episodes    = 0
    num_rl_episodes    = 0

    try:
        while True:
            done_id, job_list = ray.wait(job_list)
            job_results, metrics, info = ray.get(done_id)[0]

            if info['is_imitation']:
                if job_results:
                    write_il_metrics(writer, metrics, curr_episode)
                    num_il_episodes += 1
            else:
                if job_results:
                    tensorboard_data.append(metrics)
                    num_rl_episodes += 1

            write_episode_ratio(writer, num_il_episodes, num_rl_episodes,
                                curr_episode, curr_episode)

            # apply gradients
            if job_results:
                apply_gradients(global_model, optimizer, job_results, curr_episode)

            if len(tensorboard_data) >= SUMMARY_WINDOW:
                write_rl_metrics(writer, tensorboard_data, curr_episode)
                tensorboard_data = []

            # send updated weights back
            weights = get_weights()
            curr_episode += 1
            job_list.append(meta_agents[info['id']].job.remote(weights, curr_episode))

            # checkpoint
            if curr_episode % 100 == 0:
                torch.save({
                    'model':     global_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'episode':   curr_episode,
                }, os.path.join(model_path, f'model_{curr_episode}.pt'))
                torch.save({
                    'model':     global_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'episode':   curr_episode,
                }, os.path.join(model_path, 'model_latest.pt'))
                print(f"Saved checkpoint at episode {curr_episode}")

    except KeyboardInterrupt:
        print("Stopping...")
        for a in meta_agents:
            ray.kill(a)
    finally:
        writer.close()


if __name__ == '__main__':
    main()
