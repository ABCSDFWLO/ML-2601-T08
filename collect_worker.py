import numpy as np
np.bool = np.bool_
np.int = np.int_
np.float = np.float64
import sys, torch, os, time
sys.path.append('/workspace/thinker/thinker/thinker')
sys.path.append('/workspace/thinker/csokoban')
from thinker.env import Environment
from thinker.net import ActorNet, ModelNet
import thinker.util as util

WORKER_ID = int(sys.argv[1])
CKPT_DIR = '/workspace/base'
SAVE_DIR = '/workspace/data'
N_STEPS = 16000
SAVE_INTERVAL = 1000
os.makedirs(SAVE_DIR, exist_ok=True)

flags = util.parse(['--load_checkpoint', CKPT_DIR])
flags.env_n = 1

actor_ckp = torch.load(f'{CKPT_DIR}/ckp_actor.tar', map_location='cpu')
model_ckp = torch.load(f'{CKPT_DIR}/ckp_model.tar', map_location='cpu')

actor_net = ActorNet(obs_shape=(79,1,1), gym_obs_shape=(3,80,80), num_actions=5, flags=flags)
actor_net.set_weights(actor_ckp['actor_net_state_dict'])
actor_net.eval()

model_net = ModelNet(obs_shape=(3,80,80), num_actions=5, flags=flags)
model_net.set_weights(model_ckp['model_net_state_dict'])
model_net.eval()

final_outs_dict = {}
def hook_fn(module, inp, output):
    final_outs_dict['v'] = inp[0].detach().clone()
actor_net.policy.register_forward_hook(hook_fn)

env = Environment(flags, model_wrap=True)
env_out = env.initial(model_net)
actor_state = actor_net.initial_state(batch_size=1)

all_fo, all_ac, all_obs, all_tr = [], [], [], []
count = 0
t0 = time.time()

while count < N_STEPS:
    with torch.no_grad():
        actor_out, actor_state = actor_net(env_out, actor_state)
    if env_out.cur_t[0, 0] == 0:
        all_fo.append(final_outs_dict['v'].squeeze().numpy())
        all_ac.append(actor_out.action[0].item())
        all_obs.append(env_out.gym_env_out[0, 0].numpy())
        all_tr.append(env_out.model_out[0, 0].squeeze().numpy())
        count += 1
        if count % SAVE_INTERVAL == 0:
            np.save(f'{SAVE_DIR}/fo_w{WORKER_ID}_{count}.npy', np.stack(all_fo[-SAVE_INTERVAL:]))
            np.save(f'{SAVE_DIR}/ac_w{WORKER_ID}_{count}.npy', np.array(all_ac[-SAVE_INTERVAL:]))
            np.save(f'{SAVE_DIR}/obs_w{WORKER_ID}_{count}.npy', np.stack(all_obs[-SAVE_INTERVAL:]))
            np.save(f'{SAVE_DIR}/tr_w{WORKER_ID}_{count}.npy', np.stack(all_tr[-SAVE_INTERVAL:]))
            speed = count / (time.time() - t0)
            print(f'w{WORKER_ID}: {count}/{N_STEPS} ({speed:.1f} s/s)', flush=True)
    action = torch.stack([actor_out.action, actor_out.im_action, actor_out.reset_action], dim=-1)
    env_out = env.step(action, model_net)

print(f'w{WORKER_ID}: done', flush=True)
