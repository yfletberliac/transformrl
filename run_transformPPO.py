import os

from stable_baselines import PPO2
from stable_baselines.common import make_vec_env

for env_id in ["AntBulletEnv-v0"]:
    for seed in [1, 2, 3, 4]:

        log_dir = "./logs/%s/new_transformPPO_seed%s_zsize64_mask10_coef1_mlp32_L2" % (env_id, seed)
        os.makedirs(log_dir, exist_ok=True)
        env = make_vec_env(env_id, 1, seed, monitor_dir=log_dir)
        model = PPO2('MlpPolicy', env, verbose=1, seed=seed,
                     transformer_coef=1., vf_coef=0.5, tensorboard_log=log_dir)
        model.learn(total_timesteps=1000000, tb_log_name="tb/transformPPO")
