import os

from stable_baselines import PPO2
from stable_baselines.common import make_vec_env

for env_id in ["AntBulletEnv-v0"]:
    for seed in [2]:

        log_dir = "./logs/%s/fri_sumdim1_nodropout_seed%s_zsize16_nomask_coef001_mlp6464_huber_head8_depth2" % (env_id, seed)
        os.makedirs(log_dir, exist_ok=True)
        env = make_vec_env(env_id, 1, seed, monitor_dir=log_dir)
        model = PPO2('MlpPolicy', env, verbose=1, seed=seed,
                     transformer_coef=.01, vf_coef=0.5, tensorboard_log=log_dir)
        model.learn(total_timesteps=1000000, tb_log_name="tb/transformPPO")
