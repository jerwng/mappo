#!/bin/sh
env="MPE"
scenario="simple_tag"
num_landmarks=3
num_agents=3
algo="mat"
exp="check"
seed_max=1

echo "env is ${env}"
for seed in `seq ${seed_max}`
do
    CUDA_VISIBLE_DEVICES=0 python /home/chenyuanwang01/mappo/onpolicy/scripts/render/render_mpe.py --save_gifs --env_name ${env} --algorithm_name ${algo} \
    --experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} \
    --n_training_threads 1 --n_rollout_threads 1 --use_render --episode_length 25 --render_episodes 5 --use_wandb \
    --model_dir "/home/chenyuanwang01/mappo/onpolicy/scripts/results/MPE/simple_tag/mat/check/run48/models/transformer_0.pt"
done
