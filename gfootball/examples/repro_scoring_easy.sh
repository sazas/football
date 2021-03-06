#!/bin/bash

python3 -u -m gfootball.examples.run_ppo2 \
  --level 11_vs_11_kaggle \
  --reward_experiment scoring \
  --policy mlp \
  --cliprange 0.08 \
  --gamma 0.993 \
  --ent_coef 0.003 \
  --num_timesteps 500000000 \
  --max_grad_norm 0.64 \
  --lr 1.6e-5 \
  --num_envs 8 \
  --noptepochs 8 \
  --nminibatches 8 \
  --nsteps 512 \
  --load_path "/home/alex/football/checkpoints/openai-2020-11-26-12-35-02-877222/checkpoints/04200"\
  --opponent "GFootball_with_Memory_Patterns:right_players=1"\
  "$@"
