python -m train \
--xpid=seed_1 \
--env_name=MultiGrid-GoalLastVariableBlocksAdversarialEnv-v7 \
--use_gae=True \
--gamma=0.995 \
--gae_lambda=0.95 \
--seed=1 \
--num_control_points=12 \
--recurrent_arch=lstm \
--recurrent_agent=True \
--recurrent_adversary_env=False \
--recurrent_hidden_size=256 \
--use_global_critic=False \
--lr=0.0001 \
--num_steps=256 \
--num_processes=32 \
--num_env_steps=250000000 \
--ppo_epoch=5 \
--num_mini_batch=1 \
--entropy_coef=0.0 \
--value_loss_coef=0.5 \
--clip_param=0.2 \
--clip_value_loss=True \
--adv_entropy_coef=0.0 \
--max_grad_norm=0.5 \
--algo=ppo \
--ued_algo=generative_model \
--use_plr=False \
--level_replay_prob=0.0 \
--level_replay_rho=1.0 \
--level_replay_seed_buffer_size=5000 \
--level_replay_score_transform=rank \
--level_replay_temperature=0.1 \
--staleness_coef=0.3 \
--no_exploratory_grad_updates=False \
--use_editor=False \
--level_editor_prob=0 \
--level_editor_method=random \
--num_edits=0 \
--base_levels=batch \
--use_generator=True \
--generator_model_path=diffusion_human_feedback/log/minigrid_60_uniform/model300000.pt \
--use_guidance=True \
--regret_metric=cvar-mean \
--regret_guidance_weight=5.0 \
--num_tutor=1 \
--tutor_update_iteration=5 \
--tutor_buffer_size=1600 \
--tutor_batch_size=128 \
--use_categorical_tutor=True \
--num_bins_tutor=100 \
--logsumexp_temperature=0.1 \
--cvar_alpha=0.15 \
--return_add_noise_std=0.01 \
--log_interval=25 \
--screenshot_interval=100 \
--env_save_interval=10 \
--env_save_batch_size=32 \
--log_grad_norm=False \
--handle_timelimits=True \
--checkpoint_basis=student_grad_updates \
--archive_interval=5000 \
--test_env_names=MultiGrid-SixteenRooms-v0,MultiGrid-Maze-v0,MultiGrid-Labyrinth-v0 \
--log_dir=~/logs/minigrid_60/add \
--log_action_complexity=True \
--log_plr_buffer_stats=True \
--log_replay_complexity=True \
--reject_unsolvable_seeds=False \
--checkpoint=True

python -m train \
--xpid=seed_2 \
--env_name=MultiGrid-GoalLastVariableBlocksAdversarialEnv-v7 \
--use_gae=True \
--gamma=0.995 \
--gae_lambda=0.95 \
--seed=2 \
--num_control_points=12 \
--recurrent_arch=lstm \
--recurrent_agent=True \
--recurrent_adversary_env=False \
--recurrent_hidden_size=256 \
--use_global_critic=False \
--lr=0.0001 \
--num_steps=256 \
--num_processes=32 \
--num_env_steps=250000000 \
--ppo_epoch=5 \
--num_mini_batch=1 \
--entropy_coef=0.0 \
--value_loss_coef=0.5 \
--clip_param=0.2 \
--clip_value_loss=True \
--adv_entropy_coef=0.0 \
--max_grad_norm=0.5 \
--algo=ppo \
--ued_algo=generative_model \
--use_plr=False \
--level_replay_prob=0.0 \
--level_replay_rho=1.0 \
--level_replay_seed_buffer_size=5000 \
--level_replay_score_transform=rank \
--level_replay_temperature=0.1 \
--staleness_coef=0.3 \
--no_exploratory_grad_updates=False \
--use_editor=False \
--level_editor_prob=0 \
--level_editor_method=random \
--num_edits=0 \
--base_levels=batch \
--use_generator=True \
--generator_model_path=diffusion_human_feedback/log/minigrid_60_uniform/model300000.pt \
--use_guidance=True \
--regret_metric=cvar-mean \
--regret_guidance_weight=5.0 \
--num_tutor=1 \
--tutor_update_iteration=5 \
--tutor_buffer_size=1600 \
--tutor_batch_size=128 \
--use_categorical_tutor=True \
--num_bins_tutor=100 \
--logsumexp_temperature=0.1 \
--cvar_alpha=0.15 \
--return_add_noise_std=0.01 \
--log_interval=25 \
--screenshot_interval=100 \
--env_save_interval=10 \
--env_save_batch_size=32 \
--log_grad_norm=False \
--handle_timelimits=True \
--checkpoint_basis=student_grad_updates \
--archive_interval=5000 \
--test_env_names=MultiGrid-SixteenRooms-v0,MultiGrid-Maze-v0,MultiGrid-Labyrinth-v0 \
--log_dir=~/logs/minigrid_60/add \
--log_action_complexity=True \
--log_plr_buffer_stats=True \
--log_replay_complexity=True \
--reject_unsolvable_seeds=False \
--checkpoint=True

python -m train \
--xpid=seed_3 \
--env_name=MultiGrid-GoalLastVariableBlocksAdversarialEnv-v7 \
--use_gae=True \
--gamma=0.995 \
--gae_lambda=0.95 \
--seed=3 \
--num_control_points=12 \
--recurrent_arch=lstm \
--recurrent_agent=True \
--recurrent_adversary_env=False \
--recurrent_hidden_size=256 \
--use_global_critic=False \
--lr=0.0001 \
--num_steps=256 \
--num_processes=32 \
--num_env_steps=250000000 \
--ppo_epoch=5 \
--num_mini_batch=1 \
--entropy_coef=0.0 \
--value_loss_coef=0.5 \
--clip_param=0.2 \
--clip_value_loss=True \
--adv_entropy_coef=0.0 \
--max_grad_norm=0.5 \
--algo=ppo \
--ued_algo=generative_model \
--use_plr=False \
--level_replay_prob=0.0 \
--level_replay_rho=1.0 \
--level_replay_seed_buffer_size=5000 \
--level_replay_score_transform=rank \
--level_replay_temperature=0.1 \
--staleness_coef=0.3 \
--no_exploratory_grad_updates=False \
--use_editor=False \
--level_editor_prob=0 \
--level_editor_method=random \
--num_edits=0 \
--base_levels=batch \
--use_generator=True \
--generator_model_path=diffusion_human_feedback/log/minigrid_60_uniform/model300000.pt \
--use_guidance=True \
--regret_metric=cvar-mean \
--regret_guidance_weight=5.0 \
--num_tutor=1 \
--tutor_update_iteration=5 \
--tutor_buffer_size=1600 \
--tutor_batch_size=128 \
--use_categorical_tutor=True \
--num_bins_tutor=100 \
--logsumexp_temperature=0.1 \
--cvar_alpha=0.15 \
--return_add_noise_std=0.01 \
--log_interval=25 \
--screenshot_interval=100 \
--env_save_interval=10 \
--env_save_batch_size=32 \
--log_grad_norm=False \
--handle_timelimits=True \
--checkpoint_basis=student_grad_updates \
--archive_interval=5000 \
--test_env_names=MultiGrid-SixteenRooms-v0,MultiGrid-Maze-v0,MultiGrid-Labyrinth-v0 \
--log_dir=~/logs/minigrid_60/add \
--log_action_complexity=True \
--log_plr_buffer_stats=True \
--log_replay_complexity=True \
--reject_unsolvable_seeds=False \
--checkpoint=True

python -m train \
--xpid=seed_4 \
--env_name=MultiGrid-GoalLastVariableBlocksAdversarialEnv-v7 \
--use_gae=True \
--gamma=0.995 \
--gae_lambda=0.95 \
--seed=4 \
--num_control_points=12 \
--recurrent_arch=lstm \
--recurrent_agent=True \
--recurrent_adversary_env=False \
--recurrent_hidden_size=256 \
--use_global_critic=False \
--lr=0.0001 \
--num_steps=256 \
--num_processes=32 \
--num_env_steps=250000000 \
--ppo_epoch=5 \
--num_mini_batch=1 \
--entropy_coef=0.0 \
--value_loss_coef=0.5 \
--clip_param=0.2 \
--clip_value_loss=True \
--adv_entropy_coef=0.0 \
--max_grad_norm=0.5 \
--algo=ppo \
--ued_algo=generative_model \
--use_plr=False \
--level_replay_prob=0.0 \
--level_replay_rho=1.0 \
--level_replay_seed_buffer_size=5000 \
--level_replay_score_transform=rank \
--level_replay_temperature=0.1 \
--staleness_coef=0.3 \
--no_exploratory_grad_updates=False \
--use_editor=False \
--level_editor_prob=0 \
--level_editor_method=random \
--num_edits=0 \
--base_levels=batch \
--use_generator=True \
--generator_model_path=diffusion_human_feedback/log/minigrid_60_uniform/model300000.pt \
--use_guidance=True \
--regret_metric=cvar-mean \
--regret_guidance_weight=5.0 \
--num_tutor=1 \
--tutor_update_iteration=5 \
--tutor_buffer_size=1600 \
--tutor_batch_size=128 \
--use_categorical_tutor=True \
--num_bins_tutor=100 \
--logsumexp_temperature=0.1 \
--cvar_alpha=0.15 \
--return_add_noise_std=0.01 \
--log_interval=25 \
--screenshot_interval=100 \
--env_save_interval=10 \
--env_save_batch_size=32 \
--log_grad_norm=False \
--handle_timelimits=True \
--checkpoint_basis=student_grad_updates \
--archive_interval=5000 \
--test_env_names=MultiGrid-SixteenRooms-v0,MultiGrid-Maze-v0,MultiGrid-Labyrinth-v0 \
--log_dir=~/logs/minigrid_60/add \
--log_action_complexity=True \
--log_plr_buffer_stats=True \
--log_replay_complexity=True \
--reject_unsolvable_seeds=False \
--checkpoint=True

python -m train \
--xpid=seed_5 \
--env_name=MultiGrid-GoalLastVariableBlocksAdversarialEnv-v7 \
--use_gae=True \
--gamma=0.995 \
--gae_lambda=0.95 \
--seed=5 \
--num_control_points=12 \
--recurrent_arch=lstm \
--recurrent_agent=True \
--recurrent_adversary_env=False \
--recurrent_hidden_size=256 \
--use_global_critic=False \
--lr=0.0001 \
--num_steps=256 \
--num_processes=32 \
--num_env_steps=250000000 \
--ppo_epoch=5 \
--num_mini_batch=1 \
--entropy_coef=0.0 \
--value_loss_coef=0.5 \
--clip_param=0.2 \
--clip_value_loss=True \
--adv_entropy_coef=0.0 \
--max_grad_norm=0.5 \
--algo=ppo \
--ued_algo=generative_model \
--use_plr=False \
--level_replay_prob=0.0 \
--level_replay_rho=1.0 \
--level_replay_seed_buffer_size=5000 \
--level_replay_score_transform=rank \
--level_replay_temperature=0.1 \
--staleness_coef=0.3 \
--no_exploratory_grad_updates=False \
--use_editor=False \
--level_editor_prob=0 \
--level_editor_method=random \
--num_edits=0 \
--base_levels=batch \
--use_generator=True \
--generator_model_path=diffusion_human_feedback/log/minigrid_60_uniform/model300000.pt \
--use_guidance=True \
--regret_metric=cvar-mean \
--regret_guidance_weight=5.0 \
--num_tutor=1 \
--tutor_update_iteration=5 \
--tutor_buffer_size=1600 \
--tutor_batch_size=128 \
--use_categorical_tutor=True \
--num_bins_tutor=100 \
--logsumexp_temperature=0.1 \
--cvar_alpha=0.15 \
--return_add_noise_std=0.01 \
--log_interval=25 \
--screenshot_interval=100 \
--env_save_interval=10 \
--env_save_batch_size=32 \
--log_grad_norm=False \
--handle_timelimits=True \
--checkpoint_basis=student_grad_updates \
--archive_interval=5000 \
--test_env_names=MultiGrid-SixteenRooms-v0,MultiGrid-Maze-v0,MultiGrid-Labyrinth-v0 \
--log_dir=~/logs/minigrid_60/add \
--log_action_complexity=True \
--log_plr_buffer_stats=True \
--log_replay_complexity=True \
--reject_unsolvable_seeds=False \
--checkpoint=True