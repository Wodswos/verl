set -x

python3 -m verl.trainer.main_ppo --config-path=./config --config-name='ppo_megatron_trainer_llama_3.2_1b'\
    data.train_files=/root/autodl-fs/gsm8k/train.parquet \
    data.val_files=/root/autodl-fs/gsm8k/test.parquet \
    data.train_batch_size=32 \
    data.val_batch_size=32 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    actor_rollout_ref.model.path=/root/autodl-fs/Llama-3.2-1B-Instruct \
    actor_rollout_ref.actor.optim.lr=2e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size=8 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=8 \
    critic.optim.lr=2e-5 \
    critic.model.path=/root/autodl-fs/Llama-3.2-1B-Instruct \
    critic.model.enable_gradient_checkpointing=False \
    critic.ppo_micro_batch_size=8 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='verl_megatron_gsm8k_examples' \
    trainer.experiment_name='llama_2_7b_function_rm' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.total_epochs=15 $@