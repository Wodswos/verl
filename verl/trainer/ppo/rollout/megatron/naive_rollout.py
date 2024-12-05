from megatron.core import ModelParallelConfig
from megatron.core import mpu
import torch
import torch.nn as nn
from tensordict import TensorDict

from verl import DataProto
from ..base import BaseRollout


# TODO:
# 1. small demo about parallel forward in megatron
#   1.1 how initialize
#   1.2 how different part cooperate with each other and do forward?
#   1.3 will call eight time on LLMEngine in vLLM rollout?
# 2. Clarify Rollout 返回内容，Actor/Critic 如何基于这些返回内容进行训练。
# 3. Sampling.


class MegatronNaiveRollout(BaseRollout):
    def __init__(self, module: nn.ModuleList, config: ModelParallelConfig):
        super().__init__()
        self.config = config

        # nn.Modules list -> model ??
        self.module = module
    
    def generate(self):
        """Given prompts and input parameters, run inference and return:
        tokens: prompts plus the generated tokens.
        lengths: length of the prompt + generations. Note that we can
            discard tokens in the tokens tensor that are after the
            corresponding length.
        output_log_probs: log probs of the tokens.
        """

        # Make sure input params are avaialble to all ranks.
        values = [tokens_to_generate,
                return_output_log_probs,
                top_k_sampling, top_p_sampling, top_p_decay, top_p_bound,
                temperature, add_BOS, use_eod_token_for_early_termination,
                stop_on_double_eol,
                stop_on_eol,
                prevent_newline_after_colon,
                random_seed]
        values_float_tensor = broadcast_float_list(len(values), float_list=values)
        tokens_to_generate = int(values_float_tensor[0].item())
        return_output_log_probs = bool(values_float_tensor[1].item())
        top_k_sampling = int(values_float_tensor[2].item())
        top_p_sampling = values_float_tensor[3].item()
        top_p_decay = values_float_tensor[4].item()
        top_p_bound = values_float_tensor[5].item()
        temperature = values_float_tensor[6].item()
        add_BOS = bool(values_float_tensor[7].item())
        use_eod_token_for_early_termination = bool(values_float_tensor[8].item())
        stop_on_double_eol = bool(values_float_tensor[9].item())
        stop_on_eol = bool(values_float_tensor[10].item())
        prevent_newline_after_colon = bool(values_float_tensor[11].item())
        random_seed = int(values_float_tensor[12].item())

        if random_seed != -1:
            torch.random.manual_seed(random_seed)

        # Tokenize prompts and get the batch.
        # Note that these tensors are broadcaseted to all ranks.
        if torch.distributed.get_rank() == 0:
            assert prompt_ids is not None
        
        context_tokens_tensor, context_length_tensor = tokenize_prompts(
            prompts=prompts, tokens_to_generate=tokens_to_generate, add_BOS=add_BOS)

        if tokens_to_generate == 0:
            return score_and_return_on_first_stage(
                model, context_tokens_tensor, context_length_tensor)
        
        # Main inference function.
        # Note that the outputs are available on the first stage.
        return self.generate_tokens_probs_and_return_on_first_stage(
            model, context_tokens_tensor, context_length_tensor,
            return_output_log_probs=return_output_log_probs,
            top_k=top_k_sampling,
            top_p=top_p_sampling,
            top_p_decay=top_p_decay,
            top_p_bound=top_p_bound,
            temperature=temperature,
            use_eod_token_for_early_termination=use_eod_token_for_early_termination,
            stop_on_double_eol=stop_on_double_eol,
            stop_on_eol=stop_on_eol,
            prevent_newline_after_colon=prevent_newline_after_colon)

    def generate_tokens_probs_and_return_on_first_stage(
            self,
            prompt_ids: torch.Tensor, # [batch_size, ]
            max_response_length, # []
            return_output_log_probs=False,
            top_k=0,
            top_p=0.0,
            top_p_decay=0.0,
            top_p_bound=0.0,
            temperature=1.0,
            use_eod_token_for_early_termination=True,
            stop_on_double_eol=False,
            stop_on_eol=False,
            prevent_newline_after_colon=True
            ):

        batch_size = prompt_ids.
        min_prompt_length = lengths.min().item()
        max_sequence_length = tokens.size(1)

        # if max_sequence_length * batch_size > max_tokens_to_oom:
        #     raise ValueError("Too many tokens.  " + str(max_sequence_length*batch_size)+ " is greater than "+str(args.max_tokens_to_oom))

        # forward step.
        forward_step = ForwardStep(self.model, batch_size, max_sequence_length)

        # Added termination_id to support the case that we want to terminate the
        # generation once that id is generated.
        # if hasattr(args, 'eos_id'):
        #     termination_id = args.eos_id
        # else:
        #     termination_id = tokenizer.eod
        termination_id = self.tokenizer.eos_token

        # ===================
        # Pre-allocate memory
        # ===================

        # Log probability of the sequence (prompt + generated tokens).
        output_log_probs = None
        output_log_probs_size = (batch_size, max_sequence_length - 1)
        # Lengths of generated seuquence including including prompts.
        generated_sequence_lengths = None
        if mpu.is_pipeline_last_stage():
            if return_output_log_probs:
                output_log_probs = torch.empty(output_log_probs_size,
                                            dtype=torch.float32,
                                            device=torch.cuda.current_device())
            generated_sequence_lengths = torch.ones(
                    batch_size, dtype=torch.int64,
                    device=torch.cuda.current_device()) * max_sequence_length
        
        # Whether we have reached a termination id.
        is_generation_done = torch.zeros(batch_size, dtype=torch.uint8,
                                        device=torch.cuda.current_device())

        # =============
        # Run infernece
        # =============

        with torch.no_grad():
            # attention_mask, position_ids = _build_attention_mask_and_position_ids(
            #     tokens)
            prev_context_length = 0
            for context_length in range(min_prompt_length, max_sequence_length):

                # Pick the slice that we need to pass through the network.
                tokens2use = tokens[:, prev_context_length:context_length]
                positions2use = position_ids[:, prev_context_length:context_length]
                attention_mask2use = attention_mask[
                    ..., prev_context_length:context_length, :context_length]

                # logits will be meanigful only in the last pipeline stage.
                logits = forward_step(tokens2use, positions2use, attention_mask2use)

                if mpu.is_pipeline_last_stage():
                    if prevent_newline_after_colon:
                        logits[tokens2use[:, -1] == tokenizer.tokenize(':')[0], -1, tokenizer.tokenize('\n')[0]] = -1e10 # disable "\n" after ":"
                    # Always the last stage should have an output.
                    assert logits is not None

                    # Sample.
                    last_token_logits = logits[:, -1, :]
                    new_sample = sample(last_token_logits,
                                        top_k=top_k,
                                        top_p=top_p,
                                        temperature=temperature,
                                        vocab_size=tokenizer.vocab_size)
                    if top_p > 0.0 and top_p_decay > 0.0:
                        top_p = top_p * top_p_decay
                        if top_p_bound > 0.0:
                            top_p = max(top_p, top_p_bound)

                    # If a prompt length is smaller or equal th current context
                    # length, it means we have started generating tokens
                    started = lengths <= context_length
                    # Update the tokens.
                    tokens[started, context_length] = new_sample[started]

                    # Calculate the log probabilities.
                    if return_output_log_probs:
                        log_probs = F.log_softmax(logits, dim=2)
                        if return_output_log_probs:
                            # Pick the tokens that we need to get the log
                            # probabilities for. Note that next input token is
                            # the token which we selected in the current logits,
                            # so shift by 1.
                            indices = torch.unsqueeze(
                                tokens[
                                    :,
                                    (prev_context_length + 1):(context_length + 1)],
                                2)
                            output_log_probs[:,
                                            prev_context_length:context_length] = \
                                torch.gather(log_probs, 2, indices).squeeze(2)

                # Update the tokens on the first stage so the next input to
                # the network is correct.
                copy_from_last_to_first_pipeline_stage(
                    batch_size, 
                    torch.int64,
                    tokens[:, context_length]
                )

                # Update the context length for the next token generation.
                prev_context_length = context_length

                # Check if all the sequences have hit the termination_id.
                done = None
                if mpu.is_pipeline_last_stage():
                    # TODO(rprenger) These stopping methods are tokenizer dependent
                    # instead tokenization should be in the inference loop so stop sequences can be used
                    if stop_on_double_eol:
                        hit_double_eol = (new_sample == 628).byte() & started.byte()
                        hit_two_eols = (new_sample == 198).byte() & (tokens[:, context_length-1] == 198).byte() & started.byte()
                        done_token = hit_double_eol | hit_two_eols
                    elif stop_on_eol:
                        hit_double_eol = (new_sample == 628).byte() & started.byte()
                        hit_eol = (new_sample == 198).byte() & started.byte()
                        done_token = hit_double_eol | hit_eol
                    else: 
                        done_token = (new_sample == termination_id).byte() & \
                            started.byte()
                    
                    just_finished = (done_token & ~is_generation_done).bool()
                    generated_sequence_lengths[just_finished.view(-1)] = \
                        context_length + 1
                    is_generation_done = is_generation_done | done_token
                    done = torch.all(is_generation_done)
                done = broadcast_from_last_pipeline_stage(1, torch.uint8,
                                                        tensor=done)
                if use_eod_token_for_early_termination and done:
                    break
                
        # ===================================================
        # Update the length of based on max generated length.
        # ===================================================

        tokens = tokens[:, :(context_length + 1)]
        if mpu.is_pipeline_last_stage():
            if return_output_log_probs:
                output_log_probs = output_log_probs[:, :context_length]

        # ======================================
        # Broadcast to the first pipeline stage.
        # ======================================

        generated_sequence_lengths = broadcast_from_last_to_first_pipeline_stage(
            batch_size, torch.int64, generated_sequence_lengths)
        if return_output_log_probs:
            output_log_probs_size = (batch_size, context_length)
            output_log_probs = broadcast_from_last_to_first_pipeline_stage(
                output_log_probs_size, torch.float32, output_log_probs)

        return tokens, generated_sequence_lengths, output_log_probs, None

    def forward_backward_batch(self):
        pass

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        idx = prompts.batch['input_ids'].cuda()
        attention_mask = prompts.batch['attention_mask'].cuda()
        position_ids = prompts.batch['position_ids'].cuda()

        # from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
        # forward_backward_func = get_forward_backward_func()
        # from megatron.text_generation import generate

        output = self.generate(
            model=self.module,
            prompts=idx,
            # tokens_to_generate=self.config.
            tokens_to_generate=512,
            return_output_log_probs=True,
        )

        seq = torch.cat([idx, output])

        print(output)
        batch = TensorDict(
            {
                'prompts': idx,
                'responses': output,
                # 'input_ids': seq,
                'attention_mask': attention_mask,
                'position_ids': position_ids
            }
        )

        return DataProto(

        )



if __name__ == '__main__':

    # test case, start with ray
    import ray
    import torch
    from megatron.core import parallel_state

    @ray.remote
    def test_inference(rank, world_size):
        # init disitributed environment.
        # parallel_state.destroy_model_parallel()
        torch.distribute.init_process_group(world_size=world_size, rank=rank)
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=world_size,
            pipeline_model_parallel_size=1
        )

        # Define the model 
        # Use the basic GPTModel defined in Megatron
        # from megatron.core.models.gpt.gpt_model import GPTModel
        # from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec

        # Or just use the modeling in veRL
        from transformers import AutoConfig
        module_config = AutoConfig.from_pretrained()

        from megatron.core import ModelParallelConfig
        megatron_config = ModelParallelConfig(
            tensor_model_parallel_size=world_size,
            pipeline_model_parallel_size=1,
            virtual_pipeline_model_parallel_size=None,
            bf16=True,
            fp16=False,
        )

        # refer to megatron_worker._build_model_optimizer()
        from verl.utils.model import get_parallel_model_from_config
        parallel_model = get_parallel_model_from_config(
            config=module_config,
            megatron_config=megatron_config
        )

        rollout = MegatronNaiveRollout(parallel_model, megatron_config)

        result = rollout.generate_sequences()


    
    # model_path = '/'
    # world_size = 2
    # refs = []
    # for rank in range(world_size):
    #     refs.append(test_inference.remote(rank, world_size))
    # ray.get(refs)

