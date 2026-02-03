import torch
import os
from torch.nn import functional as F
import numpy as np
from PIL import Image
from safetensors import safe_open
# from multiprocessing import set_start_method
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoModelForImageTextToText, AutoProcessor
import torch.multiprocessing as mp
from transformers.cache_utils import DynamicCache


def top_k_logits(logits, k):
    """ 保留概率最大的k个值，其他的设为-inf以屏蔽 """
    if k <= 0:
        return logits
    else:
        values, _ = torch.topk(logits, k)
        min_values = values[..., -1, None]
        return torch.where(logits < min_values, torch.full_like(logits, float('-inf')), logits)


def top_p_logits(logits, p):
    """ nucleus sampling: 保留累积概率大于p的前几个token，其他设为-inf """
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    # 获得大于p的第一个index并截断
    sorted_mask = cumulative_probs > p
    # 保证至少保留一个token
    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
    sorted_mask[..., 0] = False
    mask_indices = torch.scatter(torch.full_like(logits, False, dtype=torch.bool),
                                 -1, sorted_indices, sorted_mask)
    logits = logits.masked_fill(mask_indices, float('-inf'))
    return logits


def sample_with_temperature_topk_topp(logits, temperature=1.0, top_k=0, top_p=1.0):
    orig_shape = logits.shape[:-1]    # [batch, block]
    vocab_size = logits.shape[-1]

    logits = logits.reshape(-1, vocab_size)  # [batch*block, vocab]

    if temperature != 1.0:
        logits = logits / temperature
    if top_k > 0:
        logits = top_k_logits(logits, top_k)
    if top_p < 1.0:
        logits = top_p_logits(logits, top_p)
    probs = F.softmax(logits, dim=-1)  # shape: [batch*block, vocab]
    assert probs.dim() == 2
    token = torch.multinomial(probs, num_samples=1) # [batch*block, 1]
    token_prob = torch.gather(probs, -1, token)     # [batch*block, 1]

    return token.view(*orig_shape), token_prob.view(*orig_shape)


def get_num_transfer_tokens(block_length, steps):
    base = block_length // steps
    remainder = block_length % steps
    num_transfer_tokens = torch.zeros(steps, dtype=torch.int64) + base
    num_transfer_tokens[:remainder] += 1
    return num_transfer_tokens


@torch.no_grad()
def block_diffusion_generate(
    model,
    inputs,
    mask_id,
    gen_length=128,
    block_length=8,
    denoising_steps=8, 
    temperature=0.,
    top_k=0,
    top_p=1.0,
    remasking_strategy='low_confidence',
    stopping_criteria_idx=None):
    model.eval()
    input_ids = inputs['input_ids']
    prompt_length = input_ids.shape[1]
    past_key_values = DynamicCache()

    num_blocks = (prompt_length + gen_length + block_length - 1) // block_length
    total_length = num_blocks * block_length

    block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=model.device))
    block_diffusion_attention_mask = block_mask.repeat_interleave(block_length, dim=0)\
                                               .repeat_interleave(block_length, dim=1).unsqueeze(0)
    position_ids = torch.arange(total_length, device=model.device).unsqueeze(0)

    x = torch.full((1, total_length), mask_id, dtype=torch.long, device=model.device)
    x[:, :prompt_length] = input_ids
    prefill_blocks = prompt_length // block_length
    prefill_length = prefill_blocks * block_length

    if prefill_length > 0:
        cur_x = x[:, :prefill_length]
        cur_attn_mask = block_diffusion_attention_mask[:, :prefill_length, :prefill_length]
        cur_position_ids = position_ids[:, :prefill_length]
        model(input_ids=cur_x, pixel_values=inputs['pixel_values'], image_sizes=inputs['image_sizes'], attention_mask=cur_attn_mask, position_ids=cur_position_ids, past_key_values=past_key_values, use_cache=True, store_kv=True)

    num_transfer_tokens = get_num_transfer_tokens(block_length, denoising_steps)

    prob_distributions = {}  # 存prob的记录
    decoded_timesteps = {}  # 存解码步骤记录
    print(num_blocks)
    for num_block in range(prefill_blocks, num_blocks):
        cur_x = x[:, num_block*block_length:(num_block+1)*block_length].clone()
        cur_attn_mask = block_diffusion_attention_mask[
            :, num_block*block_length:(num_block+1)*block_length, :(num_block+1)*block_length
        ]
        cur_position_ids = position_ids[:, num_block*block_length:(num_block+1)*block_length]
        prob_distributions["block_{}".format(num_block)] = []
        for step in range(denoising_steps + 1):
            # 统计一个 step 的时间
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            start_time.record()
            mask_index = (cur_x == mask_id)
            if mask_index.sum() == 0:
                model(cur_x, attention_mask=cur_attn_mask, position_ids=cur_position_ids, past_key_values=past_key_values, use_cache=True, store_kv=True)
                break

            logits = model(cur_x, attention_mask=cur_attn_mask, position_ids=cur_position_ids, past_key_values=past_key_values, use_cache=True, store_kv=False).logits
            # p = F.softmax(logits, dim=-1)

            # logits_noise = add_gumbel_noise(logits, temperature)
            # x0 = torch.argmax(logits_noise, -1)
            # x0_p = torch.squeeze(torch.gather(p, dim=-1, index=x0.unsqueeze(-1)),-1)
            
            x0, x0_p = sample_with_temperature_topk_topp(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )

            # 顺序解码过程
            # print(remasking_strategy)
            if remasking_strategy == 'sequential':
                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(cur_x.shape[0]):
                    if mask_index[j].any():
                        # 找到第一个mask位置
                        first_mask_index = mask_index[j].nonzero(as_tuple=True)[0].min().item()
                        transfer_index[j, first_mask_index:first_mask_index + num_transfer_tokens[step]] = True
                    else:
                        raise ValueError("No mask tokens found in the current block.")

            elif remasking_strategy == 'low_confidence_static':
                # 静态解码过程
                confidence = torch.where(mask_index, x0_p, -np.inf)
                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(confidence.shape[0]):
                    _, idx = torch.topk(confidence[j], num_transfer_tokens[step])
                    transfer_index[j, idx] = True

            elif remasking_strategy == 'low_confidence_dynamic':
                # 动态解码过程
                confidence_threshold = 0.85  # 设置你的置信度阈值
                confidence = torch.where(mask_index, x0_p, -np.inf)
                transfer_index = torch.zeros_like(x0, dtype=torch.bool)

                for j in range(confidence.shape[0]):
                    # 找到满足置信度阈值的token
                    high_conf_mask = confidence[j] > confidence_threshold
                    num_high_confidence = high_conf_mask.sum()

                    if num_high_confidence >= num_transfer_tokens[step]:
                        # 如果高置信度数量已经超过或等于num_transfer_tokens，直接使用这些token
                        transfer_index[j] = high_conf_mask
                    else:
                        # 当高置信度tokens数量不够时，直接简单地从整体confidence中选取topk即可
                        _, idx = torch.topk(confidence[j], num_transfer_tokens[step])
                        transfer_index[j, idx] = True
            else:
                raise ValueError(f"Unknown remasking strategy: {remasking_strategy}")

            # 记录首次被解码step
            for j, pos in zip(*torch.nonzero(transfer_index, as_tuple=True)):
                global_pos = num_block * block_length + pos.item()
                token_key = (num_block, global_pos)
                if token_key not in decoded_timesteps:
                    decoded_timesteps[token_key] = step

            cur_x[transfer_index] = x0[transfer_index]

            # 存prob分布
            # prob_distributions["block_{}".format(num_block)].append({
            #     'step': step,
            #     'probs': p,
            #     'output_ids': cur_x.clone().cpu().numpy(),
            # })
            # 记录当前step的时间
            end_time.record()
            torch.cuda.synchronize()  # 等待事件完成
            elapsed_time = start_time.elapsed_time(end_time)  # 获取时间差，单位为毫秒
            print(f"Block {num_block}, Step {step}, Time taken: {elapsed_time:.2f} ms")


        x[:, num_block*block_length:(num_block+1)*block_length] = cur_x
        if stopping_criteria_idx is not None and any(stop_idx in x[:,prompt_length:] for stop_idx in stopping_criteria_idx):
            break

    return {
        'output_ids': x,
        'prob_distributions': prob_distributions,
        'token_decoded_timesteps': decoded_timesteps
    }


if __name__ == "__main__":

    import torch

    model_dir = "/mnt/shared-storage-user/chengshuang/projects/mdllm/llama_factory_sdar/inference/sdar_atage2_1"
    model_dir = "/mnt/shared-storage-user/chengshuang/projects/mdllm/llama_factory_sdar/inference/sdar_v"
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="cuda:0"
    )
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)

    # origin_prompt = [
    #     dict(role="user", content="Given the function $f(x) = \\frac{4x^2 - 4x + 4}{x^2 + 2x + 4}$, where $x \\in \\mathbb{R}$, determine its minimum value.\nPlease reason step by step, and put your final answer within \\boxed{}.\n"),
    #     # dict(role="user", content="If the domain of the function $\\log x^2$ is $x < a$ or $x > b$, for some $a$ and $b$, find $a + b$.\nPlease reason step by step, and put your final answer within \\boxed{}.\n")
    #     # dict(role="user", content="Find the sum of all integer bases $b>9$ for which $17_{b}$ is a divisor of $97_{b}$.\nRemember to put your final answer within \\boxed{}.\n"),
    #     # dict(role="user", content="Find the number of ordered pairs $(x,y)$, where both $x$ and $y$ are integers between $-100$ and $100$, inclusive, such that $12x^{2}-xy-6y^{2}=0$.\nRemember to put your final answer within \\boxed{}.\n"),
    # ]
    conversation = [
        {
        "role": "user",
        "content": [
            {"type": "text", "text": "What are these? Please answer using English"},
            {"type": "image"},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    image_file = "./assert/000000039769.jpg"
    raw_image = Image.open(image_file)
    inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(0, torch.float16)

    # 顺序解码
    denoising_steps = 4  # 设置去噪步数
    block_length = 4  # 设置块长度
    gen_length = 2048  # 设置生成长度

    remasking_strategy = 'low_confidence_dynamic'
    print(f"generation args: denoising_steps={denoising_steps}, block_length={block_length}, remasking_strategy={remasking_strategy}")
    results = block_diffusion_generate(
        model,
        inputs=inputs,
        mask_id=151669,
        gen_length=gen_length,
        block_length=denoising_steps,
        denoising_steps=block_length,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        remasking_strategy=remasking_strategy,
        stopping_criteria_idx=[151645, 151643]
    )

    output_ids = results['output_ids'][0]
    output_text = processor.decode(output_ids, skip_special_tokens=False)
    cleaned_text = output_text.replace('<|MASK|>', '')
    print(cleaned_text)
