# source code for "Ours" and "Ours+DEER" under greedy decoding senario

import warnings
warnings.filterwarnings("ignore") # Ignore all warnings

import os
import json
import time
import argparse
import sys
import torch
import torch.nn.functional as F
from vllm.outputs import CompletionOutput
from typing import Any, Dict, List
from nltk import ngrams
from collections import Counter

from transformers import AutoTokenizer
from tqdm import tqdm
from vllm import LLM, SamplingParams
import pdb

import math
import numpy as np
import random
from utils.utils import (
    append_jsonl,
    write_jsonl,
    read_jsonl,
)
import torch.multiprocessing as mp

QWEN_BUT_WAIT_TOKEN_IDS = [
    14190, 1988, 714, 13824, 3983, 3783, 11489, 8088
]
QWEN_ALTER_TOKEN_IDS = [
    38478, 41109, 75763, 92014
]
QWEN_HMM_TOKEN_IDS = [
    80022, 88190
]

LLAMA_BUT_WAIT_TOKEN_IDS = [
    14524, 14144, 2030, 719, 4071, 3868, 11748, 8248
]
LLAMA_ALTER_TOKEN_IDS = [
    39578, 42209, 76863, 93114
]
LLAMA_HMM_TOKEN_IDS =[
    81122, 89290
]

def set_seeds(seed=42):
    # Set Python built-in random seed
    random.seed(seed)
    # Set NumPy random seed
    np.random.seed(seed)
    # Set PyTorch CPU random seed
    torch.manual_seed(seed)
    # If using GPU (especially CUDA)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)      # Set seed for current GPU
        torch.cuda.manual_seed_all(seed)      # Also effective for multi-GPU
        # For better reproducibility, enable cudnn determinism mode
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # Optional: Set generator (for DataLoader with multi-threading)
    g = torch.Generator()
    g.manual_seed(seed)

def calculate_certrainty_score(logprobs_list) -> float:
    num_tokens = len(logprobs_list)
    start_index = 1 # pass {
    end_index = num_tokens
    if num_tokens < 1:
        return 0.0
    entropies = []
    for i in range(start_index, end_index):
        if logprobs_list[i]:
            probs = [np.exp(logprob.logprob).item() for logprob in logprobs_list[i].values()]
            entropy = -sum(p * math.log(p) for p in probs if p > 0)
            entropies.append(entropy)

    certainty_score = 1 - (np.mean(entropies) / np.log(len(logprobs_list[0]))) if entropies else 0.0
    return certainty_score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, )
    parser.add_argument('--dataset_dir', type=str, default="./data/")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--max-model-len", "--model-context-len", type=int, default=32768, dest="model_context_len") 
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument('--delta', type=float, default=1.0) # The answer certainty score threshold for suppression probability calculation.
    parser.add_argument('--max_generated_tokens', '--max-len', type=int, default=16384, dest="max_len") 
    parser.add_argument('--dataset', type=str, default='math') 
    parser.add_argument('--output_path', type=str, default='./outputs') 
    parser.add_argument('--think_ratio', type=float, default=0.6) 
    parser.add_argument('--batch_size', type=int, default=2000) 
    parser.add_argument('--temperature', type=float, default=0.6) 
    parser.add_argument('--repetition_penalty', type=float, default=1.0) 
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--trial_idx', type=int, default=0)
    parser.add_argument('--prob_check_max_tokens', type=int, default=20, help="Max tokens for probability check phase") 
    args = parser.parse_args()
    return args

def main(args):

    print("\nInitializing vLLM LLM engine...")
    available_gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')

    llm_engine = LLM(
        model=args.model_name_or_path,
        tensor_parallel_size=len(available_gpus),
        dtype=args.dtype,
        max_model_len=args.max_len+1024, # for safety
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True, 
        seed=args.trial_idx
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    sys_prompt = ['Please reason step by step, and put your final answer within \\boxed{}.'][0] 
    dataset_path = f'{args.dataset_dir}/{args.dataset}/test.jsonl'
    try:
        questions_json = read_jsonl(dataset_path)
        if not questions_json:
            print(f"Error: No questions loaded from {dataset_path}.")
            sys.exit(1)
        print(f"Successfully loaded dataset: {dataset_path}, total {len(questions_json)} questions")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        sys.exit(1)

    model_dir_name = os.path.basename(os.path.normpath(args.model_name_or_path))
    output_dir = f'{args.output_path}/{model_dir_name}/{args.dataset}'
    os.makedirs(output_dir, exist_ok=True)
    sample_prefix = "greedy_" if args.temperature < 1e-6 else ""
    # reppen1.05
    output_file = f'{output_dir}/{sample_prefix}sr{str(args.suppress_ratio)}_p{str(args.delta)}_ratio{str(args.think_ratio)}_len{str(args.max_len)}_temperature{str(args.temperature)}_rep{args.rep}_points{args.points}_reppen{args.repetition_penalty}_trial{args.trial_idx}.jsonl' # Update filename to include new parameter

    print(f"\nStarting processing, total questions: {len(questions_json)}")
    start_time = time.time()

    continue_str = "\n\n" 

    answer_prompt_str = "\n**Final Answer**\n\\boxed" # Prompt string to guide answer generation
    if 'gpqa' in args.dataset:
        answer_prompt_str = "\n**Final Answer**\nI believe the final answer, rather than the option, is \\boxed"

    # Get token IDs for stop conditions and strings to append
    last_token_ids = []
    for s in last_token_strs:
        ids = tokenizer.encode(s, add_special_tokens=False)
        if ids: last_token_ids.extend(ids)
    last_token_ids = list(set(last_token_ids)) # Remove duplicate IDs
    questions_state = {} # Dictionary to store processing state for each question
    last_token_strs = ["</think>"] # Strings marking end of thinking

    continue_ids = tokenizer.encode(continue_str, add_special_tokens=False)
    if not continue_ids:
          print(f"Warning: Unable to tokenize continue string '{continue_str}'. This may affect logic.")

    global QWEN_BUT_WAIT_TOKEN_IDS
    global QWEN_HMM_TOKEN_IDS
    global QWEN_ALTER_TOKEN_IDS
    global LLAMA_BUT_WAIT_TOKEN_IDS
    global LLAMA_HMM_TOKEN_IDS
    global LLAMA_ALTER_TOKEN_IDS
    if "Qwen" in args.model_name_or_path or "QwQ" in args.model_name_or_path:
        suppress_token_ids = QWEN_ALTER_TOKEN_IDS + QWEN_HMM_TOKEN_IDS + QWEN_BUT_WAIT_TOKEN_IDS 
    else :
        assert "Llama" in args.model_name_or_path
        suppress_token_ids = LLAMA_ALTER_TOKEN_IDS + LLAMA_HMM_TOKEN_IDS + LLAMA_BUT_WAIT_TOKEN_IDS 
    suppress_token_ids = list(set(suppress_token_ids)) 

    generation_stop_tokens = [continue_str] + last_token_strs + [tokenizer.eos_token] 
    answer_stop_tokens = [tokenizer.eos_token]
    think_limit_tokens = int(args.max_len * args.think_ratio)

    for i, question_data in enumerate(questions_json):
        
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": question_data['problem']}
        ]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if "Qwen" in args.model_name_or_path or "QwQ" in args.model_name_or_path:
            formatted_prompt += " Let's think step by step and output the final answer within \\boxed{}."

        questions_state[i] = {
            'question_data': question_data,
            'state': 'needs_thought_chunk', 
            'formatted_prompt': formatted_prompt, 
            'current_full_sequence': formatted_prompt, 
            'generated_thinking_history': "", 
            'generated_answer_history': "", 
            'certainty_score': 0.0, # Initialize confidence
            'too_long': 0,
            'high_prob': 0,
            'thinking_steps': 0,
            'output_dict': {}, 
            'error_message': None, 
            'question_index': i
        }

    active_questions_indices = list(questions_state.keys()) # List of currently processing question indices
    pbar = tqdm(total=len(questions_json), desc="Processing questions")

    print("\nRunning a simple test generation...")
    test_outputs = llm_engine.generate(["Hello, world!"], SamplingParams(max_tokens=10, temperature=args.temperature), use_tqdm=False)
    test_generated_text = test_outputs[0].outputs[0].text
    if not test_generated_text:
        print("Error: Test generation returned empty output. Please check model and tokenizer compatibility.")
        sys.exit(1)
    else:
        print("Test generation successful. Sample output:", test_generated_text)

    # Main processing loop: continue while there are active questions
    while active_questions_indices: # indices [0,1,2,...,n]
        batch_prompts = [] # Current batch prompts
        batch_sampling_params = [] # Current batch sampling parameters
        batch_request_info = [] # Store (question index, step type) for output processing

        current_batch_count = 0
        current_active_indices_for_batching = active_questions_indices[:]

        # Build current batch
        for q_idx in current_active_indices_for_batching:
            if current_batch_count >= args.batch_size:
                break

            state = questions_state[q_idx]
            if state['state'] in ['finished', 'error']:
                continue

            prompt_for_batch = None
            sampling_params_for_batch = None
            step_type = None # 'think', 'prob_check_gen', 'answer'

            try:
                current_full_sequence_tokens = tokenizer.encode(state['current_full_sequence'], add_special_tokens=False)
                current_full_sequence_len = len(current_full_sequence_tokens)
                current_generated_thinking_tokens = len(tokenizer.encode(state['generated_thinking_history'], add_special_tokens=False))
                initial_prompt_len = len(tokenizer.encode(state['formatted_prompt'], add_special_tokens=False))

                remaining_context_window = args.model_context_len - current_full_sequence_len
                if remaining_context_window <= 0:
                    state['state'] = 'error'
                    state['error_message'] = f"Exceeded model context window limit ({args.model_context_len}). Current sequence length: {current_full_sequence_len}"
                    state['output_dict'] = {'question': state['question_data']['problem'], 'generated_responses': [state['generated_thinking_history'] + state['generated_answer_history'] + "\n" + state['error_message']], 'gold_answer': state['question_data']['answer']}
                    print(f"\nQuestion {q_idx}: {state['error_message']}")
                    if q_idx in active_questions_indices:
                        active_questions_indices.remove(q_idx)
                        pbar.update(1)
                    continue 

                logits_processors_list = None
                current_certainty_score = state['certainty_score'] # Get the latest calculated certrainty score

                if current_certainty_score > args.delta:
                    suppression_probability = (current_certainty_score - args.delta) / (1.0 - args.delta)

                    def probabilistic_suppress_processor(input_ids: List[int], scores: torch.Tensor) -> torch.Tensor:
                        random_float = torch.rand(1, device=scores.device).item()

                        if random_float < suppression_probability:
                            ids_on_device = torch.tensor(suppress_token_ids, dtype=torch.long).to(scores.device)
                            scores.index_fill_(0, ids_on_device, -float('inf')) 
                        return scores
                    
                    logits_processors_list = [probabilistic_suppress_processor]
                else:
                    logits_processors_list = None # No suppression if certainty is very low

                # Initial state          
                if state['state'] == 'needs_thought_chunk':
                    max_new_tokens_for_thought = min(
                        think_limit_tokens - current_generated_thinking_tokens, 
                        remaining_context_window 
                    )
                    if max_new_tokens_for_thought <= 0:
                        print(f"\nQuestion {q_idx}: Reached thinking limit ({current_generated_thinking_tokens}/{think_limit_tokens}).")
                        state['too_long'] = 1
                        continue 
                        
                    prompt_for_batch = state['current_full_sequence']

                    sampling_params_for_batch = SamplingParams(
                        max_tokens=max_new_tokens_for_thought,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        stop=generation_stop_tokens,
                        repetition_penalty=args.repetition_penalty,
                        logits_processors=logits_processors_list 
                        )
                    step_type = 'think'

                # Check state
                elif state['state'] == 'needs_prob_check':
                    # Build prompt for probability check generation:
                    prompt_for_prob_check = state['current_full_sequence'] + answer_prompt_str
                    prob_check_prompt_len = len(tokenizer.encode(prompt_for_prob_check, add_special_tokens=False))
                    required_space_for_prob_check = prob_check_prompt_len + args.prob_check_max_tokens

                    if required_space_for_prob_check > args.model_context_len:
                        state['state'] = 'error'
                        state['error_message'] = f"Probability check generation prompt exceeds context window ({args.model_context_len}). Estimated length: {required_space_for_prob_check}"
                        state['output_dict'] = {'question': state['question_data']['problem'], 'generated_responses': [state['generated_thinking_history'] + state['generated_answer_history'] + "\n" + state['error_message']], 'gold_answer': state['question_data']['answer']}
                        print(f"\nQuestion {q_idx}: {state['error_message']}")
                        if q_idx in active_questions_indices:
                            active_questions_indices.remove(q_idx)
                            pbar.update(1)
                        continue
                    
                    prompt_for_batch = prompt_for_prob_check
                    sampling_params_for_batch = SamplingParams(
                        max_tokens=args.prob_check_max_tokens,
                        logprobs=5,
                        repetition_penalty=args.repetition_penalty
                    )
                    step_type = 'prob_check_gen' 

                elif state['state'] == 'needs_answer':
                    # Build final answer prompt
                    
                    if state['too_long'] == 1:
                        final_answer_prompt = state['formatted_prompt'] + state['generated_thinking_history'] + '\n</think>\n\n'
                        state['generated_thinking_history'] = state['generated_thinking_history'] + '\n</think>\n\n'
                        
                    else:
                        final_answer_prompt = state['formatted_prompt'] + state['generated_thinking_history'] + '\n</think>\n\n'#+ answer_prompt_str
                        state['generated_thinking_history'] = state['generated_thinking_history'] + '\n</think>\n\n'
                    
                    len_final_answer_prompt = len(tokenizer.encode(final_answer_prompt, add_special_tokens=False))
                    total_tokens_before_answer_prompt = current_generated_thinking_tokens
                    # Calculate remaining total budget
                    remaining_total_budget = args.max_len - total_tokens_before_answer_prompt
                    max_new_tokens_answer = min(
                       remaining_total_budget,
                       args.model_context_len - len_final_answer_prompt
                    )
                    
                    if max_new_tokens_answer <= 0:
                        
                        state['state'] = 'error'
                        state['output_dict'] = {'question': state['question_data']['problem'], 'generated_responses': [state['generated_thinking_history'] + "\nSkipped answer generation due to length limit."], 'gold_answer': state['question_data']['answer'], 'too_long': state['too_long'], 'thinking_steps': state['thinking_steps']}
                        print(f"\nQuestion {q_idx}: Skipped answer generation due to length limit.")
                        
                        if q_idx in active_questions_indices:
                            active_questions_indices.remove(q_idx)
                            pbar.update(1)
                        continue
                    else:
                        prompt_for_batch = final_answer_prompt
                        sampling_params_for_batch = SamplingParams(
                            max_tokens=max_new_tokens_answer,
                            temperature=args.temperature,
                            stop=answer_stop_tokens,
                            top_p=args.top_p,
                            logits_processors=logits_processors_list 
                           )
                        step_type = 'answer'

                else:
                    raise ValueError(f"Unsupported state: {state['state']}. Supported states are 'needs_thought_chunk', 'needs_prob_check', and 'needs_answer'.")

                if prompt_for_batch is None:
                    state['state'] = 'error'
                    state['error_message'] = f"Internal error: No prompt generated for state {state['state']}."
                    state['output_dict'] = {'question': state['question_data']['problem'], 'generated_responses': [state['generated_thinking_history'] + state['generated_answer_history'] + "\n" + state['error_message']], 'gold_answer': state['question_data']['answer']}
                    print(f"\nQuestion {q_idx}: {state['error_message']}")
                    if q_idx in active_questions_indices:
                        active_questions_indices.remove(q_idx)
                        pbar.update(1)
                    continue 
                
                batch_prompts.append(prompt_for_batch)
                batch_sampling_params.append(sampling_params_for_batch)
                batch_request_info.append((q_idx, step_type))
                current_batch_count += 1

            except Exception as e:
                state['state'] = 'error'
                state['error_message'] = f"Error preparing batch request for state '{state['state']}': {e}"
                state['output_dict'] = {'question': state['question_data']['problem'], 'generated_responses': [state['generated_thinking_history'] + state['generated_answer_history'] + "\n" + state['error_message']], 'gold_answer': state['question_data']['answer']}
                print(f"\nQuestion {q_idx}: {state['error_message']}")
                if q_idx in active_questions_indices:
                    active_questions_indices.remove(q_idx)
                    pbar.update(1)

        if not batch_prompts:
            all_stuck = True
            for q_idx in active_questions_indices:
                if questions_state[q_idx]['state'] not in ['finished', 'error']:
                    all_stuck = False
                    break
            if all_stuck:
                print("Warning: Batch generated no requests. All remaining questions are completed or in error state.")
                break
            else:
                print("Error: Active questions remain but no batch requests generated. Possible logic error.")
                for q_idx in list(active_questions_indices):
                    state = questions_state[q_idx]
                    if state['state'] not in ['finished', 'error']:
                        state['state'] = 'error'
                        state['error_message'] = "Processing aborted: Unable to generate request in batch loop."
                        state['output_dict'] = {'question': state['question_data']['problem'], 'generated_responses': [state['generated_thinking_history'] + state['generated_answer_history'] + "\n" + state['error_message']], 'gold_answer': state['question_data']['answer']}
                        print(f"\nQuestion {q_idx}: Marked as error due to processing abort.")
                        if q_idx in active_questions_indices: # Should be, but double-checking
                            active_questions_indices.remove(q_idx)
                            pbar.update(1)
                break

        batch_outputs = llm_engine.generate(batch_prompts, batch_sampling_params, use_tqdm=False)
    

        # --- Process batch outputs ---
        for i, output in enumerate(batch_outputs):
            q_idx, step_type = batch_request_info[i]
            state = questions_state[q_idx]

            if state['state'] in ['finished', 'error']:
                continue

            try:
                if not output.outputs: # skip, exception handling
                    # vLLM returned empty output, possibly due to prompt issues, length limits or other internal errors
                    error_msg = f"vLLM returned empty output for request {output.request_id} (question {q_idx}, step {step_type})."
                    if hasattr(output, 'error') and output.error:
                        error_msg += f" vLLM error: {output.error}"
                    current_full_sequence_len = len(tokenizer.encode(state['current_full_sequence'], add_special_tokens=False))
                    initial_prompt_len = len(tokenizer.encode(state['formatted_prompt'], add_special_tokens=False))
                    current_generated_thinking_tokens = len(tokenizer.encode(state['generated_thinking_history'], add_special_tokens=False))
 
                    if step_type == 'think':
                        max_new = min(think_limit_tokens - current_generated_thinking_tokens, args.model_context_len - current_full_sequence_len)
                        error_msg += f" State: think, attempting to generate {max_new} tokens."
                        if max_new <= 0: error_msg += " Note: Calculated max new tokens <= 0."
                        if args.model_context_len - current_full_sequence_len <= 0: error_msg += " Note: Already exceeded context window."
                        if think_limit_tokens - current_generated_thinking_tokens <= 0: error_msg += " Note: Reached thinking token limit."

                    elif step_type == 'prob_check_gen':
                        prob_check_prompt = state['current_full_sequence'] + answer_prompt_str
                        prob_check_prompt_len = len(tokenizer.encode(prob_check_prompt, add_special_tokens=False))
                        error_msg += f" State: prob_check_gen, prompt length: {prob_check_prompt_len}, attempting to generate {args.prob_check_max_tokens} tokens."
                        if prob_check_prompt_len + args.prob_check_max_tokens > args.model_context_len: error_msg += " Note: Prompt+generation exceeds context window limit."

                    elif step_type == 'answer':
                        final_answer_prompt = state['formatted_prompt'] + state['generated_thinking_history'] + answer_prompt_str
                        len_final_answer_prompt = len(tokenizer.encode(final_answer_prompt, add_special_tokens=False))
                        total_tokens_before_answer_prompt = initial_prompt_len + current_generated_thinking_tokens
                        max_new_answer = min(args.max_len - total_tokens_before_answer_prompt, args.model_context_len - len_final_answer_prompt)
                        error_msg += f" State: answer, prompt length: {len_final_answer_prompt}, attempting to generate {max_new_answer} tokens."
                        if max_new_answer <= 0: error_msg += " Note: Calculated max new tokens <= 0."
                        if args.model_context_len - len_final_answer_prompt <= 0: error_msg += " Note: Already exceeded context window."
                        if args.max_len - total_tokens_before_answer_prompt <= 0: error_msg += " Note: Reached total token limit."

                    raise ValueError(error_msg) 

                completion_output = output.outputs[0]
                generated_text = completion_output.text
                generated_ids = completion_output.token_ids
                last_token_id = generated_ids[-1]
                
                
                if step_type == 'think':
                    if last_token_id in last_token_ids:
                        state['state'] = 'needs_answer'
                        state['generated_thinking_history'] += generated_text
                        state['current_full_sequence'] = state['formatted_prompt'] + state['generated_thinking_history']
                    else: 
                        clean_generated_text = generated_text
                        for stop_tok in generation_stop_tokens:
                            if clean_generated_text.endswith(stop_tok):
                                clean_generated_text = clean_generated_text[:-len(stop_tok)]
                                break
                        state['generated_thinking_history'] += clean_generated_text + continue_str # " " + continue_str
                        state['current_full_sequence'] = state['formatted_prompt'] + state['generated_thinking_history']
                        state['thinking_steps'] += 1
                        state['state'] = 'needs_prob_check' # Proceed to probability check after a thinking step
                
                elif step_type == 'prob_check_gen':
                    certainty_score = calculate_certrainty_score(completion_output.logprobs)
                    state['certainty_score'] = certainty_score
                    current_generated_thinking_tokens = len(tokenizer.encode(state['generated_thinking_history'], add_special_tokens=False))
                    if current_generated_thinking_tokens >= think_limit_tokens:
                        state['state'] = 'needs_answer'
                    else:
                        state['state'] = 'needs_thought_chunk' 
                
                elif step_type == 'answer':
                    state['generated_answer_history'] += generated_text
                    state['state'] = 'finished'
                    state['output_dict'] = {
                        'question': state['question_data']['problem'],
                        'generated_responses': [state['generated_thinking_history'] + state['generated_answer_history']],
                        'gold_answer': state['question_data']['answer'],
                        'certainty_score': state['certainty_score'],
                        'too_long': state['too_long'],
                        'high_prob': state['high_prob'],
                        'thinking_steps': state['thinking_steps']
                    }
                    print(f"\nQuestion {q_idx}: Final answer generated. Marking as finished.")
                    if q_idx in active_questions_indices:
                        active_questions_indices.remove(q_idx)
                        pbar.update(1)
                
                else:
                    raise ValueError(f"Unsupported step type: {step_type}. Supported types are 'think', 'prob_check_gen', and 'answer'.")
                    
            except Exception as e:
                state['state'] = 'error'
                state['error_message'] = f"Error processing output for question {q_idx} (step {step_type}): {e}"
                state['output_dict'] = {'question': state['question_data']['problem'], 'generated_responses': [state['generated_thinking_history'] + state['generated_answer_history'] + "\n" + state['error_message']], 'gold_answer': state['question_data']['answer']}
                print(f"\nQuestion {q_idx}: {state['error_message']}")
                if q_idx in active_questions_indices:
                    active_questions_indices.remove(q_idx)
                    pbar.update(1)
    
    pbar.close()

    final_results = [state['output_dict'] for state in questions_state.values() if state['state'] in ['finished', 'error']]
    
    problem_to_index = {item['problem']: i for i, item in enumerate(questions_json)}
    final_results.sort(key=lambda x: problem_to_index.get(x['question'], len(questions_json)))

    print("\nAll questions processed, saving results...")
    try:
        write_jsonl(final_results, output_file)
    except Exception as e:
        print(f"Error saving results to {output_file}: {e}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Evaluation completed! Attempted to process {len(questions_json)} questions in total, successfully recorded {len(final_results)} results, took {elapsed_time:.2f} seconds")
    print(f"Results saved to: {output_file}")
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nProcessing complete. Total time: {total_time:.2f} seconds.")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    args = parse_args()
    args.model_context_len = args.max_len
    set_seeds(args.trial_idx)
    main(args)
