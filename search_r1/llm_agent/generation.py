import torch
import re
from collections import defaultdict
import os
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from .tensor_helper import TensorHelper, TensorConfig
from verl import DataProto
from verl.utils.tracking import Tracking
import shutil
import requests

@dataclass
class GenerationConfig:
    max_turns: int
    max_start_length: int          # 最大初始prompt的长度
    max_prompt_length: int         # 模型最大prompt长度（经过多轮对话prompt比初始有所增加）
    max_response_length: int
    max_obs_length: int            # 最大观测长度，即为检索返回结果长度
    num_gpus: int
    no_think_rl: bool=False
    search_url: str=None
    topk: int = 3

class LLMGenerationManager:
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: GenerationConfig,
        is_validation: bool=False
    ):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation

        self.tensor_fn = TensorHelper(
            TensorConfig(
                pad_token_id=tokenizer.pad_token_id,
                max_prompt_length=config.max_prompt_length,
                max_obs_length=config.max_obs_length,
                max_start_length=config.max_start_length
            )
        )
    
    # 后处理rollout生成的response，只提取有用的信息（去除think等无用信息）
    def _postprocess_responses(self, responses: torch.Tensor) -> torch.Tensor:
        """Process responses to stop at search operation or answer operation."""
        # 把response_ids解码转化为文本
        responses_str = self.tokenizer.batch_decode(
            responses, 
            skip_special_tokens=True
        )

        # 进行文本拼接，把response里真正有用的文本信息拼接起来

        # responses_str = [resp.split('</search>')[0] + '</search>'
        #          if '</search>' in resp 
        #          else resp.split('</answer>')[0] + '</answer>'
        #          if '</answer>' in resp 
        #          else resp
        #          for resp in responses_str]
        
        
        # <think>\n思考内容\n</think>\n<code>\n代码内容\n</code>\n<observation>\n执行结果\n</observation>\n<answer>\n回答内容\n</answer> --> <think>\n思考内容\n</think>\n<code>\n代码内容\n</code>
        responses_str = [resp.split('</code>')[0] + '</code>'
                 if '</code>' in resp 
                 else resp.split('</answer>')[0] + '</answer>'
                 if '</answer>' in resp 
                 else resp
                 for resp in responses_str]
        
        if self.config.no_think_rl:
            raise ValueError('stop')
            # if no_think_rl is enabled, only keep action in the str
            actions, _ = self.env.postprocess_predictions(responses_str)
            responses_str=[f"<answer>{envs[idx].ACTION_LOOKUP[action]}</answer>" for idx, action in enumerate(actions)]
            print("RESPONSES:", responses_str)
        
        # 最后把response_ids编码还原
        responses = self._batch_tokenize(responses_str)
        return responses, responses_str


    # 把未生成完毕的样本数量补充，确保输入数据的batchsize能够被GPU数量整除，生成序列后，再移除补充的样本
    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """
            Wrapper for generation that handles multi-GPU padding requirements.
            if num_gpus <= 1, return self.actor_rollout_wg.generate_sequences(active_batch)
            if active_batch size is not divisible by num_gpus, pad with first sequence
            then remove padding from output
        """
        # GPU数量为1或者batchsize能够整除GPU数量，则直接返回generate_sequences(active_batch)
        num_gpus = self.config.num_gpus
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)
            
        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus
        
        for key in active_batch.batch.keys():
            active_batch.batch[key] = active_batch.batch[key].long()
        if remainder == 0:
            return self.actor_rollout_wg.generate_sequences(active_batch)
        
        # 增加padding样本，数量为padding_size
        padding_size = num_gpus - remainder
        padded_batch = {}
        for k, v in active_batch.batch.items():
            # Use first sequence as padding template
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)

        # 对padding后拼起来的样本进行rollout生成，生成完成后移除padding的样本
        padded_active_batch = DataProto.from_dict(padded_batch)
        for key in padded_active_batch.batch.keys():
            padded_active_batch.batch[key] = padded_active_batch.batch[key].long()
        # Generate with padded batch
        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)
        # Remove padding from output
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        
        # 处理meta_info信息
        # Handle meta_info if present
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta
            
        padded_output.batch = trimmed_batch
        return padded_output

    # 把执行完行动后观测到的信息编码成ids
    def _process_next_obs(self, next_obs: List[str]) -> torch.Tensor:
        next_obs_ids = self.tokenizer(
            next_obs,
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,  # Prevents adding special tokens
        )['input_ids']
        # 长度超过预设长度，就截断处理
        if next_obs_ids.shape[1] > self.config.max_obs_length:
            print(f"[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, {next_obs_ids.shape[1]} & {self.config.max_obs_length}")            
            next_obs_ids = next_obs_ids[:, :self.config.max_obs_length]
        return next_obs_ids

    # 更新当前rollout的数据，把response和观测信息next_obs_ids拼接到prompt里，用于下次输入
    def _update_rolling_state(self, rollings: DataProto, cur_responses: torch.Tensor,
                                  next_obs_ids: torch.Tensor) -> Dict:
        # 把prompt、response、观测结果拼接起来，并且调整里边的所有pad在最左边
        # [[0 0 0 8 7] [0 0 0 0 3] [0 5 9 7 3]] -> [0 0 0 8 7 0 0 0 0 3 0 5 9 7 3] -> [0 0 0 0 0 0 0 0 8 7 3 5 9 7 3]
        new_input_ids = self.tensor_fn.concatenate_with_padding(
            [rollings.batch['input_ids'], cur_responses, next_obs_ids]
        )

        # 创建 attention mask 和 position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # 重新计算最大的prompt长度，因为是把prompt、response、观测结果拼起来作为下一次的prompt
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)

        # 更新rollings数据，因为padding在左侧，需要左裁剪max_len
        new_rollings = DataProto.from_dict({
            'input_ids': new_input_ids[:, -max_len:],
            'position_ids': new_position_ids[:, -max_len:],
            'attention_mask': new_attention_mask[:, -max_len:]
        })
        new_rollings.meta_info.update(rollings.meta_info)
        return new_rollings

    # 把之前的response作为prompt，继续拼接新的response、观测信息info，并且把pad全部移到最右边
    def _info_masked_concatenate_with_padding(self,
                                              prompt: torch.Tensor,
                                              prompt_with_mask: torch.Tensor,
                                              response: torch.Tensor,
                                              info: torch.Tensor = None,
                                              pad_to_left: bool = True) -> torch.Tensor:
        pad_id = self.tokenizer.pad_token_id
        tensors = [prompt, response]
        tensors_with_mask = [prompt_with_mask, response]
        if info is not None:
            tensors.append(info)
            info_mask = torch.full(info.size(), pad_id, dtype=info.dtype, device=info.device)
            tensors_with_mask.append(info_mask)
        
        # concatenated与concatenated_with_info大小一致，只是concatenated_with_info在info部分全部设置了pad
        concatenated = torch.cat(tensors, dim=1)
        concatenated_with_info = torch.cat(tensors_with_mask, dim=1)
        # 这里pad_to_left=False，执行完argsort后pad全部移到右侧了，确保最开始的prompt不会被过长裁剪而丢掉重要信息
        mask = concatenated != pad_id if pad_to_left else concatenated == pad_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)

        # 最后拿到把pad全部移到最左边后的数据
        padded_tensor = concatenated.gather(dim=1, index=sorted_indices)
        padded_tensor_with_info = concatenated_with_info.gather(dim=1, index=sorted_indices)

        return padded_tensor, padded_tensor_with_info

    # 更新回答部分的内容即right_side，记录多轮response的信息，用于run_llm_loop的返回
    def _update_right_side(self, right_side: Dict,
                           cur_responses: torch.Tensor,
                           next_obs_ids: torch.Tensor = None) -> Dict:
        # right_side['response']是包含了历史对话的prompt和response
        # right_side['response_with_info_mask']是包含了历史对话的prompt和response，但在info部分全为pad
        if next_obs_ids != None:
            # 把之前的response作为prompt，继续拼接新的response、观测信息info，并且把pad全部移到最右边
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['response'],
                    right_side['response_with_info_mask'],
                    cur_responses,
                    info=next_obs_ids,
                    pad_to_left=False
                )
        else:
            # 把之前的response作为prompt，继续拼接新的response，并且把pad全部移到最右边
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    pad_to_left=False
                )
        
        # 前边pad已经移到最右边了，现在是左对齐，需要对最右边做截断
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        return {
            'responses': responses[:, :max_len],
            'responses_with_info_mask': responses_with_info_mask[:, :max_len]
        }

    # 最核心的部分：多轮循环完成任务
    def run_llm_loop(self, gen_batch, initial_input_ids: torch.Tensor) -> Tuple[Dict, Dict]:
        """Run main LLM generation loop."""

        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length]}
        original_right_side = {'responses': initial_input_ids[:, []], 'responses_with_info_mask': initial_input_ids[:, []]}

        # 记录当前输入是否rollout完成，如果完成active_mask对应位置更新为0
        active_mask = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.bool)
        # 记录统计信息，轮次等
        turns_stats = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        valid_action_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        valid_search_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)

        active_num_list = [active_mask.sum().item()]   # 记录剩余未rollout完成的数量
        rollings = gen_batch

        # 多轮循环进行rollout
        for step in range(self.config.max_turns):
            # 如果一个batch里全部都rollout完成，退出循环
            if not active_mask.sum():
                break
            
            # 根据attention_mask，算出来当前batch里最大有效长度，把左侧无效的pad裁掉，节省空间
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            # 取出未生成完毕的样本。rollings.batch是个字典，items()方法是把字典键和值拆开
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })

            # 把未生成完毕的样本数量补充，确保输入数据的batchsize能够被GPU数量整除，生成序列后，再移除补充的样本
            # 内部核心执行：self.actor_rollout_wg.generate_sequences(active_batch)
            gen_output = self._generate_with_gpu_padding(rollings_active)
            meta_info = gen_output.meta_info

            # 后处理本次rollout生成的response，只提取有用的信息（去除think等无用信息）
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            # 根据active_mask，把样本重新填充为batchsize大小（部分生成完了会变少）
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)
            
            # 根据responses_str执行行动，返回行动的观测obs，并记录如下：
            # next_obs观测结果，dones是否完成，valid_action是否有效行动，do_search是否执行代码或搜索
            next_obs, dones, valid_action, is_search = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, active_mask
            )

            # 执行完一次行动之后，更新active_mask
            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask

            # 记录统计信息
            active_num_list.append(active_mask.sum().item())
            turns_stats[curr_active_mask] += 1
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)

            # 把执行完行动后观测到的信息编码成ids
            next_obs_ids = self._process_next_obs(next_obs)
            # 更新当前rollout的数据，把response和观测信息next_obs_ids拼接到prompt里，用于下次输入
            rollings = self._update_rolling_state(rollings, responses_ids, next_obs_ids)

            # 更新回答部分的内容即right_side，记录多轮response的信息，用于run_llm_loop的返回
            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
                next_obs_ids
            )
        
        # 如果经过了self.config.max_turns，还有没完成的rollout，则执行最后一次
        if active_mask.sum():
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })

            # 把未生成完毕的样本数量补充，生成序列后，再移除补充的样本
            gen_output = self._generate_with_gpu_padding(rollings_active)
            meta_info = gen_output.meta_info

            # 后处理本次rollout生成的response，只提取有用的信息（去除think等无用信息）
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            # 根据active_mask，把样本重新填充为batchsize大小
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)
            
            # 根据responses_str执行行动，do_search为False，并记录如下：
            next_obs, dones, valid_action, is_search = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, active_mask, do_search=False
            )

            # 执行完一次行动之后，更新active_mask
            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask

            # 记录统计信息
            active_num_list.append(active_mask.sum().item())
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)

            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
                next_obs_ids
            )
        
        # 汇总最终的统计信息
        meta_info['turns_stats'] = turns_stats.tolist()
        meta_info['active_mask'] = active_mask.tolist()
        meta_info['valid_action_stats'] = valid_action_stats.tolist()
        meta_info['valid_search_stats'] = valid_search_stats.tolist()

        print("ACTIVE_TRAJ_NUM: ", active_num_list)
        return self._compose_final_output(original_left_side, original_right_side, meta_info)

    # 根据left_side与right_side，重新处理最终返回的信息
    def _compose_final_output(self, left_side: Dict,
                            right_side: Dict,
                            meta_info: Dict) -> Tuple[Dict, Dict]:
        # 初始化final_output
        final_output = right_side.copy()
        final_output['prompts'] = left_side['input_ids']
        final_output['input_ids'] = torch.cat([
            left_side['input_ids'],
            right_side['responses']
        ], dim=1)

        # 创建input_ids的mask
        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses'])
        ], dim=1)
        final_output['info_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses_with_info_mask'])
        ], dim=1)

        final_output['position_ids'] = self.tensor_fn.create_position_ids(
            final_output['attention_mask']
        )

        final_output = DataProto.from_dict(final_output)
        final_output.meta_info.update(meta_info)

        return final_output

    # 根据模型的response_str，执行具体的行动：如执行代码，搜索等，返回观测结果
    def execute_predictions(self, predictions: List[str], pad_token: str, active_mask=None, do_search=True) -> List[str]:
        # 对responses_str提取行动并分类，返回此次的action类型，以及内容(如代码或需要搜索的工具)
        cur_actions, contents = self.postprocess_predictions(predictions)
        next_obs, dones, valid_action, is_search = [], [], [], []

        # 提取出需要执行行动的内容
        search_queries = [content for action, content in zip(cur_actions, contents) if action == 'search']
        # 把需要行动的内容执行行动
        if do_search:
            search_results = self.batch_search(search_queries)
            assert len(search_results) == sum([1 for action in cur_actions if action == 'search'])
        else:
            search_queries = [''] * sum([1 for action in cur_actions if action == 'search'])
        
        # 记录此批次执行行动的过程
        for i, (action, active) in enumerate(zip(cur_actions, active_mask)):
            if not active:
                next_obs.append('')
                dones.append(1)
                valid_action.append(0)
                is_search(0)
            else:
                if action == 'answer':
                    next_obs.append('')
                    dones.append(1)
                    valid_action.append(1)
                    is_search(0)
                elif action == 'search':
                    next_obs.append(f'\n\n<observation>{search_results.pop(0).strip()}</observation>\n\n')
                    dones.append(0)
                    valid_action.append(1)
                    is_search(1)
                else:
                    next_obs.append(f'\nMy previous action is invalid. \
If I want to run code, I should put the code between <code> and </code>. \
If I want to give the final answer, I should put the answer between <answer> and </answer>. Let me try again.\n')
                    dones.append(0)
                    valid_action.append(0)
                    is_search.append(0)

        assert len(search_results) == 0         # 确保行动结果的队列为空，全部弹出
        return next_obs, dones, valid_action, is_search


    # 对responses_str提取行动并分类，返回此次的action类型，以及内容(如代码或需要搜索的工具)
    def postprocess_predictions(self, predictions: List[Any]) -> Tuple[List[str], List[bool]]:
        actions = []
        contents = []
        for prediction in predictions:
            if isinstance(prediction, str):  # 对应llm的输出
                # 这里以代码智能体模型为例
                code_blocks = self.extract_code(prediction)
                answer_block = self.extract_answer(prediction)
                # 优先判断是否有代码生成
                if len(code_blocks) > 0:
                    actions.append('search')
                    contents.append(prediction)
                elif len(answer_block) > 0:
                    actions.append('answer')
                    contents.append(prediction)
                else:
                    contents.append('')
                    actions.append(None)

        return actions, contents
    
    # 批量执行行动的内容，例如运行代码或进行浏览器搜索，这里以在api上执行代码为例
    def batch_search(self, queries: List[str] = None) -> List[str]:
        code_execution_result_list = []
        for query in queries:
            stdout, stderr = self.exec_code(query)
            code_execution_result_list.append(f'Code output: {stdout}\nErrors: {stderr}')
        return code_execution_result_list

    def extract_code(self, text: str)-> str:
        code_block_pattern = re.compile(r'<code>(.*?)</code>', re.DOTALL)
        # Find all matches in the text
        code_blocks = code_block_pattern.findall(text)
        # If no code blocks are found, try to find indented code blocks
        if not code_blocks:
            return []
        return [block.strip() for block in code_blocks]
    
    def extract_answer(self, text: str)-> str:
        answer_block_pattern = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
        # Find all matches in the text
        answer_blocks = answer_block_pattern.findall(text)
        # If no code blocks are found, try to find indented code blocks
        if not answer_blocks:
            return []
        return [block.strip() for block in answer_blocks]

    # 在远程平台运行代码的逻辑
    def exec_code(self, prediction: str) -> Tuple[List[str], List[str]]:
        import requests
        code = self.extract_code(prediction)
        if not code:
            return '', ''
        
        code = code[0]

        url = "http://xxx.xxx.xxx.xxx:xxxx/runcode"   # 部署好的代码运行服务
        headers = {
            'Content-Type': 'application/json'
        }
        data = {
            'code': code,
            'language': 'python'
        }

        response = requests.post(url, json=data, headers=headers)
        stdout = response.json()['run_result']['stdout']
        stderr = response.json()['run_result']['stderr']
        # print(stdout, stderr)
        return stdout[:1000], stderr[:1000]