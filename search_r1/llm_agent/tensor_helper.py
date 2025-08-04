import torch
from typing import Dict, Tuple, List
from dataclasses import dataclass

@dataclass
class TensorConfig:
    pad_token_id: int
    max_prompt_length: int
    max_obs_length: int
    max_start_length: int

class TensorHelper:
    def __init__(self, config):
        self.config = config
    
    # 根据attention_mask，算出来当前batch里最大有效长度，把左侧无效的pad裁掉，节省空间
    def cut_to_effective_len(self, tensor_dict: Dict[str, torch.Tensor], 
                        keys: List[str], cut_left: bool = True) -> Dict[str, torch.Tensor]:
        effective_len = tensor_dict['attention_mask'].sum(dim=1).max()   # 取最长的数据长度
        result = tensor_dict.copy()

        for key in keys:
            if cut_left:
                result[key] = tensor_dict[key][:, -effective_len:]
            else:
                result[key] = tensor_dict[key][:, :effective_len]
        return result

    # 根据pad_token_id创建mask
    def create_attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        return torch.where(input_ids != self.config.pad_token_id, 1, 0)

    # 根据attention_mask创建position_ids
    def create_position_ids(self, attention_mask: torch.Tensor) -> torch.Tensor:
        # [[0, 1, 1, 1], [0, 0, 1, 1]] 求累计和 -> [[0, 1, 2, 3], [0, 0, 1, 2]]
        # 减1 -> [[0, 0, 1, 2], [0, 0, 0, 1]]  取mask -> [[0, 0, 1, 2], [0, 0, 0, 1]]
        return (torch.cumsum(attention_mask, dim=1) - 1) * attention_mask

    # 把pad调整在一起放在最左或最右边，[0 0 0 8 7 0 0 0 0 3 0 5 9 7 3] -> [0 0 0 0 0 0 0 0 8 7 3 5 9 7 3]
    def convert_pad_structure(self, tensor: torch.Tensor, pad_to_left: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        # [0 0 0 8 7 0 0 0 0 3 0 5 9 7 3] -> [0 0 0 1 1 0 0 0 0 1 0 1 1 1 1]
        mask = (tensor != self.config.pad_token_id) if pad_to_left else (tensor == self.config.pad_token_id)
        # 对mask排序，返回排序后的下标索引值 [0 0 0 1 1 0 0 0 0 1 0 1 1 1 1] -> [0 1 2 5 6 7 8 10  3 4 9 11 12 13 14]
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        # 最后按索引取出原始tensor的数值，[0 0 0 8 7 0 0 0 0 3 0 5 9 7 3] -> [0 0 0 0 0 0 0 0 8 7 3 5 9 7 3]
        return tensor.gather(dim=1, index=sorted_indices), sorted_indices

    # 把prompt、response、观测结果拼接起来，并且调整里边的所有pad在最左边
    def concatenate_with_padding(self, tensors: List[torch.Tensor],
                                 pad_to_left: bool = True) -> torch.Tensor:
        concatenated = torch.cat(tensors, dim=1)
        padded_tensor, _ = self.convert_pad_structure(concatenated, pad_to_left)
        return padded_tensor

    # 根据active_mask，把样本重新填充为batchsize大小（部分生成完了会变少）
    def _example_level_pad(self, responses: torch.Tensor, 
                    responses_str: List[str], 
                    active_mask: torch.Tensor) -> Tuple[torch.Tensor, List[str]]:
        
        assert active_mask.sum() == responses.shape[0]
        batch_size = active_mask.shape[0]
        seq_len = responses.shape[0]

        # responses是只有active为1的回答，把这些回答放入对应active_mask的位置上
        padded_responses = torch.full(
            (batch_size, seq_len), self.config.pad_token_id,
            dtype=responses.dtype, device=responses.device
        )
        padded_responses[active_mask] = responses

        # responses_str也是只有active为1的回答，把这些回答放入对应active_mask的位置上
        padded_responses_str = [""] * batch_size
        # 因为类型是List[str]，List不支持索引，无法这样处理：padded_responses_str[active_mask] = responses_str
        s = 0
        for i, is_active in enumerate(active_mask):
            if is_active:
                padded_responses_str[i] = responses_str[s]
                s += 1
        
        return padded_responses, padded_responses_str