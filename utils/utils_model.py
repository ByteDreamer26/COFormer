import json
import torch
import json
import argparse
from model import UniCO, UniCO_CRITIC
from model.llama import TrajLlama
from model.transformer_xl import TransformerXL
from thop import profile

def load_model(config_path, ckpt_path=None, snapshot_path=None, n_position=None, is_critic=False):
    assert (ckpt_path is None) ^ (snapshot_path is None)
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
        args = argparse.Namespace(**config_dict)    
        args.n_position = n_position if n_position is not None else args.n_position
        if is_critic:
            unico = UniCO_CRITIC(args)
        else:
            unico = UniCO(args)
    
    if ckpt_path is not None:
        if is_critic:
            loaded_state = torch.load(ckpt_path, map_location="cpu")
            critic_state = unico.state_dict()
            filtered_state = {k: v for k, v in loaded_state.items() if k in critic_state}
            critic_state.update(filtered_state)
            unico.load_state_dict(critic_state, strict=False)
        else:
            unico.load_state_dict(torch.load(ckpt_path))
    else:
        snapshot_data = torch.load(snapshot_path, map_location="cpu")
        unico.load_state_dict(snapshot_data["critic_state"] if is_critic else snapshot_data["model_state"], strict=False)
        
    return args, unico

def check_MAC(args, dataloader_val, local_rank=0, world_size=1):
    def _get_dummy_data():
        rl_task_input, batch_data_info, _ = next(dataloader_val.__iter__())
        rl_task_input.to(device=local_rank)
        batch_dataset_name = [info[0] for info in batch_data_info]
        input_ids = rl_task_input.tensor_seq
        model_inputs = {
            "input_ids": input_ids.contiguous(),
            "position_ids": torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0),
            "use_cache": False, 
            "past_key_values": None,
        }    
        dummy_data = {'tasks_input': rl_task_input, 'compute_loss': True, 'batch_dataset_name': batch_dataset_name, 'model_inputs': model_inputs}
        return dummy_data
    
    def _get_dummy_model():
        if args.model == 'transformer_xl':
            dummy_model = TransformerXL(args).to(local_rank)
            dummy_model.same_length = False
        elif args.model == 'llama':
            dummy_model = TrajLlama(args).to(local_rank)
        else:
            raise NotImplementedError
        return dummy_model
    
    def _get_parameter_size_in_gb(param):
        return param.numel() * param.element_size() / 1024 / 1024 / 1024
    
    if local_rank == 0:
        # get params quantity by count directly 
        dummy_data = _get_dummy_data()
        dummy_model = _get_dummy_model()
        total_block_params = sum(p.numel() for n, p in dummy_model.named_parameters() if 'embedding' not in n and 'encoding' not in n and n != 'lm_head')
        total_size_gb = sum(_get_parameter_size_in_gb(p) for p in dummy_model.parameters())

        # get macs with thop lib 
        dummy_model = _get_dummy_model()
        macs, _ = profile(dummy_model, tuple(dummy_data.values()), verbose=False)
        model_macs_per_batch = macs * world_size
        
        print('-'*50 + ' Model Info ' + '-'*50) 
        print(f"Total block params:  \t{total_block_params}")
        print(f"Total parameter size:\t{total_size_gb:.2f} GB")
        print(f'MACs per batch data: \t{int(model_macs_per_batch /1e6)} M')

def check_model_size(model, local_rank=0, desc=None):
    if local_rank == 0:
        total_block_params = sum(p.numel() for n, p in model.named_parameters() if 'embedding' not in n and 'encoding' not in n and n != 'lm_head')
        total_size_gb = sum(p.numel() * p.element_size() / 1024 / 1024 / 1024 for p in model.parameters())

        desc = desc + ' ' if desc is not None else ''
        print(f"{desc}Total block params:  \t{total_block_params}")
        print(f"{desc}Total parameter size:\t{total_size_gb:.2f} GB")