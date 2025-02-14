from dataclasses import dataclass, field
from typing import ClassVar
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

@dataclass
class fsdp_config:
    mixed_precision: bool=False
    use_fp16: bool=False
    seed: int=42
    fsdp_activation_checkpointing: bool=True
    limit_all_gathers: bool=True
    sharding_strategy: ShardingStrategy = ShardingStrategy.HYBRID_SHARD #FULL_SHARD #HYBRID_SHARD, SHARD_GRAD_OP
    checkpoint_type: StateDictType = StateDictType.SHARDED_STATE_DICT #FULL_STATE_DICT # alternatively can use SHARDED_STATE_DICT to avoid OOMs
    save_optimizer: bool=True
    verbose: bool=True
    checkpoint_folder: str="/data/checkpoints"
    model_save_name: str="model"
    optimizer_name: str="optim"
    dist_checkpoint_root_folder: str="/data/dist_checkpoints"
    dist_checkpoint_folder: str="distchkpt"
    save_using_num_threads: int=1
    model_name: str="mymodel"
    
    
    
