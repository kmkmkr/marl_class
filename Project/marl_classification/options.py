# -*- coding: utf-8 -*-
from typing import List, NamedTuple

MainOptions = NamedTuple(
    "MainOptions",
    [
        ("step", int),
        ("run_id", str),
        ("cuda", bool),
        ("nb_agent", int),
        ("seed", int),
    ],
)

TrainOptions = NamedTuple(
    "TrainOptions",
    [
        ("hidden_size_belief", int),
        ("hidden_size_linear_belief", int),
        ("hidden_size_linear_action", int),
        ("hidden_size_msg", int),
        ("hidden_size_state", int),
        ("hidden_size_action", int),
        ("dim", int),
        ("window_size", int),
        ("img_size", int),
        ("nb_class", int),
        ("action", List[List[int]]),
        ("nb_epoch", int),
        ("learning_rate", float),
        ("batch_size", int),
        ("resources_dir", str),
        ("output_dir", str),
        ("frozen_modules", List[str]),
        ("ft_extr_str", str),
        ("gamma", float),
        ("use_error_reward", bool),
        ("use_attn_reward", bool),
        ("vit_repo_or_dir", str),
        ("vit_name", str),
        ("attn_reward_path", str),
        
    ],
)

EvalOptions = NamedTuple(
    "EvalOptions",
    [
        ("img_size", int),
        ("state_dict_path", str),
        ("batch_size", int),
        ("json_path", str),
        ("dataset_path", str),
        ("output_dir", str),
    ],
)

InferOptions = NamedTuple(
    "InferOptions",
    [
        ("state_dict_path", str),
        ("json_path", str),
        ("images_path", List[str]),
        ("output_dir", str),
        ("class_to_idx", str),
    ],
)
