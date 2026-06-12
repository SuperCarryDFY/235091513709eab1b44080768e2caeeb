import yaml
import argparse
from omegaconf import OmegaConf
from utils.path_utils import load_env

def read_yaml(path):
    file = open(path, 'r', encoding='utf-8')
    string = file.read()
    dict = yaml.safe_load(string)
    return dict

def _fuse_dict(dic1, dic2):
    '''
    if conflict, use the latter
    '''
    for key in dic1.keys():
        if key in dic2.keys():
            if isinstance(dic1[key], dict):
                dic1[key].update(dic2[key])
            else:
                dic1[key] = dic2[key]
    return dic1

def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")



def check_dict(dic):
    for k, v in dic.items():
        # check none
        if v == "None":
            dic[k] = None
            continue
        # check float
        v = convert_str_to_number(v)
        if isinstance(v, float) and v == int(v):
            v = int(v)
        dic[k] = v

def convert_str_to_number(v):
    try:
        return float(v)
    except:
        return v


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if isinstance(v, dict):
            check_dict(v)
        if v is None or v=="None":
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)

def merge_two_dicts(dict1, dict2):
    for k, v in dict2.items():
        if isinstance(v, dict):
            dict1[k].update(v)
        else:
            dict1[k] = v
    return dict1


def _parse_args_and_yaml(parser):
    load_env()
    args = parser.parse_args()
    args_dict = args.__dict__
    configs_path = args_dict["config"]
    base_configs_path = "/".join(configs_path.split('/')[:-1]) + "/base.yaml"
    args_base = OmegaConf.load(base_configs_path)
    args_ = OmegaConf.load(configs_path)
    args = OmegaConf.merge(args_base, args_) # agrs_ should overwrite args_base
    args.update(args_dict)
    OmegaConf.resolve(args)
    return args


# YAML should not override the argparser's content
def _parse_args_and_yaml_old(parser):
    given_configs, remaining = parser.parse_known_args()
    configs_path =  given_configs.config

    base_configs_path = "/".join(configs_path.split('/')[:-1]) + "/base.yaml"
    configs_dict = read_yaml(configs_path)
    base_configs_dict = read_yaml(base_configs_path)
    configs_dict = merge_two_dicts(base_configs_dict, configs_dict)
    add_dict_to_argparser(parser, configs_dict)
    args = parser.parse_args()
    return args

# YAML should override the argparser's content
def _parse_args_and_yaml_in_sampling_old(parser):
    given_configs, remaining = parser.parse_known_args()
    configs_path =  given_configs.config

    base_configs_path = "/".join(configs_path.split('/')[:-1]) + "/base.yaml"
    configs_dict = read_yaml(configs_path)
    base_configs_dict = read_yaml(base_configs_path)
    sample_configs_dict = read_yaml(given_configs.sample_config)
    configs_dict = merge_two_dicts(base_configs_dict, configs_dict)
    add_dict_to_argparser(parser, configs_dict)
    add_dict_to_argparser(parser, sample_configs_dict)

    args = parser.parse_args()
    return args


def _parse_args_and_yaml_in_sampling(parser):

    load_env()
    args = parser.parse_args()
    args_dict = args.__dict__
    configs_path = args_dict["config"]
    sampple_configs_path = args_dict["sample_config"]
    base_configs_path = "/".join(configs_path.split('/')[:-1]) + "/base.yaml"

    configs_dict = OmegaConf.load(configs_path)
    base_configs_dict = OmegaConf.load(base_configs_path)
    sample_configs_dict = OmegaConf.load(sampple_configs_path)

    args = OmegaConf.merge(base_configs_dict, configs_dict, sample_configs_dict)
    args.update(args_dict)
    OmegaConf.resolve(args)
    return args
