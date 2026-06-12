import sys
sys.path.append('.')
from utils import config_utils
from experiments import _exp
import yaml
from omegaconf import OmegaConf
import os
import argparse
from datetime import datetime
import torch
from accelerate.utils import set_seed

def main(args):	
	exp = _exp[args.training.experiment](args)
	exp.start_sampling()
	

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	# parser.add_argument('--outdir', type=str, help='output path', default='output')
	parser.add_argument('--config', type=str, help='Path for configuration file', required=True)
	parser.add_argument('--ckpt_path', type=str, help='Path for configuration file', required=True)
	parser.add_argument('--sample_config', type=str, help='Path for configuration file', required=True)
	args = config_utils._parse_args_and_yaml_in_sampling(parser)
	
	# Set the additional deterministic args
	args.GPU["num_nodes"] = int(os.environ.get("NUM_NODES", 1))
	args.GPU["num_gpus"] = torch.cuda.device_count() * args.GPU["num_nodes"]
	args.training["num_workers"] = args.training["num_workers_per_gpu"] * args.GPU["num_gpus"]

	# configure output dir
	args.istraining = False
	sample_name= os.path.basename(args.sample_config).split('.')[0]
	args.io["outdir"] = os.path.dirname(args.ckpt_path) + f"/sample_{sample_name}"
	os.makedirs(args.io["outdir"], exist_ok=True)
	
	if args.GPU["num_gpus"]==1 or os.environ.get("LOCAL_RANK", "0") == "0":
		# write args to file
		log_path = os.path.join(args.io["outdir"], f'args.yaml')
		OmegaConf.save(args, f=log_path)

	main(args)


