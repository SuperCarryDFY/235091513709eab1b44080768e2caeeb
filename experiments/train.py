import sys
sys.path.append('.')
from utils import config_utils
from experiments import _exp
from omegaconf import OmegaConf
import torch
import os
import argparse
from datetime import datetime
def time_format():
    return f'{datetime.now()}|>'


def main(args):	
	exp = _exp[args.training.experiment](args)
	exp.start_training()


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', type=str, help='Path for configuration file', required=True)
	parser.add_argument("--continue_training", type=str, default=None)
	parser.add_argument("--not_load_optim", action="store_true")
	parser.add_argument('--debug', action='store_true')

	args = config_utils._parse_args_and_yaml(parser)
	# Set the additional deterministic args
	args.io['run_name'] = os.path.basename(args.config).split('.')[0]
	args.GPU["num_nodes"] = int(os.environ.get("NUM_NODES", 1))
	args.GPU["num_gpus"] = torch.cuda.device_count() * args.GPU["num_nodes"]
	args.training["accumulate_grad_batches"] = args.training["pre_define_batch_size"] // (args.GPU["num_gpus"] * args.training["per_gpu_batchsize"])
	args.training["max_steps"] = args.training["pre_defined_max_steps"] // args.training["pre_define_batch_size"]
	args.training["log_every_n_steps"] = args.training["pre_defined_log_every_n_steps"] // args.training["pre_define_batch_size"]
	args.training["val_every_n_steps"] = args.training["pre_defined_val_every_n_steps"] // args.training["pre_define_batch_size"]
	args.training["save_every_n_steps"] = args.training["pre_defined_save_every_n_steps"] // args.training["pre_define_batch_size"]
	args.training["enable_top_k_steps"] = args.training["pre_defined_enable_top_k_steps"] // args.training["pre_define_batch_size"]
	args.training["num_workers"] = args.training["num_workers_per_gpu"] * args.GPU["num_gpus"]
	
	# configure output dir
	args.istraining = True
	if args.continue_training:
		args.io["outdir"] = '/'.join(args.continue_training.split('/')[:-2])
	else:
		args.io["outdir"] = os.path.join(args.io["outdir"], args.io['run_name'])
		os.makedirs(args.io["outdir"], exist_ok=True)
	if args.GPU["num_gpus"]==1 or os.environ.get("LOCAL_RANK", "0") == "0":
		# write args to file
		log_path = os.path.join(args.io["outdir"], f'args.yaml')
		OmegaConf.save(args, f=log_path)
		
	main(args)
