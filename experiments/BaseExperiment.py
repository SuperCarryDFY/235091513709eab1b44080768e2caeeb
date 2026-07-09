from abc import ABC, abstractmethod
import math
import torch 
import tree
import os
from tqdm import tqdm
from models import AminoAcidTokenPredictionModel, StructureTokenPredictionModel
from utils import experiments_utils as eu
from accelerate import Accelerator, InitProcessGroupKwargs, data_loader, \
                    init_empty_weights, load_checkpoint_and_dispatch, load_checkpoint_in_model, dispatch_model
import torch.distributed as dist
from datetime import timedelta
from accelerate.utils import set_seed, infer_auto_device_map, get_balanced_memory
from deepspeed.runtime.utils import see_memory_usage
from utils import logger, trackers
from utils.metrics import TimePerStep
import time
import warnings
 
warnings.filterwarnings("ignore")
process_group_kwargs = InitProcessGroupKwargs(backend="nccl", timeout=timedelta(seconds=7200))  # 1.5 hours

class BaseExperiment(ABC):
    def __init__(self, cfg) -> None:
        self.accelerator = Accelerator(log_with=None, 
            gradient_accumulation_steps=cfg.training.get("accumulate_grad_batches", 1), 
            kwargs_handlers=[process_group_kwargs])
        set_seed(cfg.training.seed, device_specific=True)
        # init logging
        self._log = logger.MyLogger(output_dir=cfg.io['outdir'])
        self._log.info('Starting experiments')
        self._log.info(self.accelerator.state)
        self._log.info(f"outdir path: {cfg.io['outdir']}")
        self.cfg = cfg
        self.device = self.accelerator.device
        self.is_main_process = self.accelerator.is_main_process
        self.zero_stage = 0 if self.accelerator.state.deepspeed_plugin is None else self.accelerator.state.deepspeed_plugin.zero_stage
        if cfg.istraining:
            self.configure_training()
        else:
            self.configure_sampling()

    @abstractmethod
    def _sample_dataloader(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _optimizer(self) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def _dataloader(self) -> None:
        raise NotImplementedError        

    @abstractmethod
    def set_objective_and_metrics(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def set_tokenizer(self) -> None:
        raise NotImplementedError

    def configure_sampling(self) -> None:
        self.cfg.model["use_cache"] = True  # set use_cache to True for sampling
        
        # empty init and load checkpoints
        ## Big model only
        # with init_empty_weights():
        self.model = eval(self.cfg.model["module"])(self.cfg.model, self._log)
        if os.path.exists(self.cfg.ckpt_path):
            self.load_state_from_checkpoints(self.cfg.ckpt_path, map_location="cpu")
        elif self.cfg.ckpt_path.endswith("step=0/pytorch_model.bin"):
            print("CKPT step=0: No checkpoint, using random initialization")
        else:
            raise FileNotFoundError(f"Checkpoint {self.cfg.ckpt_path} not found")
        self.model = self.model.to(torch.bfloat16)
        self.model = self.model.to(self.device)
        # self.model = self.accelerator.prepare(self.model)
        
        # configure text tokenizer and sample dataloader.
        self.set_tokenizer()
        self.sample_dataloader = self._sample_dataloader()
        # self.model = self.accelerator.prepare(self.model)
        self.sample_dataloader = data_loader.prepare_data_loader(self.sample_dataloader, even_batches=False)
        self.model.eval()


    def configure_training(self) -> None:

        self.max_epochs = self.cfg.training["max_epochs"]
        self.trained_epochs, self.trained_steps = 0, 0
        self.enable_top_k_steps = self.cfg.training["enable_top_k_steps"]
        self.acumulate_grad_batches = self.cfg.training["accumulate_grad_batches"]

        # configure tracker
        self.accelerator.log_with = trackers.init_trackers(self.cfg)
        self.accelerator.init_trackers(project_name=self.cfg.io['project_name'], config=eu.args_to_dict(self.cfg))
        self.model = eval(self.cfg.model["module"])(self.cfg.model, self._log)
        self.optimizer, self.scheduler = self._optimizer()
        self.train_loader, self.val_loader = self._dataloader()
        
        # Define log steps
        self.val_steps = self.cfg.training["val_every_n_steps"]
        self.log_every_n_steps = self.cfg.training["log_every_n_steps"]
        self.save_every_n_steps = self.cfg.training["save_every_n_steps"]
        self.max_steps = self.cfg.training["max_steps"]
        self.steps_per_epoch = math.ceil(len(self.train_loader) / self.accelerator.num_processes)
        self.skip_first_batches = False
        self.save_states_suffix = ""
        if self.cfg.continue_training and self.cfg.not_load_optim:
            ## load weight only model before accelerate preparation. 
            if not os.path.exists(os.path.join(self.cfg.continue_training, "pytorch_model.bin")):
                raise Exception(f"Checkpoint {self.cfg.continue_training} does not exist. Please gather the paramter of the model before training.")
            self._log.info('loading checkpoint (weight only) from {}'.format(self.cfg.continue_training))
            self.load_state_from_checkpoints(os.path.join(self.cfg.continue_training, "pytorch_model.bin"))
            
            res = torch.load(f"{self.cfg.continue_training}/training_state.pth")
            self.trained_steps = res["trained_steps"]
            self.trained_epochs = res["trained_epochs"]
            # skip batches
            self._log.info(f"Skipping {self.trained_epochs} epoches and {self.trained_steps} batches...")
            # add datetime to the output dir
            self.save_states_suffix = f"_date-{eu.get_date()}"
       
        # To moniter trained time costs
        self.time_per_step = TimePerStep(accumulate_grad_batches=self.acumulate_grad_batches).to(self.device)
        if self.cfg.model["supports_gradient_checkpointing"]:
            for child_model in self.model.children():
                if hasattr(child_model, 'gradient_checkpointing_enable'):
                    child_model.gradient_checkpointing_enable()

        # configure object and set metrics
        self.set_objective_and_metrics()
        # count the number of parameters
        eu.count_parameters(self)
        # accelerate prepraration
        self.model, self.optimizer, self.scheduler, self.train_loader = self.accelerator.prepare(
            self.model, self.optimizer, self.scheduler, self.train_loader)
        if isinstance(self.val_loader, list):
            self.val_loader = [self.accelerator.prepare(val_loader) for val_loader in self.val_loader]
        else:
            self.val_loader = self.accelerator.prepare(self.val_loader)
        
        if self.cfg.continue_training and not self.cfg.not_load_optim:
            self._log.info('loading checkpoint from {}'.format(self.cfg.continue_training))
            self.skipped_dataloader = self.load_states()
            # add datetime to the output dir
            self.save_states_suffix = f"_date-{eu.get_date()}"

    def update_fn(self, idx, batch):
        """Updates the state using some data and returns metrics."""
        self.optimizer.zero_grad()
        res = self.loss_fn(idx, batch)
        self.accelerator.backward(res["loss"])
        self.optimizer.step()
        self.scheduler.step()
        return res

    def loss_fn(self, idx, batch):
        out = self.model(batch)        
        res = self.objective(batch, out)
        if torch.isnan(res["loss"]):
            raise Exception(f'train loss NaN encountered')
        return res


    def train_epoch(self, train_loader) -> bool:
        self.model.train()
        # with tqdm(total=len(train_loader), disable=not self.is_main_process) as t:
        #     t.set_description(f"Training on epoch {self.trained_epochs}")
        for idx, batch in enumerate(train_loader):
            begin_time = time.time()
            with self.accelerator.accumulate(self.model):
                batch = tree.map_structure(
                    lambda x: x.to(self.device) if isinstance(x, torch.Tensor) else x, batch)
                res = self.update_fn(idx, batch)
            end_time = time.time()
        
            # update metrics
            float(getattr(self, f"train_{self.objective_name}_loss")(res["loss"]).cpu().detach())
            float(getattr(self, f"train_{self.objective_name}_accuracy")(res["logits"], res["labels"]).cpu().detach())
            self.time_per_step.update(end_time - begin_time)
            
            # Logging
            if idx % self.acumulate_grad_batches == 0 and idx != 0:
                self.trained_steps += 1
                log_lr = {}
                ## Logging learning rate
                for idx, param_groups in enumerate(self.optimizer.param_groups):
                    log_lr[f"optimizer/group_{idx}"] = param_groups["lr"]
                self.accelerator.log(log_lr, step=self.trained_steps)

                ## Logging train metrics
                if self.trained_steps % self.log_every_n_steps == 0:
                    loss = float(getattr(self, f"train_{self.objective_name}_loss").compute().cpu().detach())
                    getattr(self, f"train_{self.objective_name}_loss").reset()
                    acc = float(getattr(self, f"train_{self.objective_name}_accuracy").compute().cpu().detach())
                    getattr(self, f"train_{self.objective_name}_accuracy").reset()
                    time_cost = self.time_per_step.compute()
                    self.time_per_step.reset()
                    log_metrics = {f"{self.objective_name}/train/loss": float(loss), 
                                   f"{self.objective_name}/train/accuracy": float(acc),
                                   "time_per_step": float(time_cost), 
                                   "epoch": self.trained_epochs}

                    self._log.info_dic_step(log_metrics, step=self.trained_steps)
                    self.accelerator.log(log_metrics, step=self.trained_steps)

                
                ## Logging validation metrics
                if self.trained_steps % self.val_steps == 0:
                    self._log.info("Staring validation ...")
                
                    log_metrics = self.val_epoch_warpper(self.val_loader)
                    self._log.info_dic_step(log_metrics, step=self.trained_steps)
                    self.accelerator.log(log_metrics, step=self.trained_steps)
                    self.model.train()
                    
                
                if self.trained_steps % self.save_every_n_steps == 0:
                    self.save_states()
                    self.accelerator.wait_for_everyone()
                if self.trained_steps == self.max_steps:
                    return True # finish training
        return False # False means continue


    @torch.no_grad()
    def val_epoch(self, val_loader, val_name=""):
        self.model.eval()
        for idx, batch in tqdm(enumerate(val_loader), total=len(val_loader), desc=f"validating val{val_name}...", disable=True):
            batch = tree.map_structure(
                lambda x: x.to(self.device) if isinstance(x, torch.Tensor) else x, batch)
            res = self.loss_fn(idx, batch)
            # update metrics
            getattr(self, f"val_{self.objective_name}_loss")(res["loss"])
            getattr(self, f"val_{self.objective_name}_accuracy")(res["logits"], res["labels"])

        # log validation metrics
        loss = float(getattr(self, f"val_{self.objective_name}_loss").compute().cpu().detach())
        getattr(self, f"val_{self.objective_name}_loss").reset()
        acc = float(getattr(self, f"val_{self.objective_name}_accuracy").compute().cpu().detach())
        getattr(self, f"val_{self.objective_name}_accuracy").reset()
        log_metrics = {f"{self.objective_name}/val{val_name}/loss": float(loss), f"{self.objective_name}/val{val_name}/accuracy": float(acc), "epoch": self.trained_epochs}

        return log_metrics
        

    @torch.no_grad()
    def test_epoch(self, test_loader):
        self.model.eval()
        if getattr(self, f"test_{self.objective_name}_loss", None) is None:
            from utils.metrics import Accuracy, Scalar
            setattr(self, f"test_{self.objective_name}_loss", Accuracy().to(self.device))
            setattr(self, f"test_{self.objective_name}_accuracy", Scalar().to(self.device))
            
        for idx, batch in tqdm(enumerate(test_loader), total=len(test_loader), desc=f"testing...", disable=not self.is_main_process):
            batch = tree.map_structure(
                lambda x: x.to(self.device) if isinstance(x, torch.Tensor) else x, batch)
            res = self.loss_fn(idx, batch)
            # update metrics
            getattr(self, f"test_{self.objective_name}_loss")(res["loss"])
            getattr(self, f"test_{self.objective_name}_accuracy")(res["logits"], res["labels"])
        
        loss = float(getattr(self, f"test_{self.objective_name}_loss").compute().cpu().detach())
        getattr(self, f"test_{self.objective_name}_loss").reset()
        acc = float(getattr(self, f"test_{self.objective_name}_accuracy").compute().cpu().detach())
        getattr(self, f"test_{self.objective_name}_accuracy").reset()

        log_metrics = {f"{self.objective_name}/test/loss": float(loss), f"{self.objective_name}/test/accuracy": float(acc), "epoch": self.trained_epochs}

        return log_metrics
        
    def val_epoch_warpper(self, val_loader_list, val_name_list=None):
        log_metrics = {}
        if not isinstance(val_loader_list, list):
            val_loader_list = [val_loader_list]    
        if val_name_list is None:
            val_name_list = [f"_{idx}" for idx in range(len(val_loader_list))]
            val_name_list[0] = "" # overwrite the first one
        for val_loader, val_name in zip(val_loader_list, val_name_list):
            log_metrics.update(self.val_epoch(val_loader, val_name))
        return log_metrics

    def start_testing(self) -> None:
        self.get_test_loader()
        self._log.info('Starting testing...')
        log_metrics = self.test_epoch(self.test_loader)
        self._log.info_dic_step(log_metrics, step=self.trained_steps)
        # self.accelerator.log(log_metrics, step=int(self.trained_steps / self.acumulate_grad_batches))
        self._log.info('Done')

    def start_training(self) -> None:
        self._log.info('Starting training...')
        # Do validation to log the performance at the begenning
        if self.cfg.training["val_before_train"]:
            log_metrics = self.val_epoch_warpper(self.val_loader)
            self._log.info_dic_step(log_metrics, step=self.trained_steps)
            self.accelerator.log(log_metrics, step=int(self.trained_steps / self.acumulate_grad_batches))
        
        # save step 0 checkpoint
        self.save_states()
        isfinish= False or self.trained_epochs == self.max_epochs
        while not isfinish:
            if getattr(self, "skipped_dataloader", None):
                cur_train_loader = self.skipped_dataloader
                self.skipped_dataloader = None
            else:
                cur_train_loader = self.train_loader
            isfinish = self.train_epoch(
                cur_train_loader, 
            )
            self.trained_epochs += 1
            isfinish = isfinish or self.trained_epochs == self.max_epochs
        
        self.accelerator.end_training()
        self._log.info('Done')
    

    def load_states(self, _path=None, weight_only=False):
        if _path is None:
            _path = self.cfg.continue_training
        self.accelerator.load_state(_path)
        if weight_only:
            return
        res = torch.load(f"{_path}/training_state.pth")
        self.trained_steps = res["trained_steps"]
        self.trained_epochs = res["trained_epochs"]
        # skip batches
        self._log.info(f"Skipping {self.trained_epochs} epoches and {self.trained_steps} batches...")
        skipped_steps = (self.trained_steps * self.acumulate_grad_batches) % self.steps_per_epoch
        skipped_dataloader = self.accelerator.skip_first_batches(self.train_loader, skipped_steps)
        return skipped_dataloader

    def save_states(self):
        output_dir = os.path.join(self.cfg.io["outdir"], 'IntervalCheckpoints')
        self._log.info(f"Saving to {output_dir}")
        output_path = os.path.join(output_dir, f"step={self.trained_steps}{self.save_states_suffix}")
        self.accelerator.save_state(output_dir=output_path, safe_serialization=False)
        save_dict = {
            "trained_steps": self.trained_steps,
            "trained_epochs": self.trained_epochs,
        }
        if self.accelerator.is_main_process:
            torch.save(save_dict, output_path + "/training_state.pth")
        self.accelerator.wait_for_everyone()