from accelerate.tracking import GeneralTracker, on_main_process
from typing import Optional, Union, List, Any
import os
import wandb
from accelerate.utils import listify
import time
import yaml
from utils.experiments_utils import on_main_process
from accelerate.logging import get_logger

import dataclasses
from collections import defaultdict
from pathlib import Path
from argparse import Namespace

logger = get_logger(__name__)
    

@on_main_process
def init_trackers(cfg):
    trackers = cfg.training.trackers
    res = []
    if trackers:
        for tracker in trackers:
            if tracker == "wandb":
                res.append(CustomWandBTracker(run_name=cfg.io['project_name'], name=cfg.io['run_name']))
            elif tracker == "tensorboard":
                res.append(CustomTensorBoardTracker(run_name="tblogs", logging_dir=cfg.io["outdir"]))
            else:
                raise ValueError(f"Tracker {tracker} not supported")
    else:
        res.append(EmptyTracker(run_name="EmptyTracker", name="text", logging_dir=cfg.io["outdir"]))
    return res



class EmptyTracker(GeneralTracker):
    name = "EmptyTraker"
    requires_logging_directory = True

    @on_main_process
    def __init__(self, run_name: str, logging_dir: Union[str, os.PathLike], **kwargs):
        super().__init__()
        # self.logging_dir = logging_dir
        # logger.info(f"Logging file path {logging_dir}/logging.txt")


    @property
    def tracker(self):
        return None

    @on_main_process
    def store_init_configuration(self, values: dict):
 
        return 


    @on_main_process
    def log(self, values: dict, step: Optional[int] = None, **kwargs):
        return
        _str = f"### step={step} ###\n"
        loggging_str = f"step = {step}\t"

        for k, v in values.items():
            _str += "### " + k + " : " + str(v) + " ###\n"
            loggging_str += f"{k} = {v}\t"
        print(_str)
        with open(f"{self.logging_dir}/logging.txt", "a") as f:
            f.write(loggging_str + "\n")
        return 


    @on_main_process
    def finish(self):
        return 


class CustomWandBTracker(GeneralTracker):
    """
    A `Tracker` class that supports `wandb`. Should be initialized at the start of your script.

    Args:
        run_name (`str`):
            The name of the experiment run.
        kwargs:
            Additional key word arguments passed along to the `wandb.init` method.
    """

    name = "wandb"
    requires_logging_directory = False
    main_process_only = False

    @on_main_process
    def __init__(self, run_name: str, name: str, **kwargs):
        super().__init__()
        self.run_name = run_name

        import wandb

        self.run = wandb.init(project=self.run_name, name=name, **kwargs)
        logger.debug(f"Initialized WandB project {self.run_name}")
        logger.debug(
            "Make sure to log any initial configurations with `self.store_init_configuration` before training!"
        )

    @property
    def tracker(self):
        return self.run

    @on_main_process
    def store_init_configuration(self, values: dict):
        """
        Logs `values` as hyperparameters for the run. Should be run at the beginning of your experiment.

        Args:
            values (Dictionary `str` to `bool`, `str`, `float` or `int`):
                Values to be stored as initial hyperparameters as key-value pairs. The values need to have type `bool`,
                `str`, `float`, `int`, or `None`.
        """
        import wandb

        # wandb.config.update(make_json_safe(values), allow_val_change=True)
        logger.debug("Stored initial configuration hyperparameters to WandB")

    @on_main_process
    def log(self, values: dict, step: Optional[int] = None, **kwargs):
        """
        Logs `values` to the current run.

        Args:
            values (Dictionary `str` to `str`, `float`, `int` or `dict` of `str` to `float`/`int`):
                Values to be logged as key-value pairs. The values need to have type `str`, `float`, `int` or `dict` of
                `str` to `float`/`int`.
            step (`int`, *optional*):
                The run step. If included, the log will be affiliated with this step.
            kwargs:
                Additional key word arguments passed along to the `wandb.log` method.
        """
        self.run.log(values, step=step, **kwargs)
        logger.debug("Successfully logged to WandB")

    @on_main_process
    def log_images(self, values: dict, step: Optional[int] = None, **kwargs):
        """
        Logs `images` to the current run.

        Args:
            values (Dictionary `str` to `List` of `np.ndarray` or `PIL.Image`):
                Values to be logged as key-value pairs. The values need to have type `List` of `np.ndarray` or
            step (`int`, *optional*):
                The run step. If included, the log will be affiliated with this step.
            kwargs:
                Additional key word arguments passed along to the `wandb.log` method.
        """
        import wandb

        for k, v in values.items():
            self.log({k: [wandb.Image(image) for image in v]}, step=step, **kwargs)
        logger.debug("Successfully logged images to WandB")

    @on_main_process
    def log_table(
        self,
        table_name: str,
        columns: List[str] = None,
        data: List[List[Any]] = None,
        dataframe: Any = None,
        step: Optional[int] = None,
        **kwargs,
    ):
        """
        Log a Table containing any object type (text, image, audio, video, molecule, html, etc). Can be defined either
        with `columns` and `data` or with `dataframe`.

        Args:
            table_name (`str`):
                The name to give to the logged table on the wandb workspace
            columns (list of `str`, *optional*):
                The name of the columns on the table
            data (List of List of Any data type, *optional*):
                The data to be logged in the table
            dataframe (Any data type, *optional*):
                The data to be logged in the table
            step (`int`, *optional*):
                The run step. If included, the log will be affiliated with this step.
        """
        import wandb

        values = {table_name: wandb.Table(columns=columns, data=data, dataframe=dataframe)}
        self.log(values, step=step, **kwargs)

    @on_main_process
    def finish(self):
        """
        Closes `wandb` writer
        """
        self.run.finish()
        logger.debug("WandB run closed")



class CustomTensorBoardTracker(GeneralTracker):
    """
    A `Tracker` class that supports `tensorboard`. Should be initialized at the start of your script.

    Args:
        run_name (`str`):
            The name of the experiment run
        logging_dir (`str`, `os.PathLike`):
            Location for TensorBoard logs to be stored.
        kwargs:
            Additional key word arguments passed along to the `tensorboard.SummaryWriter.__init__` method.
    """

    name = "tensorboard"
    requires_logging_directory = True

    @on_main_process
    def __init__(self, run_name: str, logging_dir: Union[str, os.PathLike], **kwargs):

        from torch.utils import tensorboard

        super().__init__()
        self.run_name = run_name
        self.logging_dir = os.path.join(logging_dir, run_name)
        self.writer = tensorboard.SummaryWriter(self.logging_dir, **kwargs)
        logger.debug(f"Initialized TensorBoard project {self.run_name} logging to {self.logging_dir}")
        logger.debug(
            "Make sure to log any initial configurations with `self.store_init_configuration` before training!"
        )

    @property
    def tracker(self):
        return self.writer

    @on_main_process
    def store_init_configuration(self, values: dict):
        """
        Logs `values` as hyperparameters for the run. Should be run at the beginning of your experiment. Stores the
        hyperparameters in a yaml file for future use.

        Args:
            values (Dictionary `str` to `bool`, `str`, `float` or `int`):
                Values to be stored as initial hyperparameters as key-value pairs. The values need to have type `bool`,
                `str`, `float`, `int`, or `None`.
        """
        return 
        self.writer.add_hparams(values, metric_dict={})
        self.writer.flush()
        project_run_name = time.time()
        dir_name = os.path.join(self.logging_dir, str(project_run_name))
        os.makedirs(dir_name, exist_ok=True)
        with open(os.path.join(dir_name, "hparams.yml"), "w") as outfile:
            try:
                yaml.dump(values, outfile)
            except yaml.representer.RepresenterError:
                logger.error("Serialization to store hyperparameters failed")
                raise
        logger.debug("Stored initial configuration hyperparameters to TensorBoard and hparams yaml file")

    @on_main_process
    def log(self, values: dict, step: Optional[int] = None, **kwargs):
        """
        Logs `values` to the current run.

        Args:
            values (Dictionary `str` to `str`, `float`, `int` or `dict` of `str` to `float`/`int`):
                Values to be logged as key-value pairs. The values need to have type `str`, `float`, `int` or `dict` of
                `str` to `float`/`int`.
            step (`int`, *optional*):
                The run step. If included, the log will be affiliated with this step.
            kwargs:
                Additional key word arguments passed along to either `SummaryWriter.add_scaler`,
                `SummaryWriter.add_text`, or `SummaryWriter.add_scalers` method based on the contents of `values`.
        """
        values = listify(values)
        for k, v in values.items():
            if isinstance(v, (int, float)):
                self.writer.add_scalar(k, v, global_step=step, **kwargs)
            elif isinstance(v, str):
                self.writer.add_text(k, v, global_step=step, **kwargs)
            elif isinstance(v, dict):
                self.writer.add_scalars(k, v, global_step=step, **kwargs)
        self.writer.flush()
        logger.debug("Successfully logged to TensorBoard")


    @on_main_process
    def finish(self):
        """
        Closes `TensorBoard` writer
        """
        self.writer.close()
        logger.debug("TensorBoard writer closed")
