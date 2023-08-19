r"""
Early Stopping
^^^^^^^^^^^^^^

Monitor a metric and stop training when it stops improving.

"""

import logging
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import numpy as np
import lightning.pytorch as pl
from lightning.fabric.utilities.rank_zero import _get_rank
from lightning.pytorch.utilities.rank_zero import rank_prefixed_message, rank_zero_warn
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.callbacks import callback


logger = logging.getLogger("lightning.pytorch")


class EarlyStopping(callback.Callback):
    r"""
    Monitor a metric and stop training when it stops improving.

    Args:
        monitor: quantity to be monitored.
        min_delta: minimum change in the monitored quantity to qualify as an improvement, i.e. an absolute
            change of less than or equal to `min_delta`, will count as no improvement.
        patience: number of checks with no improvement
            after which training will be stopped. Under the default configuration, one check happens after
            every training epoch. However, the frequency of validation can be modified by setting various parameters on
            the ``Trainer``, for example ``check_val_every_n_epoch`` and ``val_check_interval``.

            .. note::

                It must be noted that the patience parameter counts the number of validation checks with
                no improvement, and not the number of training epochs. Therefore, with parameters
                ``check_val_every_n_epoch=10`` and ``patience=3``, the trainer will perform at least 40 training
                epochs before being stopped.

        verbose: verbosity mode.
        mode: one of ``'min'``, ``'max'``. In ``'min'`` mode, training will stop when the quantity
            monitored has stopped decreasing and in ``'max'`` mode it will stop when the quantity
            monitored has stopped increasing.
        strict: whether to crash the training if `monitor` is not found in the validation metrics.
        check_finite: When set ``True``, stops training when the monitor becomes NaN or infinite.
        stopping_threshold: Stop training immediately once the monitored quantity reaches this threshold.
        divergence_threshold: Stop training as soon as the monitored quantity becomes worse than this threshold.
        check_on_train_epoch_end: whether to run early stopping at the end of the training epoch.
            If this is ``False``, then the check runs at the end of the validation.
        log_rank_zero_only: When set ``True``, logs the status of the early stopping callback only for rank 0 process.

    Raises:
        MisconfigurationException:
            If ``mode`` is none of ``"min"`` or ``"max"``.
        RuntimeError:
            If the metric ``monitor`` is not available.

    Example::

        >>> from lightning.pytorch import Trainer
        >>> from lightning.pytorch.callbacks import EarlyStopping
        >>> early_stopping = EarlyStopping('val_loss')
        >>> trainer = Trainer(callbacks=[early_stopping])

    .. tip:: Saving and restoring multiple early stopping callbacks at the same time is supported under variation in the
        following arguments:

        *monitor, mode*

        Read more: :ref:`Persisting Callback State <extensions/callbacks_state:save callback state>`
    """
    mode_dict = {"min": torch.lt, "max": torch.gt}

    order_dict = {"min": "<", "max": ">"}

    def __init__(
        self,
        monitor: str,
        min_delta: float = 0.0,
        patience: int = 3,
        verbose: bool = False,
        mode: str = "min",
        strict: bool = True,
        check_finite: bool = True,
        stopping_threshold: Optional[float] = None,
        divergence_threshold: Optional[float] = None,
        check_on_train_epoch_end: Optional[bool] = None,
        log_rank_zero_only: bool = False,
    ):
        super().__init__()
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.mode = mode
        self.strict = strict
        self.check_finite = check_finite
        self.stopping_threshold = stopping_threshold
        self.divergence_threshold = divergence_threshold
        self.wait_count = 0
        self.stopped_epoch = 0
        self._check_on_train_epoch_end = check_on_train_epoch_end
        self.log_rank_zero_only = log_rank_zero_only

        if self.mode not in self.mode_dict:
            raise MisconfigurationException(
                f"`mode` can be {', '.join(self.mode_dict.keys())}, got {self.mode}")

        self.min_delta *= 1 if self.monitor_op == torch.gt else -1
        torch_inf = torch.tensor(np.Inf)
        self.best_score = torch_inf if self.monitor_op == torch.lt else -torch_inf

    @property
    def state_key(self) -> str:
        return self._generate_state_key(monitor=self.monitor, mode=self.mode)

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        if self._check_on_train_epoch_end is None:
            # if the user runs validation multiple times per training epoch or multiple training epochs without
            # validation, then we run after validation instead of on train epoch end
            self._check_on_train_epoch_end = trainer.val_check_interval == 1.0 and trainer.check_val_every_n_epoch == 1

    def _validate_condition_metric(self, logs: Dict[str, torch.Tensor]) -> bool:
        monitor_val = logs.get(self.monitor)

        error_msg = (
            f"Early stopping conditioned on metric `{self.monitor}` which is not available."
            " Pass in or modify your `EarlyStopping` callback to use any of the following:"
            f' `{"`, `".join(list(logs.keys()))}`'
        )

        if monitor_val is None:
            if self.strict:
                raise RuntimeError(error_msg)
            if self.verbose > 0:
                rank_zero_warn(error_msg, category=RuntimeWarning)

            return False

        return True

    @property
    def monitor_op(self) -> Callable:
        return self.mode_dict[self.mode]

    def state_dict(self) -> Dict[str, Any]:
        return {
            "wait_count": self.wait_count,
            "stopped_epoch": self.stopped_epoch,
            "best_score": self.best_score,
            "patience": self.patience,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.wait_count = state_dict["wait_count"]
        self.stopped_epoch = state_dict["stopped_epoch"]
        self.best_score = state_dict["best_score"]
        self.patience = state_dict["patience"]

    def _should_skip_check(self, trainer: "pl.Trainer") -> bool:
        from lightning.pytorch.trainer.states import TrainerFn

        return trainer.state.fn != TrainerFn.FITTING or trainer.sanity_checking

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not self._check_on_train_epoch_end or self._should_skip_check(trainer):
            return
        self._run_early_stopping_check(trainer)

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self._check_on_train_epoch_end or self._should_skip_check(trainer):
            return
        self._run_early_stopping_check(trainer)

    def _run_early_stopping_check(self, trainer: "pl.Trainer") -> None:
        """Checks whether the early stopping condition is met and if so tells the trainer to stop the training."""
        logs = trainer.callback_metrics

        if trainer.fast_dev_run or not self._validate_condition_metric(  # disable early_stopping with fast_dev_run
            logs
        ):  # short circuit if metric not present
            return

        current = logs[self.monitor].squeeze()
        should_stop, reason = self._evaluate_stopping_criteria(current)

        # stop every ddp process if any world process decides to stop
        should_stop = trainer.strategy.reduce_boolean_decision(
            should_stop, all=False)
        trainer.should_stop = trainer.should_stop or should_stop
        if should_stop:
            self.stopped_epoch = trainer.current_epoch
        if reason and self.verbose:
            self._log_info(trainer, reason, self.log_rank_zero_only)

    def _evaluate_stopping_criteria(self, current: torch.Tensor) -> Tuple[bool, Optional[str]]:
        should_stop = False
        reason = None
        if self.check_finite and not torch.isfinite(current):
            should_stop = True
            reason = (
                f"Monitored metric {self.monitor} = {current} is not finite."
                f" Previous best value was {self.best_score:.3f}. Signaling Trainer to stop."
            )
        elif self.stopping_threshold is not None and self.monitor_op(current, self.stopping_threshold):
            should_stop = True
            reason = (
                "Stopping threshold reached:"
                f" {self.monitor} = {current} {self.order_dict[self.mode]} {self.stopping_threshold}."
                " Signaling Trainer to stop."
            )
        elif self.divergence_threshold is not None and self.monitor_op(-current, -self.divergence_threshold):
            should_stop = True
            reason = (
                "Divergence threshold reached:"
                f" {self.monitor} = {current} {self.order_dict[self.mode]} {self.divergence_threshold}."
                " Signaling Trainer to stop."
            )
        elif self.monitor_op(current - self.min_delta, self.best_score.to(current.device)):
            should_stop = False
            reason = None
            self.best_score = current
            self.wait_count = 0
        else:
            self.wait_count += 1
            if self.wait_count >= self.patience:
                should_stop = True
                reason = (
                    f"Monitored metric {self.monitor} did not improve in the last {self.wait_count} records."
                    f" Best score: {self.best_score:.3f}. Signaling Trainer to stop."
                )

        return should_stop, reason

    @staticmethod
    def _log_info(trainer: Optional["pl.Trainer"], message: str, log_rank_zero_only: bool) -> None:
        rank = _get_rank(
            strategy=(trainer.strategy if trainer is not None else None),
        )
        if trainer is not None and trainer.world_size <= 1:
            rank = None
        message = rank_prefixed_message(message, rank)
        if rank is None or not log_rank_zero_only or rank == 0:
            logger.info(f"EarlyStopping: {message}")


class DiffEarlyStopping(EarlyStopping):
    r"""
    Monitor a metric and stop training when it stops improving.

    Args:
        diff_threshold: minimum diff of the value of val_loss - train_loss.
    """

    def __init__(self, monitor1: str = 'val_loss', monitor2: str = 'train_loss', diff_threshold: float = 0.5, *args, **kwargs):
        super().__init__(monitor=None, *args, **kwargs)
        self.monitor1 = monitor1
        self.monitor2 = monitor2
        self.diff_threshold = diff_threshold

    def _evaluate_stopping_criteria(self) -> Tuple[bool, Optional[str]]:
        should_stop = False
        reason = None

        metric1_value = self.logs.get(self.monitor1)
        metric2_value = self.logs.get(self.monitor2)
        if metric1_value is None or metric2_value is None:
            return
        if metric1_value - metric2_value > self.diff_threshold:
            should_stop = True
            reason = (
                f"Monitored metrics {self.monitor1}-{self.monitor2} >= {self.diff_threshold}."
                f" Best score: {self.best_score:.3f}. Signaling Trainer to stop."
            )

        return should_stop, reason

    def _validate_condition_metric(self) -> bool:
        monitor_val1 = self.logs.get(self.monitor1)
        monitor_val2 = self.logs.get(self.monitor2)

        error_msg = (
            f"Early stopping conditioned on metric `{self.monitor1} and {self.monitor2}` which is not available."
            " Pass in or modify your `EarlyStopping` callback to use any of the following:"
            f' `{"`, `".join(list(self.logs.keys()))}`'
        )

        if not (monitor_val1 and monitor_val2):
            if self.strict:
                raise RuntimeError(error_msg)
            if self.verbose > 0:
                rank_zero_warn(error_msg, category=RuntimeWarning)

            return False

        return True

    def _run_early_stopping_check(self, trainer):
        self.logs = trainer.callback_metrics

        if trainer.fast_dev_run or not self._validate_condition_metric(  # disable early_stopping with fast_dev_run
        ):  # short circuit if metric not present
            return

        should_stop, reason = self._evaluate_stopping_criteria()

        # stop every ddp process if any world process decides to stop
        should_stop = trainer.strategy.reduce_boolean_decision(
            should_stop, all=False)
        trainer.should_stop = trainer.should_stop or should_stop

        if should_stop:
            self.stopped_epoch = trainer.current_epoch
        if reason and self.verbose:
            self._log_info(trainer, reason, self.log_rank_zero_only)

    @staticmethod
    def _log_info(trainer: Optional[pl.Trainer], message: str, log_rank_zero_only: bool) -> None:

        rank = _get_rank(
            strategy=(trainer.strategy if trainer is not None else None),
        )
        if trainer is not None and trainer.world_size <= 1:
            rank = None
        message = rank_prefixed_message(message, rank)
        if rank is None or not log_rank_zero_only or rank == 0:
            logger.info(f"DiffEarlyStopping: {message}")
