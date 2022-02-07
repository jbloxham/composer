# Copyright 2021 MosaicML. All Rights Reserved.

import logging
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import torch
import yahp as hp
from torch.optim.lr_scheduler import (CosineAnnealingLR, CosineAnnealingWarmRestarts, ExponentialLR, LambdaLR,
                                      MultiStepLR, StepLR, _LRScheduler)

from composer.core import State
from composer.core.time import Time, Timer, TimeUnit
from composer.core.types import ComposerSchedulerFn, Optimizer, Scheduler, Schedulers
from composer.optim.pytorch_future import LinearLR, WarmUpLR
from composer.utils._time_conversion import convert as convert_time
from composer.utils.iter_helpers import ensure_tuple

log = logging.getLogger(__name__)

_interval_doc = 'frequency of step() calls, either "batch" or "epoch". Default: "epoch"'

# Allow (batch, batches) or (epoch, epochs). Also accept "step" ~ "batch"
INTERVAL_MAP = {
    'batch': 'batch',
    'batches': 'batch',
    'epoch': 'epoch',
    'epochs': 'epoch',
    'step': 'batch',
    'steps': 'batch'
}


def _convert_time_fields(interval: str,
                         kwargs: Dict[str, Any],
                         max_training_duration: Optional[Union[str, Time[int]]] = None,
                         steps_per_epoch: Optional[int] = None,
                         samples_per_epoch: Optional[int] = None,
                         dataset_num_tokens: Optional[int] = None) -> None:
    """Converts all fields in ``kwargs`` that were provided as timestrings (e.g. "32ep") into integers, representing
    either epochs or batches, depending on the ``interval``.

    Modifies ``kwargs`` in place.
    """
    interval_unit = TimeUnit(INTERVAL_MAP[interval])

    for field_name, field_value in kwargs.items():

        if field_name not in ('interval', 'warmup_method'):
            if isinstance(field_value, list) and all(isinstance(x, str) for x in field_value):
                kwargs[field_name] = [
                    convert_time(t,
                                 unit=interval_unit,
                                 steps_per_epoch=steps_per_epoch,
                                 max_training_duration=max_training_duration,
                                 samples_per_epoch=samples_per_epoch,
                                 dataset_num_tokens=dataset_num_tokens).value for t in field_value
                ]
                continue
            if isinstance(field_value, str):
                kwargs[field_name] = convert_time(field_value,
                                                  unit=interval_unit,
                                                  steps_per_epoch=steps_per_epoch,
                                                  max_training_duration=max_training_duration,
                                                  samples_per_epoch=samples_per_epoch,
                                                  dataset_num_tokens=dataset_num_tokens).value


def _convert_time(time: Union[str, Time], state: State) -> Time[int]:
    if isinstance(time, str):
        time = Time.from_timestring(time)

    if time.unit == TimeUnit.DURATION:
        time = convert_time(time=time, unit=state.max_duration.unit, max_training_duration=state.max_duration)

    return time


class ComposerScheduler(ABC):

    @abstractmethod
    def __call__(self, state: State) -> float:
        pass


class StepScheduler(ComposerScheduler):

    def __init__(self, step_size: Union[str, Time], gamma: float = 0.1):
        self.step_size = step_size
        self.gamma = gamma

    def __call__(self, state: State) -> float:
        step_size = _convert_time(self.step_size, state)
        current_time = state.timer.get(step_size.unit)
        steps = int(current_time / step_size)

        return self.gamma**steps


class MultiStepScheduler(ComposerScheduler):

    def __init__(self, milestones: List[Union[str, Time]], gamma: float = 0.1):
        self.milestones = milestones
        self.gamma = gamma

    def __call__(self, state: State):
        milestones = [_convert_time(milestone, state) for milestone in self.milestones]

        factor = 1.0
        for milestone in milestones:
            if state.timer >= milestone:
                factor *= self.gamma

        return factor


class ConstantScheduler(ComposerScheduler):

    def __init__(self, factor: float = 1.0 / 3, total_time: Union[str, Time] = '5ep'):
        self.factor = factor
        self.total_time = total_time

    def __call__(self, state: State):
        total_time = _convert_time(self.total_time, state)

        if state.timer < total_time:
            return self.factor

        return 1.0


class LinearScheduler(ComposerScheduler):

    def __init__(self, start_factor: float = 1.0 / 3, end_factor: float = 1.0, total_time: Union[str, Time] = '5ep'):
        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_time = total_time

    def __call__(self, state: State):
        total_time = _convert_time(self.total_time, state)
        current_time = state.timer.get(total_time.unit)
        frac_of_total = min(1.0, current_time / total_time)

        current_factor = self.start_factor + frac_of_total * (self.end_factor - self.start_factor)

        return current_factor


class ExponentialScheduler(ComposerScheduler):

    def __init__(self, gamma: float, time_unit: TimeUnit = TimeUnit.EPOCH):
        self.gamma = gamma
        self.time_unit = time_unit

    def __call__(self, state: State):
        current_time = state.timer.get(self.time_unit)

        return self.gamma**current_time.value

# TODO: SequentialScheduler

class CosineAnnealingLR(ComposerScheduler):

    def __init__(self, T_max: Union[str, Time], min_factor: float = 0):
        self.T_max = T_max
        self.min_factor = min_factor

    def __call__(self, state: State):



@dataclass
class SchedulerHparams(hp.Hparams, ABC):

    scheduler_object = None  # type: Optional[Callable[..., Scheduler]]
    interval = 'step'  # type: str

    def initialize_object(
        self,
        optimizer: Optimizer,
        steps_per_epoch: Optional[int] = None,
        samples_per_epoch: Optional[int] = None,
        dataset_num_tokens: Optional[int] = None,
        max_training_duration: Optional[Union[str, Time[int]]] = None,
    ) -> Scheduler:
        """Create the scheduler object from the current hparams.

        Args:
            optimizer (Optimizer): the optimizer associated with this scheduler
            steps_per_epoch (int, optional): The number of optimization steps per epoch.
            samples_per_epoch (int, optional): The number of samples trained per epoch.
            dataset_num_tokens (int, optional): The number of tokens in the dataset.
            max_training_duration (str or Time, optional): The total training duration.
        Returns:
            Scheduler: The parametrized scheduler instance
        """

        assert self.scheduler_object is not None, "Scheduler Hparams needs scheduler_object to initialize."
        kwargs = {k: v for k, v in asdict(self).items() if k not in ['interval']}

        _convert_time_fields(interval=self.interval,
                             kwargs=kwargs,
                             max_training_duration=max_training_duration,
                             steps_per_epoch=steps_per_epoch,
                             samples_per_epoch=samples_per_epoch,
                             dataset_num_tokens=dataset_num_tokens)

        # we pass the interval to the trainer directly
        obj = self.scheduler_object(optimizer, **kwargs)
        obj.interval = self.interval  # type: ignore
        obj.steps_per_epoch = steps_per_epoch  # type: ignore
        return obj


class ConstantLR(_LRScheduler):
    """Scheduler that does not change the optimizer's learning rate.

    Args:
        optimizer (Optimizer): the optimizer associated with this scheduler.
        last_epoch (int, optional): The index of the last epoch. Can be used to restore the state of the
                                    learning rate schedule. Default: ``-1``.
        verbose (bool, optional): If ``True``, prints a message to stdout for each update. Default: ``False``.
    """

    def __init__(self, optimizer: Optimizer, last_epoch: int = -1, verbose: int = False):

        self.optimizer = optimizer
        super(ConstantLR, self).__init__(optimizer, last_epoch, verbose)  # type: ignore

    def get_lr(self):
        """Get the current learning rate for each parameter group.

        Returns:
            List of float: The current learning rate for each parameter group.
        """
        return self.base_lrs  # type: ignore

    def _get_closed_form_lr(self):
        """Get the current learning rate for each parameter group.

        Returns:
            List of float: The current learning rate for each parameter group.
        """
        return [base_lr for base_lr in self.base_lrs]  # type: ignore


class PolynomialLR(_LRScheduler):
    """PolynomialLR scales the learning rate by the remaining train time percentage raised to a specific power.

    Args:
        optimizer (Optimizer): the optimizer associated with this scheduler.
        T_max (Time): the number of iterations to perform, either in terms of epochs or batches.
        power (float): the power to use on the remaining train time percentage for the current schedule coeffecient.
        eta_min (float): the minimum learning rate to decay to. Default is ``0``.
        last_epoch (int): the index of the last epoch. Can be used to restore the learning rate schedule state.
            Default: ``-1``
        verbose (bool): If ``True``, prints a message to stdout for each update. Default: ``False``.
    """

    def __init__(self,
                 optimizer: Optimizer,
                 T_max: Time,
                 power: float,
                 eta_min: float = 0,
                 last_epoch: int = -1,
                 verbose: bool = False):

        self.optimizer = optimizer
        self.T_max = T_max
        self.power = power
        self.eta_min = eta_min
        super(PolynomialLR, self).__init__(optimizer, last_epoch, verbose)  # type: ignore

    def get_lr(self):
        coeff = (1 - self.last_epoch / self.T_max)**self.power  # type: ignore
        return [(base_lr - self.eta_min) * coeff + self.eta_min for base_lr in self.base_lrs]  # type: ignore


@dataclass
class PolynomialLRHparams(SchedulerHparams):
    """Hyperparameters for the :class:`PolynomialLR` scheduler."""
    T_max: str = hp.required(doc='Maximum number of iterations.')
    power: float = hp.required(doc='Power of LR schedule.')
    eta_min: float = hp.optional(default=0.0, doc='Minimum learning rate.')
    verbose: bool = hp.optional(default=False, doc='Prints message to stdout.')
    interval: str = hp.optional(default='step', doc=_interval_doc)

    scheduler_object = PolynomialLR


@dataclass
class ConstantLRHparams(SchedulerHparams):
    """Hyperparameters for the :class:`ConstantLR` scheduler."""
    verbose: bool = hp.optional(default=False, doc='prints message to stdout')
    interval: str = hp.optional(default='step', doc=_interval_doc)

    scheduler_object = ConstantLR


@dataclass
class StepLRHparams(SchedulerHparams):
    """Hyperparameters for the `StepLR.

    <https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html#torch.optim.lr_scheduler.StepLR>`_
    scheduler.
    """

    step_size: str = hp.required(doc='Period of learning rate decay')
    gamma: float = hp.optional(default=0.1, doc='multiplicative factor of decay')
    verbose: bool = hp.optional(default=False, doc='prints message to stdout')
    interval: str = hp.optional(default='step', doc=_interval_doc)

    scheduler_object = torch.optim.lr_scheduler.StepLR


@dataclass
class MultiStepLRHparams(SchedulerHparams):
    """Hyperparameters for the `MultiStepLR <https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.MultiSte
    pLR.html#torch.optim.lr_scheduler.MultiStepLR>`_ scheduler."""

    milestones: List[str] = hp.required(doc='List of milestone time strings')
    gamma: float = hp.optional(default=0.1, doc='multiplicative factor of decay')
    verbose: bool = hp.optional(default=False, doc='prints message to stdout')
    interval: str = hp.optional(default='step', doc=_interval_doc)

    scheduler_object = torch.optim.lr_scheduler.MultiStepLR


@dataclass
class ExponentialLRHparams(SchedulerHparams):
    """Hyperparameters for the `ExponentialLR <https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.Expone
    ntialLR.html#torch.optim.lr_scheduler.ExponentialLR>`_ scheduler."""

    gamma: float = hp.required(doc='multiplicative factor of decay')
    verbose: bool = hp.optional(default=False, doc='prints message to stdout')
    interval: str = hp.optional(default='step', doc=_interval_doc)

    scheduler_object = torch.optim.lr_scheduler.ExponentialLR


@dataclass
class CosineAnnealingLRHparams(SchedulerHparams):
    """Hyperparameters for the `CosineAnnealingLR <https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.Co
    sineAnnealingLR.html#torch.optim.lr_scheduler.CosineAnnealingLR>`_ scheduler."""

    T_max: str = hp.required(doc="Maximum scheduler duration.")
    eta_min: float = hp.optional(default=0.0, doc='minimum learning rate.')
    verbose: bool = hp.optional(default=False, doc='prints message to stdout')
    interval: str = hp.optional(default='step', doc=_interval_doc)

    scheduler_object = torch.optim.lr_scheduler.CosineAnnealingLR


@dataclass
class CosineAnnealingWarmRestartsHparams(SchedulerHparams):
    """Hyperparameters for the ``CosineAnnealingWarmRestarts` <https://pytorch.org/docs/stable/generated/torch.optim.lr_
    scheduler.CosineAnnealingWarmRestarts.html#torch.optim.lr_scheduler.CosineAnnealingWarmRestarts>`_ scheduler."""

    T_0: str = hp.required("Duration for the first restart.")
    eta_min: float = hp.optional(default=0.0, doc='minimum learning rate.')
    verbose: bool = hp.optional(default=False, doc='prints message to stdout')
    interval: str = hp.optional(default='step', doc=_interval_doc)
    T_mult: int = hp.optional("A factor increases :math:`T_{i}` after a restart. Default: 1.", default=1)

    scheduler_object = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts


@dataclass
class LinearLRHparams(SchedulerHparams):
    """Hyperparameters for the `LinearLRHparams.

    <https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LinearLR.html>`_ scheduler.
    """

    start_factor: float = hp.optional("Number to multiply learning rate at the start.", default=1.0 / 3)
    end_factor: float = hp.optional("Number to multiply learning rate at the end .", default=1.0)
    total_iters: str = hp.optional("Duration of linear decay steps. Default: 5 iterations.", default="5ba")
    verbose: bool = hp.optional('Prints message to stdout', default=False)
    interval: str = hp.optional(default='step', doc=_interval_doc)

    scheduler_object = LinearLR


@dataclass
class WarmUpLRHparams(SchedulerHparams):
    """Hyperparameters for the :class:`~composer.optim.pytorch_future.WarmUpLR` scheduler.

    See the documentation for :class:`~composer.optim.pytorch_future.WarmUpLR`.
    """

    warmup_factor: float = hp.optional("Number to multiply learning rate at start.", default=1.0 / 3)
    warmup_iters: str = hp.optional("Warmup duration. Default: 5 iterations.", default="5ba")
    warmup_method: str = hp.optional("Warmup method (linear or constant)", default='linear')
    verbose: bool = hp.optional('Prints message to stdout', default=False)
    interval: str = hp.optional('Warmup the LR every step or epoch. Default: epoch', default='step')

    scheduler_object = WarmUpLR


def ensure_warmup_last(schedulers: List[SchedulerHparams]) -> List[SchedulerHparams]:
    """Ensure that WarmUp-based schedulers appear last in the provided list.

    Args:
        schedulers (List[SchedulerHparams]): List of schedulers.

    Returns:
        List[SchedulerHparams]: A sorted list of schedulers with WarmUp-based schedulers at the end.
    """

    return sorted(schedulers, key=lambda x: isinstance(x, (WarmUpLR, WarmUpLRHparams)))


def get_num_warmup_batches(scheduler_hparams: Sequence[SchedulerHparams], steps_per_epoch: Optional[int] = None) -> int:
    """Gets the number of warmup steps declared by a list of schedulers.

    Args:
        scheduler_hparams (Sequence[SchedulerHparams]): List of schedulers
        steps_per_epoch (Optional[int], optional): Number of steps in a single epoch. Default: ``None``.

    Returns:
        int: Number of warmup steps
    """

    warmup_scheduler_hparams = [scheduler for scheduler in scheduler_hparams if isinstance(scheduler, WarmUpLRHparams)]
    if len(warmup_scheduler_hparams):
        warmup_iters = warmup_scheduler_hparams[0].warmup_iters
        if isinstance(warmup_iters, str):
            interval_unit = TimeUnit(INTERVAL_MAP[warmup_scheduler_hparams[0].interval])
            return convert_time(
                time=warmup_iters,
                unit=interval_unit,
                steps_per_epoch=steps_per_epoch,
            ).value
        else:
            return warmup_iters
    return 0


class ComposedScheduler(_LRScheduler):
    """Handles warmup for a chained list of schedulers.

    With one call, will run each scheduler's ``step()``. If :class:`WarmUpLR` is in the list, will delay the stepping of
    schedulers that need to be silent during warmup. ``ComposedScheduler`` handles warmups, where as `ChainedScheduler <https://pytorch.org/docs/1.10./generated/torch.optim.lr_scheduler.ChainedScheduler.html?highlight=chained#torch.optim.lr_scheduler.ChainedScheduler>`_
    only combines schedulers.

    `CosineAnnealingLR
    <https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html#torch.optim.lr_scheduler.CosineAnnealingLR>`_
    and `ExponentialLR
    <https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ExponentialLR.html#torch.optim.lr_scheduler.ExponentialLR>`_
    are not stepped during the warmup period. Other schedulers, such as
    `MultiStepLR
    <https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.MultiStepLR.html#torch.optim.lr_scheduler.MultiStepLR>`_
    are still stepped, to keep their milestones unchanged.

    Handles running the :class:`WarmUpLR` at every step if :attr:`WarmUpLR.interval='batch'`, and other schedulers at
    every epoch.

    Args:
        schedulers (list): List of chained schedulers.
    """

    def __init__(self, schedulers: Schedulers):
        schedulers = ensure_tuple(schedulers)
        self._validate_same_optimizers(schedulers)
        self.schedulers = schedulers
        self.intervals = [getattr(scheduler, "interval", "epoch") for scheduler in schedulers]

        # generous with spelling (batch, batches)/(step, steps) and (epoch, epochs)
        self.intervals = [INTERVAL_MAP[interval] for interval in self.intervals]

        warmup = [(scheduler, interval)
                  for scheduler, interval in zip(self.schedulers, self.intervals)
                  if isinstance(scheduler, WarmUpLR)]
        if warmup:
            assert len(warmup) == 1, "ComposedScheduler only supports one WarmUpLR " \
                                     f"in the provided list, found {len(warmup)}."
            warmup, interval = warmup[0]
            self.warmup_iters = warmup.warmup_iters
            log.info(f'Setting LR Warmup to {self.warmup_iters} {interval}')
        else:
            self.warmup_iters = 0

        # these schedulers need to be silent during warmup
        self.delay_schedulers = [CosineAnnealingLR, CosineAnnealingWarmRestarts, ExponentialLR, LinearLR]
        self._warmup_counter = 0  # counter to track warmups

    def step(self, interval: str = 'epoch'):
        """Step all applicable schedulers.

        Args:
            interval (str, optional): The interval of the current step. Must be either ``'step'`` or ``'epoch'``.
                                      Default: ``epoch``.
        """
        for scheduler, scheduler_interval in zip(self.schedulers, self.intervals):
            if self._warmup_counter < self.warmup_iters and \
                any(isinstance(scheduler, delay) for delay in self.delay_schedulers):
                continue

            if interval == scheduler_interval:
                scheduler.step()
                if isinstance(scheduler, WarmUpLR):
                    self._warmup_counter += 1

    def _validate_schedulers(self, warmup_epochs: int) -> None:
        """Verify that any stepwise schedulers do not change the LR during the desired warmup period.

        Args:
            warmup_epochs (int): Number of epochs for warmup.
        """
        # since WarmUpLR is non-chainable form, step LR milestones must
        # occur after warmup is completed
        lr_step_schedulers = [
            scheduler for scheduler in self.schedulers if isinstance(scheduler, (StepLR, MultiStepLR))
        ]
        for scheduler in lr_step_schedulers:
            if isinstance(scheduler, StepLR) and scheduler.step_size <= warmup_epochs:  # type: ignore
                raise ValueError(f'StepLR step_size {scheduler.step_size} must '  # type: ignore
                                 'be greater than warmup_iters {self.warmup_iters}')
            elif isinstance(scheduler, MultiStepLR):
                if any(ms <= warmup_epochs for ms in scheduler.milestones.elements()):  #type: ignore
                    raise ValueError(f'MultiStepLR milestones must be greater than warmup_iters {warmup_epochs}')

    def state_dict(self) -> Dict[str, Any]:
        """Returns a dictionary containing the state of all composed schedulers.

        Returns:
            Dict: the state dictionary
        """
        state_dict = {
            "schedulers": {scheduler.__class__.__qualname__: scheduler.state_dict() for scheduler in self.schedulers},
            "_warmup_counter": self._warmup_counter,
        }
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load the state of all composed schedulers from the provided dictionary.

        Args:
            state_dict (Dict[str, Any]): A dict containing the state of all composed schedulers. Should be an object
            returned from a call to :meth:`state_dict()`.
        """
        for scheduler in self.schedulers:
            scheduler.load_state_dict(state_dict["schedulers"][scheduler.__class__.__qualname__])
        self._warmup_counter = state_dict["_warmup_counter"]

    def _validate_same_optimizers(self, schedulers: Schedulers):
        """Verify that all schedulers correspond to the same optimizer."""
        schedulers = ensure_tuple(schedulers)
        for i, scheduler in enumerate(schedulers):
            if (getattr(scheduler, "optimizer") != getattr(schedulers[0], "optimizer")):
                raise ValueError("ComposedScheduler expects all schedulers to belong to the same optimizer, but "
                                 f"got schedulers at index 0 and {i} to be different")
