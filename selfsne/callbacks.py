# -*- coding: utf-8 -*-
# Copyright 2021 Jacob M. Graving <jgraving@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBar
from pytorch_lightning.utilities.types import STEP_OUTPUT
from typing_extensions import override


class TotalStepsProgressBar(RichProgressBar):
    """
    A progress bar that tracks global training steps (current step / total steps)
    instead of batches within each epoch. Additionally, if max_epochs is provided,
    it displays the current epoch/max_epochs in the training bar (unless max_steps is set).
    This callback also adds progress bars for validation, testing, and predicting.
    """

    def __init__(self):
        super().__init__()
        self.train_progress_bar_id = None
        self.val_progress_bar_id = None
        self.test_progress_bar_id = None
        self.predict_progress_bar_id = None

    @override
    def on_train_start(self, trainer, pl_module) -> None:
        super().on_train_start(trainer, pl_module)
        if trainer.max_steps is not None and trainer.max_steps > 0:
            total_steps = trainer.max_steps
            description = "Training"
        else:
            total_steps = trainer.max_epochs * trainer.num_training_batches
            description = (
                f"Training (Epoch {trainer.current_epoch+1}/{trainer.max_epochs})"
                if trainer.max_epochs and trainer.max_epochs > 1
                else "Training"
            )
        if self.train_progress_bar_id is None:
            self.train_progress_bar_id = self.progress.add_task(
                description, total=total_steps, visible=True
            )
        self.refresh()

    @override
    def on_train_epoch_start(self, trainer, pl_module) -> None:
        if trainer.max_steps is None or trainer.max_steps <= 0:
            if trainer.max_epochs and trainer.max_epochs > 1:
                self.progress.update(
                    self.train_progress_bar_id,
                    description=f"Training (Epoch {trainer.current_epoch+1}/{trainer.max_epochs})",
                )
        self.refresh()

    @override
    def on_train_batch_end(
        self, trainer, pl_module, outputs: STEP_OUTPUT, batch: any, batch_idx: int
    ) -> None:
        if self.is_disabled:
            return
        self.progress.update(self.train_progress_bar_id, completed=trainer.global_step)
        self._update_metrics(trainer, pl_module)
        self.refresh()
