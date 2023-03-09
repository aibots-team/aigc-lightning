:orphan:

.. _data:

#############
Managing Data
#############

Why Use LightningDataModule?
============================

For information about the :class:`~lightning.pytorch.core.datamodule.LightningDataModule`, please refer to the :ref:`datamodules <datamodules>` section in the documentation.

---------

Arbitrary iterable support
==========================

Python iterables are objects that can be iterated or looped over. Examples of iterables in Python include lists and dictionaries.
In PyTorch, a :class:`torch.utils.data.DataLoader` is also an iterable which typically retrieves data from a :class:`torch.utils.data.Dataset` or :class:`torch.utils.data.IterableDataset`.

The :class:`~lightning.pytorch.trainer.trainer.Trainer` works with arbitrary iterables, but most people will use a :class:`torch.utils.data.DataLoader` as the iterable to feed data to the model.

.. _multiple-dataloaders:

******************
Multiple Iterables
******************

In addition to supporting arbitrary iterables, the ``Trainer`` also supports arbitrary collections of iterables. Some examples of this are:

.. code-block:: python

    return DataLoader(...)
    return list(range(1000))

    # pass loaders as a dict. This will create batches like this:
    # {'a': batch_from_loader_a, 'b': batch_from_loader_b}
    return {"a": DataLoader(...), "b": DataLoader(...)}

    # pass loaders as list. This will create batches like this:
    # [batch_from_dl_1, batch_from_dl_2]
    return [DataLoader(...), DataLoader(...)]

    # {'a': [batch_from_dl_1, batch_from_dl_2], 'b': [batch_from_dl_3, batch_from_dl_4]}
    return {"a": [dl1, dl2], "b": [dl3, dl4]}

Lightning automatically collates the batches from multiple iterables based on a "mode". This is done with our
:class:`~lightning.pytorch.utilities.combined_loader.CombinedLoader` class.
The list of modes available can be found by looking at the :paramref:`~lightning.pytorch.utilities.combined_loader.CombinedLoader.mode` documentation.

By default, the ``"max_size_cycle"`` mode is used during training and the ``"sequential"`` mode is used during validation, testing, and prediction.
To choose a different mode, you can use the :class:`~lightning.pytorch.utilities.combined_loader.CombinedLoader` class directly with your mode of choice:

.. code-block:: python

    from lightning.pytorch.utilities import CombinedLoader

    iterables = {"a": DataLoader(), "b": DataLoader()}
    combined_loader = CombinedLoader(iterables, mode="min_size")
    model = ...
    trainer = Trainer()
    trainer.fit(model, combined_loader)


Currently, ``trainer.validate``, ``trainer.test``, and ``trainer.predict`` methods only support the ``"sequential"`` mode, while ``trainer.fit`` method does not support it.
Support for this feature is tracked in this `issue <https://github.com/Lightning-AI/lightning/issues/16830>`__.

Note that when using the ``"sequential"`` mode, you need to add an additional argument ``dataloader_idx`` to some specific hooks.
Lightning will `raise an error <https://github.com/Lightning-AI/lightning/pull/16837>`__ informing you of this requirement.


Using LightningDataModule
=========================

You can set more than one :class:`~torch.utils.data.DataLoader` in your :class:`~lightning.pytorch.core.datamodule.LightningDataModule` using its DataLoader hooks
and Lightning will use the correct one.

.. testcode::

    class DataModule(LightningDataModule):
        def train_dataloader(self):
            # any iterable or collection of iterables
            return DataLoader(self.train_dataset)

        def val_dataloader(self):
            # any iterable or collection of iterables
            return [DataLoader(self.val_dataset_1), DataLoader(self.val_dataset_2)]

        def test_dataloader(self):
            # any iterable or collection of iterables
            return DataLoader(self.test_dataset)

        def predict_dataloader(self):
            # any iterable or collection of iterables
            return DataLoader(self.predict_dataset)


Using LightningModule Hooks
===========================

The exact same code as above works when overriding :class:`~lightning.pytorch.core.module.LightningModule`


Passing the iterables to the Trainer
====================================

The same support for arbitrary iterables, or collection of iterables applies to the dataloader arguments of
:meth:`~lightning.pytorch.trainer.trainer.Trainer.fit`, :meth:`~lightning.pytorch.trainer.trainer.Trainer.validate`,
:meth:`~lightning.pytorch.trainer.trainer.Trainer.test`, :meth:`~lightning.pytorch.trainer.trainer.Trainer.predict`

--------------

*********************
Accessing DataLoaders
*********************

In the case that you require access to the DataLoader or Dataset objects, DataLoaders for each step can be accessed
via the trainer properties :meth:`~lightning.pytorch.trainer.trainer.Trainer.train_dataloader`,
:meth:`~lightning.pytorch.trainer.trainer.Trainer.val_dataloaders`,
:meth:`~lightning.pytorch.trainer.trainer.Trainer.test_dataloaders`, and
:meth:`~lightning.pytorch.trainer.trainer.Trainer.predict_dataloaders`.

.. code-block:: python

    dataloaders = trainer.train_dataloader
    dataloaders = trainer.val_dataloaders
    dataloaders = trainer.test_dataloaders
    dataloaders = trainer.predict_dataloaders

These properties will match exactly what was returned in your ``*_dataloader`` hooks or passed to the ``Trainer``,
meaning that if you returned a dictionary of dataloaders, these will return a dictionary of dataloaders.

If you are using a :class:`~lightning.pytorch.utilities.CombinedLoader`. A flattened list of DataLoaders can be accessed by doing:

.. code-block:: python

    from lightning.pytorch.utilities import CombinedLoader

    iterables = {"dl1": dl1, "dl2": dl2}
    combined_loader = CombinedLoader(iterables)
    # access the original iterables
    assert combined_loader.iterables is iterables
    # the `.flattened` property can be convenient
    assert combined_loader.flattened == [dl1, dl2]
    # for example, to do a simple loop
    updated = []
    for dl in combined_loader.flattened:
        new_dl = apply_some_transformation_to(dl)
        updated.append(new_dl)
    # it also allows you to easily replace the dataloaders
    combined_loader.flattened = updated

--------------

.. _sequential-data:

***************
Sequential Data
***************

Lightning has built in support for dealing with sequential data.


Packed Sequences as Inputs
==========================

When using :class:`~torch.nn.utils.rnn.PackedSequence`, do two things:

1. Return either a padded tensor in dataset or a list of variable length tensors in the DataLoader's `collate_fn <https://pytorch.org/docs/stable/data.html#dataloader-collate-fn>`_ (example shows the list implementation).
2. Pack the sequence in forward or training and validation steps depending on use case.

|

.. testcode::

    # For use in DataLoader
    def collate_fn(batch):
        x = [item[0] for item in batch]
        y = [item[1] for item in batch]
        return x, y


    # In LightningModule
    def training_step(self, batch, batch_idx):
        x = rnn.pack_sequence(batch[0], enforce_sorted=False)
        y = rnn.pack_sequence(batch[1], enforce_sorted=False)

Iterable Datasets
=================
Lightning supports using :class:`~torch.utils.data.IterableDataset` as well as map-style Datasets. IterableDatasets provide a more natural
option when using sequential data.

.. note:: When using an :class:`~torch.utils.data.IterableDataset` you must set the ``val_check_interval`` to 1.0 (the default) or an int
    (specifying the number of training batches to run before each validation loop) when initializing the Trainer. This is
    because the IterableDataset does not have a ``__len__`` and Lightning requires this to calculate the validation
    interval when ``val_check_interval`` is less than one. Similarly, you can set ``limit_{mode}_batches`` to a float or
    an int. If it is set to 0.0 or 0, it will set ``num_{mode}_batches`` to 0, if it is an int, it will set ``num_{mode}_batches``
    to ``limit_{mode}_batches``, if it is set to 1.0 it will run for the whole dataset, otherwise it will throw an exception.
    Here ``mode`` can be train/val/test/predict.

When iterable datasets are used, Lightning will pre-fetch 1 batch (in addition to the current batch) so it can detect
when the training will stop and run validation if necessary.

.. testcode::

    # IterableDataset
    class CustomDataset(IterableDataset):
        def __init__(self, data):
            self.data_source = data

        def __iter__(self):
            return iter(self.data_source)


    # Setup DataLoader
    def train_dataloader(self):
        seq_data = ["A", "long", "time", "ago", "in", "a", "galaxy", "far", "far", "away"]
        iterable_dataset = CustomDataset(seq_data)

        dataloader = DataLoader(dataset=iterable_dataset, batch_size=5)
        return dataloader


.. testcode::

    # Set val_check_interval as an int
    trainer = Trainer(val_check_interval=100)

    # Disable validation: Set limit_val_batches to 0.0 or 0
    trainer = Trainer(limit_val_batches=0.0)

    # Set limit_val_batches as an int
    trainer = Trainer(limit_val_batches=100)
