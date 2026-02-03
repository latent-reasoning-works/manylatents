# manylatents/lightning/callbacks/tests/test_audit.py
import pytest
from manylatents.lightning.callbacks.audit import AuditTrigger


def test_audit_trigger_step_based():
    trigger = AuditTrigger(every_n_steps=100)

    assert trigger.should_fire(step=0, epoch=0) is True   # First step
    assert trigger.should_fire(step=50, epoch=0) is False
    assert trigger.should_fire(step=100, epoch=0) is True
    assert trigger.should_fire(step=200, epoch=0) is True


def test_audit_trigger_epoch_based():
    trigger = AuditTrigger(every_n_epochs=2)

    assert trigger.should_fire(step=0, epoch=0, epoch_end=True) is True
    assert trigger.should_fire(step=0, epoch=1, epoch_end=True) is False
    assert trigger.should_fire(step=0, epoch=2, epoch_end=True) is True


def test_audit_trigger_combined():
    trigger = AuditTrigger(every_n_steps=100, every_n_epochs=1)

    # Steps trigger
    assert trigger.should_fire(step=100, epoch=0) is True
    # Epoch also triggers
    assert trigger.should_fire(step=50, epoch=1, epoch_end=True) is True


def test_audit_trigger_disabled():
    trigger = AuditTrigger()  # No triggers set

    assert trigger.should_fire(step=100, epoch=5) is False
