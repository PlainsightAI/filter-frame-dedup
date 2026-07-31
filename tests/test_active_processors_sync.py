"""Regression test for PR #17: active_processors is authoritative over the legacy use_*
flags. Before this, active_processors=['model_dedup'] with the default use_model_dedup=False
raised in ModelProcessor.__init__ ("Model deduplication is disabled..."). normalize_config
now syncs the flags from active_processors so the requested pipeline actually runs."""

from filter_frame_dedup.filter import FilterFrameDedup


def test_model_dedup_in_active_processors_sets_use_model_dedup():
    cfg = FilterFrameDedup.normalize_config({'active_processors': ['model_dedup']})
    assert cfg.use_model_dedup is True
    assert cfg.use_hash_dedup is False


def test_hash_dedup_in_active_processors_sets_use_hash_dedup():
    cfg = FilterFrameDedup.normalize_config({'active_processors': ['motion_gate', 'hash_dedup']})
    assert cfg.use_hash_dedup is True
    assert cfg.use_model_dedup is False


def test_motion_only_leaves_both_flags_false():
    cfg = FilterFrameDedup.normalize_config({'active_processors': ['motion_gate']})
    assert cfg.use_hash_dedup is False
    assert cfg.use_model_dedup is False


def test_default_pipeline_still_builds_when_active_processors_absent():
    # active_processors=None -> derived from use_* defaults (hash on, model off), and the
    # flags stay consistent with the derived list.
    cfg = FilterFrameDedup.normalize_config({})
    assert 'motion_gate' in cfg.active_processors
    assert 'ssim_dedup' in cfg.active_processors
    assert cfg.use_model_dedup is False
