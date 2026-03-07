"""Tests for experiment config generator."""
from __future__ import annotations

import hashlib

import pytest
import yaml

from skills.experiment_config_generator import ExperimentConfigGenerator
from schemas.experiments import ExperimentType


@pytest.fixture
def generator():
    return ExperimentConfigGenerator()


@pytest.fixture
def sample_config(generator):
    return generator.generate_from_suggestion(
        suggestion_id="sug_abc123",
        bot_id="bot_alpha",
        param_name="rsi_threshold",
        current_value=30,
        proposed_value=25,
        title="Test RSI threshold",
        duration_days=14,
    )


class TestGenerateFromSuggestion:
    def test_correct_experiment_id_bot_id_variants(self, generator):
        """Generate config from suggestion: correct experiment_id, bot_id, variants."""
        config = generator.generate_from_suggestion(
            suggestion_id="sug_001",
            bot_id="bot_alpha",
            param_name="lookback",
            current_value=20,
            proposed_value=30,
        )
        assert config.bot_id == "bot_alpha"
        assert len(config.experiment_id) == 16
        assert len(config.variants) == 2
        assert config.experiment_type == ExperimentType.PARAMETER_AB

    def test_control_variant_has_current_value(self, sample_config):
        """Control variant has current_value."""
        control = next(v for v in sample_config.variants if v.name == "control")
        assert control.params["rsi_threshold"] == 30

    def test_treatment_variant_has_proposed_value(self, sample_config):
        """Treatment variant has proposed_value."""
        treatment = next(v for v in sample_config.variants if v.name == "treatment")
        assert treatment.params["rsi_threshold"] == 25

    def test_allocation_sums_to_100(self, sample_config):
        """Allocation sums to 100%."""
        total = sum(v.allocation_pct for v in sample_config.variants)
        assert abs(total - 100.0) < 0.01

    def test_deterministic_experiment_id(self, generator):
        """Deterministic experiment_id from suggestion_id + param."""
        config_a = generator.generate_from_suggestion(
            suggestion_id="sug_x",
            bot_id="bot1",
            param_name="threshold",
            current_value=1,
            proposed_value=2,
        )
        config_b = generator.generate_from_suggestion(
            suggestion_id="sug_x",
            bot_id="bot2",
            param_name="threshold",
            current_value=10,
            proposed_value=20,
        )
        # Same suggestion_id + param_name => same experiment_id
        assert config_a.experiment_id == config_b.experiment_id

        # Different param_name => different experiment_id
        config_c = generator.generate_from_suggestion(
            suggestion_id="sug_x",
            bot_id="bot1",
            param_name="other_param",
            current_value=1,
            proposed_value=2,
        )
        assert config_a.experiment_id != config_c.experiment_id

    def test_source_suggestion_id_links_to_original(self, sample_config):
        """source_suggestion_id links to original suggestion."""
        assert sample_config.source_suggestion_id == "sug_abc123"

    def test_default_title_when_none_provided(self, generator):
        """Default title generated when none provided."""
        config = generator.generate_from_suggestion(
            suggestion_id="sug_y",
            bot_id="bot_beta",
            param_name="stop_loss",
            current_value=0.02,
            proposed_value=0.015,
        )
        assert "stop_loss" in config.title
        assert "bot_beta" in config.title


class TestGenerateBotYaml:
    def test_yaml_output_is_valid(self, generator, sample_config):
        """YAML output is valid YAML and parseable."""
        yaml_str = generator.generate_bot_yaml(sample_config)
        parsed = yaml.safe_load(yaml_str)
        assert "experiments" in parsed
        assert len(parsed["experiments"]) == 1
        exp = parsed["experiments"][0]
        assert exp["experiment_id"] == sample_config.experiment_id
        assert "control" in exp["variants"]
        assert "treatment" in exp["variants"]
        assert exp["max_duration_days"] == 14

    def test_yaml_contains_allocation(self, generator, sample_config):
        """YAML contains allocation percentages."""
        yaml_str = generator.generate_bot_yaml(sample_config)
        parsed = yaml.safe_load(yaml_str)
        exp = parsed["experiments"][0]
        assert exp["variants"]["control"]["allocation_pct"] == 50.0
        assert exp["variants"]["treatment"]["allocation_pct"] == 50.0


class TestGenerateExperimentPR:
    def test_pr_targets_correct_repo_and_branch(self, generator, sample_config):
        """PR request targets correct repo and has correct branch name."""
        pr = generator.generate_experiment_pr(sample_config, repo_dir="/repos/bot_alpha")
        assert pr.repo_dir == "/repos/bot_alpha"
        assert pr.branch_name.startswith("ta/experiment-")
        assert sample_config.experiment_id[:8] in pr.branch_name
        assert pr.bot_id == "bot_alpha"

    def test_pr_file_change_path(self, generator, sample_config):
        """PR has file change targeting experiment config directory."""
        pr = generator.generate_experiment_pr(sample_config, repo_dir="/repos/bot_alpha")
        assert len(pr.file_changes) == 1
        fc = pr.file_changes[0]
        assert fc.file_path == f"config/experiments/{sample_config.experiment_id}.yaml"
        assert fc.original_content == ""
        assert len(fc.new_content) > 0

    def test_pr_suggestion_id_linked(self, generator, sample_config):
        """PR suggestion_id links back to original suggestion."""
        pr = generator.generate_experiment_pr(sample_config, repo_dir="/repos/bot_alpha")
        assert pr.suggestion_id == "sug_abc123"

    def test_pr_body_contains_variants(self, generator, sample_config):
        """PR body contains variant information."""
        pr = generator.generate_experiment_pr(sample_config, repo_dir="/repos/bot_alpha")
        assert "control" in pr.body
        assert "treatment" in pr.body
        assert "A/B Experiment" in pr.body
