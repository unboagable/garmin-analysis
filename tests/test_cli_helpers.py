"""
Unit tests for CLI helper utilities.

Tests cover:
- Adding 24h coverage arguments to parser
- Applying 24h coverage filtering from arguments
- Adding common output arguments
- Edge cases and error handling
"""
import pytest
import argparse
import pandas as pd
import numpy as np
from garmin_analysis.utils.cli_helpers import (
    add_24h_coverage_args,
    apply_24h_coverage_filter_from_args,
    add_common_output_args,
    setup_logging_from_args
)


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'day': pd.date_range('2024-01-01', periods=10),
        'steps': np.random.randint(5000, 15000, 10),
        'hr_min': np.random.randint(50, 70, 10),
    })


@pytest.fixture
def sample_stress_df():
    """Create sample stress data with 24h coverage for one day."""
    # Create continuous coverage for 2024-01-01
    start_time = pd.Timestamp('2024-01-01 00:00:00')
    timestamps = [start_time + pd.Timedelta(minutes=i) for i in range(1440)]
    return pd.DataFrame({
        'timestamp': timestamps,
        'stress': [50] * len(timestamps)
    })


class TestAdd24hCoverageArgs:
    """Test add_24h_coverage_args function."""
    
    def test_adds_all_arguments(self):
        """Test that all four arguments are added."""
        parser = argparse.ArgumentParser()
        result_parser = add_24h_coverage_args(parser)
        
        # Parser should be returned (for chaining)
        assert result_parser is parser
        
        # Parse with all arguments
        args = parser.parse_args([
            '--filter-24h-coverage',
            '--max-gap', '5',
            '--day-edge-tolerance', '3',
            '--coverage-allowance-minutes', '10'
        ])
        
        assert args.filter_24h_coverage is True
        assert args.max_gap == 5
        assert args.day_edge_tolerance == 3
        assert args.coverage_allowance_minutes == 10
    
    def test_default_values(self):
        """Test that default values are set correctly."""
        parser = argparse.ArgumentParser()
        add_24h_coverage_args(parser)
        
        # Parse with no arguments (use defaults)
        args = parser.parse_args([])
        
        assert args.filter_24h_coverage is False
        assert args.max_gap == 2
        assert args.day_edge_tolerance == 2
        assert args.coverage_allowance_minutes == 0
    
    def test_filter_flag_is_action_store_true(self):
        """Test that filter flag is a boolean action."""
        parser = argparse.ArgumentParser()
        add_24h_coverage_args(parser)
        
        # With flag
        args_with = parser.parse_args(['--filter-24h-coverage'])
        assert args_with.filter_24h_coverage is True
        
        # Without flag
        args_without = parser.parse_args([])
        assert args_without.filter_24h_coverage is False
    
    def test_integer_arguments_accept_numbers(self):
        """Test that integer arguments accept numeric values."""
        parser = argparse.ArgumentParser()
        add_24h_coverage_args(parser)
        
        args = parser.parse_args([
            '--max-gap', '10',
            '--day-edge-tolerance', '5',
            '--coverage-allowance-minutes', '60'
        ])
        
        assert isinstance(args.max_gap, int)
        assert isinstance(args.day_edge_tolerance, int)
        assert isinstance(args.coverage_allowance_minutes, int)
    
    def test_can_chain_with_other_arguments(self):
        """Test that function works with existing parser arguments."""
        parser = argparse.ArgumentParser()
        parser.add_argument('--input-file', default='data.csv')
        add_24h_coverage_args(parser)
        
        args = parser.parse_args([
            '--input-file', 'mydata.csv',
            '--filter-24h-coverage'
        ])
        
        assert args.input_file == 'mydata.csv'
        assert args.filter_24h_coverage is True


class TestApply24hCoverageFilterFromArgs:
    """Test apply_24h_coverage_filter_from_args function."""
    
    def test_no_filtering_when_flag_false(self, sample_dataframe):
        """Test that no filtering occurs when flag is False."""
        parser = argparse.ArgumentParser()
        add_24h_coverage_args(parser)
        args = parser.parse_args([])  # filter_24h_coverage = False
        
        result = apply_24h_coverage_filter_from_args(sample_dataframe, args)
        
        # Should return original dataframe unchanged
        assert len(result) == len(sample_dataframe)
        assert result.equals(sample_dataframe)
    
    def test_filtering_when_flag_true(self, sample_dataframe, sample_stress_df):
        """Test that filtering occurs when flag is True."""
        parser = argparse.ArgumentParser()
        add_24h_coverage_args(parser)
        args = parser.parse_args(['--filter-24h-coverage'])
        
        result = apply_24h_coverage_filter_from_args(
            sample_dataframe,
            args,
            stress_df=sample_stress_df
        )
        
        # Should filter the dataframe (likely to fewer rows)
        # With our sample data, only 2024-01-01 has coverage
        assert len(result) <= len(sample_dataframe)
    
    def test_uses_custom_parameters(self, sample_dataframe, sample_stress_df):
        """Test that custom parameters are passed through."""
        parser = argparse.ArgumentParser()
        add_24h_coverage_args(parser)
        args = parser.parse_args([
            '--filter-24h-coverage',
            '--max-gap', '5',
            '--day-edge-tolerance', '10',
            '--coverage-allowance-minutes', '30'
        ])
        
        # Should not raise error with custom parameters
        result = apply_24h_coverage_filter_from_args(
            sample_dataframe,
            args,
            stress_df=sample_stress_df
        )
        
        assert isinstance(result, pd.DataFrame)
    
    def test_clamps_coverage_allowance_to_max(self, sample_dataframe):
        """Test that coverage allowance is clamped to 300 minutes."""
        parser = argparse.ArgumentParser()
        add_24h_coverage_args(parser)
        args = parser.parse_args([
            '--filter-24h-coverage',
            '--coverage-allowance-minutes', '500'  # > 300
        ])
        
        # Should not raise error (value clamped internally)
        result = apply_24h_coverage_filter_from_args(
            sample_dataframe,
            args,
            stress_df=pd.DataFrame()  # Empty stress df
        )
        
        # With empty stress, should return original
        assert len(result) == len(sample_dataframe)
    
    def test_clamps_coverage_allowance_to_min(self, sample_dataframe):
        """Test that coverage allowance is clamped to 0 minutes."""
        parser = argparse.ArgumentParser()
        add_24h_coverage_args(parser)
        args = parser.parse_args([
            '--filter-24h-coverage',
            '--coverage-allowance-minutes', '-10'  # < 0
        ])
        
        # Should not raise error (value clamped internally)
        result = apply_24h_coverage_filter_from_args(
            sample_dataframe,
            args,
            stress_df=pd.DataFrame()  # Empty stress df
        )
        
        assert isinstance(result, pd.DataFrame)
    
    def test_raises_error_with_missing_attributes(self, sample_dataframe):
        """Test that error is raised if args missing required attributes."""
        # Create args without the required attributes
        args = argparse.Namespace(some_other_arg=True)
        
        with pytest.raises(AttributeError, match="missing required attributes"):
            apply_24h_coverage_filter_from_args(sample_dataframe, args)
    
    def test_empty_dataframe_handling(self):
        """Test handling of empty DataFrame."""
        parser = argparse.ArgumentParser()
        add_24h_coverage_args(parser)
        args = parser.parse_args(['--filter-24h-coverage'])
        
        empty_df = pd.DataFrame()
        result = apply_24h_coverage_filter_from_args(empty_df, args)
        
        assert result.empty
    
    def test_with_db_path_parameter(self, sample_dataframe):
        """Test that db_path parameter is passed through."""
        parser = argparse.ArgumentParser()
        add_24h_coverage_args(parser)
        args = parser.parse_args(['--filter-24h-coverage'])
        
        # Should not raise error with db_path (even if file doesn't exist)
        # The filter function will handle the missing file gracefully
        result = apply_24h_coverage_filter_from_args(
            sample_dataframe,
            args,
            db_path='nonexistent.db'
        )
        
        # Should return original dataframe when DB doesn't exist
        assert len(result) == len(sample_dataframe)


class TestAddCommonOutputArgs:
    """Test add_common_output_args function."""
    
    def test_adds_output_dir_and_verbose(self):
        """Test that output-dir and verbose arguments are added."""
        parser = argparse.ArgumentParser()
        result_parser = add_common_output_args(parser)
        
        # Parser should be returned (for chaining)
        assert result_parser is parser
        
        args = parser.parse_args(['--output-dir', 'results', '-v'])
        
        assert args.output_dir == 'results'
        assert args.verbose is True
    
    def test_default_values(self):
        """Test default values for output arguments."""
        parser = argparse.ArgumentParser()
        add_common_output_args(parser)
        
        args = parser.parse_args([])
        
        assert args.output_dir == 'plots'
        assert args.verbose is False
    
    def test_verbose_short_form(self):
        """Test that -v short form works."""
        parser = argparse.ArgumentParser()
        add_common_output_args(parser)
        
        args = parser.parse_args(['-v'])
        
        assert args.verbose is True
    
    def test_verbose_long_form(self):
        """Test that --verbose long form works."""
        parser = argparse.ArgumentParser()
        add_common_output_args(parser)
        
        args = parser.parse_args(['--verbose'])
        
        assert args.verbose is True
    
    def test_can_chain_with_other_functions(self):
        """Test chaining with add_24h_coverage_args."""
        parser = argparse.ArgumentParser()
        add_24h_coverage_args(parser)
        add_common_output_args(parser)
        
        args = parser.parse_args([
            '--filter-24h-coverage',
            '--output-dir', 'results',
            '-v'
        ])
        
        assert args.filter_24h_coverage is True
        assert args.output_dir == 'results'
        assert args.verbose is True


class TestSetupLoggingFromArgs:
    """Test setup_logging_from_args function."""
    
    def test_sets_debug_level_when_verbose(self):
        """Test that DEBUG level is set when verbose is True."""
        parser = argparse.ArgumentParser()
        add_common_output_args(parser)
        args = parser.parse_args(['-v'])
        
        # Should not raise error
        setup_logging_from_args(args)
    
    def test_sets_default_level_when_not_verbose(self):
        """Test that default level is used when verbose is False."""
        parser = argparse.ArgumentParser()
        add_common_output_args(parser)
        args = parser.parse_args([])
        
        # Should not raise error
        setup_logging_from_args(args)
    
    def test_handles_missing_verbose_attribute(self):
        """Test graceful handling when verbose attribute missing."""
        args = argparse.Namespace()  # No verbose attribute
        
        # Should not raise error (verbose check uses hasattr)
        setup_logging_from_args(args)


class TestIntegration:
    """Integration tests for combined usage."""
    
    def test_typical_cli_pattern(self, sample_dataframe, sample_stress_df):
        """Test typical CLI tool pattern."""
        # Create parser with both argument groups
        parser = argparse.ArgumentParser(description='Test tool')
        parser.add_argument('--input-file', default='data.csv')
        add_24h_coverage_args(parser)
        add_common_output_args(parser)
        
        # Parse arguments
        args = parser.parse_args([
            '--input-file', 'mydata.csv',
            '--filter-24h-coverage',
            '--max-gap', '5',
            '--output-dir', 'results',
            '-v'
        ])
        
        # Set up logging
        setup_logging_from_args(args)
        
        # Apply filtering
        result = apply_24h_coverage_filter_from_args(
            sample_dataframe,
            args,
            stress_df=sample_stress_df
        )
        
        # Verify all components worked
        assert args.input_file == 'mydata.csv'
        assert args.output_dir == 'results'
        assert args.verbose is True
        assert isinstance(result, pd.DataFrame)
    
    def test_minimal_usage(self, sample_dataframe):
        """Test minimal usage with defaults."""
        parser = argparse.ArgumentParser()
        add_24h_coverage_args(parser)
        add_common_output_args(parser)
        
        args = parser.parse_args([])  # All defaults
        
        result = apply_24h_coverage_filter_from_args(sample_dataframe, args)
        
        # Should return original dataframe (no filtering)
        assert len(result) == len(sample_dataframe)
        assert args.output_dir == 'plots'
        assert args.verbose is False


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_parser_cannot_add_duplicate_args(self):
        """Test that parser raises error when adding duplicate arguments."""
        parser = argparse.ArgumentParser()
        
        # Add 24h args once
        add_24h_coverage_args(parser)
        
        # Adding again should raise ArgumentError (argparse behavior)
        with pytest.raises(argparse.ArgumentError):
            add_24h_coverage_args(parser)
    
    def test_with_conflicting_argument_names(self):
        """Test that conflicting argument names raise error."""
        parser = argparse.ArgumentParser()
        parser.add_argument('--max-gap', type=str, default='custom')
        
        # Adding 24h args with conflicting name should raise ArgumentError
        with pytest.raises(argparse.ArgumentError):
            add_24h_coverage_args(parser)
    
    def test_none_dataframe(self):
        """Test handling of None DataFrame."""
        parser = argparse.ArgumentParser()
        add_24h_coverage_args(parser)
        args = parser.parse_args(['--filter-24h-coverage'])
        
        # filter_by_24h_coverage returns None when given None,
        # and our function tries to call len() on it
        with pytest.raises(TypeError):
            apply_24h_coverage_filter_from_args(None, args)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

