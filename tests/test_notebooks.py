"""
Test suite for Jupyter notebooks.

This module tests that all notebooks can be executed end-to-end without errors.
Tests are marked as integration tests since they require real data files.

Usage:
    # Run all notebook tests
    pytest tests/test_notebooks.py -v

    # Run with integration marker
    pytest -m integration tests/test_notebooks.py -v

    # Skip notebook tests
    pytest tests/test_notebooks.py --skip-notebooks
"""

import logging
import pytest
from pathlib import Path
from typing import List, Optional
import tempfile
import shutil

try:
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError
    NBCONVERT_AVAILABLE = True
except ImportError:
    NBCONVERT_AVAILABLE = False
    nbformat = None
    ExecutePreprocessor = None
    CellExecutionError = None


logger = logging.getLogger(__name__)

# Configuration
NOTEBOOKS_DIR = Path(__file__).parent.parent / "notebooks"
TIMEOUT = 600  # 10 minutes per notebook
KERNEL_NAME = "python3"

# Notebooks that require external dependencies or special setup
# These will be skipped or handled differently
NOTEBOOKS_WITH_EXTERNAL_DEPS = {
    "hr_daily.ipynb": "Requires garmindb package not in standard dependencies"
}


def get_all_notebooks() -> List[Path]:
    """
    Discover all Jupyter notebooks in the notebooks directory.
    
    Returns:
        List of Path objects for each notebook
    """
    if not NOTEBOOKS_DIR.exists():
        logger.warning(f"Notebooks directory not found: {NOTEBOOKS_DIR}")
        return []
    
    notebooks = list(NOTEBOOKS_DIR.glob("*.ipynb"))
    
    # Exclude checkpoint files
    notebooks = [nb for nb in notebooks if ".ipynb_checkpoints" not in str(nb)]
    
    logger.info(f"Found {len(notebooks)} notebooks in {NOTEBOOKS_DIR}")
    return sorted(notebooks)


def execute_notebook(
    notebook_path: Path,
    timeout: int = TIMEOUT,
    kernel_name: str = KERNEL_NAME
) -> tuple[bool, Optional[str]]:
    """
    Execute a Jupyter notebook and return success status.
    
    Args:
        notebook_path: Path to the notebook file
        timeout: Maximum execution time in seconds
        kernel_name: Jupyter kernel to use
        
    Returns:
        Tuple of (success: bool, error_message: Optional[str])
    """
    if not NBCONVERT_AVAILABLE:
        return False, "nbconvert not available"
    
    logger.info(f"Executing notebook: {notebook_path.name}")
    
    try:
        # Read the notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = nbformat.read(f, as_version=4)
        
        # Create an executor
        executor = ExecutePreprocessor(
            timeout=timeout,
            kernel_name=kernel_name,
            # Allow errors to be caught and reported
            allow_errors=False
        )
        
        # Execute the notebook in a temporary directory to avoid conflicts
        with tempfile.TemporaryDirectory() as tmpdir:
            # Execute the notebook
            executor.preprocess(
                notebook,
                {'metadata': {'path': str(notebook_path.parent)}}
            )
        
        logger.info(f"✓ Successfully executed: {notebook_path.name}")
        return True, None
        
    except CellExecutionError as e:
        error_msg = f"Error executing cell:\n{str(e)}"
        logger.error(f"✗ Failed to execute {notebook_path.name}: {error_msg}")
        return False, error_msg
        
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(f"✗ Failed to execute {notebook_path.name}: {error_msg}")
        return False, error_msg


@pytest.fixture
def notebooks_dir() -> Path:
    """Fixture providing the notebooks directory path."""
    return NOTEBOOKS_DIR


@pytest.fixture
def all_notebooks() -> List[Path]:
    """Fixture providing list of all notebooks."""
    return get_all_notebooks()


@pytest.mark.integration
@pytest.mark.skipif(not NBCONVERT_AVAILABLE, reason="nbconvert not installed")
class TestNotebooks:
    """Test suite for Jupyter notebooks."""
    
    def test_nbconvert_available(self):
        """Test that nbconvert is available."""
        assert NBCONVERT_AVAILABLE, "nbconvert must be installed to test notebooks"
    
    def test_notebooks_directory_exists(self, notebooks_dir):
        """Test that the notebooks directory exists."""
        assert notebooks_dir.exists(), f"Notebooks directory not found: {notebooks_dir}"
        assert notebooks_dir.is_dir(), f"Notebooks path is not a directory: {notebooks_dir}"
    
    def test_notebooks_found(self, all_notebooks):
        """Test that notebooks are found in the directory."""
        assert len(all_notebooks) > 0, "No notebooks found in notebooks directory"
        logger.info(f"Found {len(all_notebooks)} notebooks to test")
    
    @pytest.mark.parametrize("notebook_path", get_all_notebooks())
    def test_notebook_executes(self, notebook_path: Path):
        """
        Test that each notebook executes without errors.
        
        This test will:
        1. Load the notebook
        2. Execute all cells in sequence
        3. Fail if any cell raises an error
        4. Skip notebooks with external dependencies with a clear message
        """
        notebook_name = notebook_path.name
        
        # Check if notebook has external dependencies
        if notebook_name in NOTEBOOKS_WITH_EXTERNAL_DEPS:
            reason = NOTEBOOKS_WITH_EXTERNAL_DEPS[notebook_name]
            pytest.skip(f"Skipping {notebook_name}: {reason}")
        
        # Execute the notebook
        success, error_message = execute_notebook(notebook_path)
        
        # Assert success
        assert success, (
            f"Notebook {notebook_name} failed to execute.\n"
            f"Error: {error_message}\n"
            f"Path: {notebook_path}"
        )
    
    def test_notebook_structure(self, all_notebooks):
        """
        Test that all notebooks have valid structure.
        
        Checks:
        - Valid JSON format
        - Has cells
        - Has valid nbformat version
        """
        if not NBCONVERT_AVAILABLE:
            pytest.skip("nbconvert not available")
        
        for notebook_path in all_notebooks:
            logger.info(f"Checking structure of {notebook_path.name}")
            
            try:
                with open(notebook_path, 'r', encoding='utf-8') as f:
                    notebook = nbformat.read(f, as_version=4)
                
                # Check that notebook has cells
                assert 'cells' in notebook, f"{notebook_path.name} has no cells"
                assert len(notebook.cells) > 0, f"{notebook_path.name} has zero cells"
                
                # Check nbformat version
                assert notebook.nbformat >= 4, (
                    f"{notebook_path.name} has old nbformat version: {notebook.nbformat}"
                )
                
                # Check for markdown introduction
                first_cell = notebook.cells[0]
                assert first_cell.cell_type == 'markdown', (
                    f"{notebook_path.name} should start with a markdown cell"
                )
                
                logger.info(f"✓ {notebook_path.name} has valid structure "
                          f"({len(notebook.cells)} cells)")
                
            except Exception as e:
                pytest.fail(f"Failed to validate {notebook_path.name}: {str(e)}")
    
    def test_notebooks_have_descriptions(self, all_notebooks):
        """
        Test that all notebooks have descriptive markdown in first cell.
        """
        if not NBCONVERT_AVAILABLE:
            pytest.skip("nbconvert not available")
        
        for notebook_path in all_notebooks:
            with open(notebook_path, 'r', encoding='utf-8') as f:
                notebook = nbformat.read(f, as_version=4)
            
            # Get first cell
            if len(notebook.cells) == 0:
                pytest.fail(f"{notebook_path.name} has no cells")
            
            first_cell = notebook.cells[0]
            
            # Check it's markdown
            assert first_cell.cell_type == 'markdown', (
                f"{notebook_path.name} should start with markdown description"
            )
            
            # Check it has content
            content = first_cell.source.strip()
            assert len(content) > 0, (
                f"{notebook_path.name} first cell is empty"
            )
            
            # Check it has a title (starts with #)
            assert content.startswith('#'), (
                f"{notebook_path.name} should start with a title (# ...)"
            )
            
            logger.info(f"✓ {notebook_path.name} has proper description")


@pytest.mark.integration
def test_analysis_notebook_comprehensive(notebooks_dir):
    """
    Comprehensive test for the main analysis notebook.
    
    This is a more detailed test specifically for analysis.ipynb
    since it's the entry point for users.
    """
    if not NBCONVERT_AVAILABLE:
        pytest.skip("nbconvert not available")
    
    analysis_nb = notebooks_dir / "analysis.ipynb"
    
    if not analysis_nb.exists():
        pytest.skip("analysis.ipynb not found")
    
    # Read notebook
    with open(analysis_nb, 'r', encoding='utf-8') as f:
        notebook = nbformat.read(f, as_version=4)
    
    # Check it has substantial content (at least 10 cells)
    assert len(notebook.cells) >= 10, (
        "analysis.ipynb should have at least 10 cells for comprehensive analysis"
    )
    
    # Check for key sections (by looking for markdown headers)
    markdown_cells = [cell for cell in notebook.cells if cell.cell_type == 'markdown']
    markdown_content = '\n'.join([cell.source for cell in markdown_cells])
    
    expected_sections = [
        'Load Data',
        'Data',
        'Summary',
        'Trend'
    ]
    
    for section in expected_sections:
        assert section in markdown_content, (
            f"analysis.ipynb should have a '{section}' section"
        )
    
    logger.info("✓ analysis.ipynb has comprehensive structure")


# Utility function for manual testing
def run_single_notebook(notebook_name: str):
    """
    Utility function to run a single notebook manually.
    
    Args:
        notebook_name: Name of the notebook file
        
    Example:
        >>> run_single_notebook("analysis.ipynb")
    """
    notebook_path = NOTEBOOKS_DIR / notebook_name
    
    if not notebook_path.exists():
        print(f"Notebook not found: {notebook_path}")
        return False
    
    success, error = execute_notebook(notebook_path)
    
    if success:
        print(f"✓ Successfully executed {notebook_name}")
    else:
        print(f"✗ Failed to execute {notebook_name}")
        print(f"Error: {error}")
    
    return success


if __name__ == "__main__":
    # Allow running this module directly for quick testing
    logging.basicConfig(level=logging.INFO)
    
    print("Discovering notebooks...")
    notebooks = get_all_notebooks()
    print(f"Found {len(notebooks)} notebooks")
    
    for nb in notebooks:
        print(f"\nTesting: {nb.name}")
        success, error = execute_notebook(nb)
        if not success:
            print(f"  ✗ FAILED: {error}")
        else:
            print(f"  ✓ PASSED")

