"""Execute the portable notebook in this Python environment, fail on any error."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('runs/notebooks/reproduce.executed.ipynb'))
    parser.add_argument('--replace', action='store_true', help='Explicitly allow replacing a notebook output')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if output.exists() and not args.replace:
        raise FileExistsError('Choose a new output path or explicitly pass --replace.')
    notebook = nbformat.read(root/'notebooks/reproduce.ipynb', as_version=4)
    # Use a temporary kernelspec pointing to this exact interpreter. Do not
    # mutate the user's Jupyter installation or accidentally run another env.
    with tempfile.TemporaryDirectory(prefix='pagr-kernel-') as directory:
        spec = Path(directory)/'pagr-current'
        spec.mkdir()
        (spec/'kernel.json').write_text(json.dumps({
            'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
            'display_name': 'PAGR current interpreter', 'language': 'python'}))
        manager = KernelManager(kernel_name='pagr-current',
                                kernel_spec_manager=KernelSpecManager(kernel_dirs=[directory]))
        try:
            NotebookClient(notebook, km=manager, timeout=300, allow_errors=False,
                           resources={'metadata': {'path': str(root)}}).execute()
        finally:
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, output)
    print(json.dumps({'executed_code_cells': sum(c.cell_type=='code' for c in notebook.cells),
                      'error_outputs': sum(o.output_type=='error' for c in notebook.cells if c.cell_type=='code' for o in c.outputs)}))


if __name__=='__main__':
    main()
