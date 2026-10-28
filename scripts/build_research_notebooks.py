"""Build teaching notebooks from their reviewable percent-format scripts.

No experiments run here. Existing executed notebooks are only replaced when
--replace is supplied. Notebook cell IDs are deterministic for clean diffs.
"""
import argparse
import hashlib
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NAMES = ('01_motivation_analytic', '02_navigation_am_and_candidates',
         '02_flow_am_terminal_gradient', '02_flow_am_dimension_coupling',
         '02_qam_td_terminal_error', '02b_joint_mc_reference_diagnostic',
         '02_qam_point_obstacle')
CURRENT = ('01_motivation_analytic', '02_qam_td_terminal_error',
           '02b_joint_mc_reference_diagnostic')


def cells_from_script(path):
    kind, lines, cells = None, [], []

    def append():
        if kind is None:
            return
        source = '\n'.join(lines).strip()
        if kind == 'markdown':
            source = '\n'.join(line[2:] if line.startswith('# ') else
                               '' if line == '#' else line for line in source.splitlines())
        cell = (nbformat.v4.new_markdown_cell(source) if kind == 'markdown'
                else nbformat.v4.new_code_cell(source))
        cell.id = hashlib.sha256(f'{len(cells)}:{source}'.encode()).hexdigest()[:12]
        cells.append(cell)

    for line in Path(path).read_text().splitlines():
        if line.startswith('# %%'):
            append()
            kind = 'markdown' if '[markdown]' in line else 'code'
            lines = []
        elif kind is not None:
            lines.append(line)
    append()
    return cells


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replace', action='store_true')
    parser.add_argument('--name', choices=NAMES, action='append')
    args = parser.parse_args()
    for name in args.name or CURRENT:
        source = ROOT/'notebooks'/f'{name}.py'
        target = source.with_suffix('.ipynb')
        if target.exists() and not args.replace:
            raise FileExistsError(f'{target.name} exists; use --replace to clear its outputs.')
        notebook = nbformat.v4.new_notebook(cells=cells_from_script(source))
        notebook.metadata.update(
            kernelspec=dict(display_name='Python 3 (ipykernel)', language='python', name='python3'),
            language_info=dict(name='python'),
            pagr=dict(source=source.name, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
        nbformat.validate(notebook)
        nbformat.write(notebook, target)
        print(f'Built {target.name}: {len(notebook.cells)} cells')


if __name__ == '__main__':
    main()
