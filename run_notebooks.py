"""Sequential notebook execution with model selection."""
import argparse
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
NOTEBOOKS = (
    '01_understand_data.ipynb',
    '02_windows_and_features.ipynb',
    '03_classical_baselines.ipynb',
    '04_error_analysis.ipynb',
)
MODEL_NAMES = {'lr': 'LR', 'rf': 'RF', 'xgboost': 'XGBoost', 'mlp': 'MLP'}

def run_notebook(path, destination):
    import nbformat
    from nbclient import NotebookClient

    class ConsoleClient(NotebookClient):
        def create_kernel_manager(self):
            manager = super().create_kernel_manager()
            if os.name == 'posix':

                manager.transport = 'ipc'
                manager.ip = str(Path(ipc_directory.name) / 'socket')

            manager.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher',
                                        '-f', '{connection_file}']
            return manager

        def process_message(self, msg, cell, cell_index):
            if msg['msg_type'] == 'stream':
                content = msg['content']
                stream = sys.stderr if content.get('name') == 'stderr' else sys.stdout
                print(content.get('text', ''), end='', file=stream, flush=True)
            return super().process_message(msg, cell, cell_index)

    notebook = nbformat.read(path, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            cell.outputs = []
            cell.execution_count = None
    ipc_directory = (tempfile.TemporaryDirectory(prefix='iml-ipc-', dir='/tmp')
                     if os.name == 'posix' else None)
    client = ConsoleClient(notebook, kernel_name='python3', timeout=None,
                           resources={'metadata': {'path': str(ROOT)}})
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        client.execute()
    finally:

        try:
            nbformat.write(notebook, destination)
        finally:
            if ipc_directory is not None:
                ipc_directory.cleanup()

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--start', type=int, choices=range(1, 5), default=1,
                        help='First notebook (default: 1; 3 reuses prepared data).')
    parser.add_argument('--end', type=int, choices=range(1, 5), default=4,
                        help='Last notebook to run (default: 4).')
    parser.add_argument('--models', nargs='+', type=str.lower, choices=MODEL_NAMES,
                        default=list(MODEL_NAMES), help='Models to run (default: all four).')
    args = parser.parse_args(argv)
    if args.start > args.end:
        parser.error('--start must be less than or equal to --end')
    selected = [MODEL_NAMES[name] for name in dict.fromkeys(args.models)]
    os.environ['HAR_MODELS'] = ','.join(selected)
    destination = ROOT / 'results' / 'executed_notebooks'
    print('Models:', ', '.join(selected), flush=True)
    print('Executed notebook copies:', destination, flush=True)
    if args.start <= 2:
        print('Included preparation notebooks will recompute their saved outputs.', flush=True)
    try:
        for name in NOTEBOOKS[args.start - 1:args.end]:
            print(f'\nRunning {name}', flush=True)
            run_notebook(ROOT / 'notebooks' / name, destination / name)
            print(f'Finished {name}', flush=True)
    except KeyboardInterrupt:
        print('\nStopped. Partial notebook saved; completed model folds remain cached.', file=sys.stderr)
        return 130
    except Exception as exc:
        print(f'\nStopped: {exc}\nExecution details are saved in the notebook.', file=sys.stderr)
        return 1
    print('\nSelected notebooks completed.', flush=True)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
