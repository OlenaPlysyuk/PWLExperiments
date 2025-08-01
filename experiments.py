#!/usr/bin/env python3
#!/usr/bin/env python3
import os
import itertools
import subprocess
import pandas as pd

def get_experiment_configs():
    """
    Generate all hyperparameter combinations plus fixed breakpoints and repeat count.
    """
    repeats = 10
    grid = {
        'fix_first':       ['nofix', 'fix'],
        'in_dim':          [2, 1, 3, 4],
        'hidden':          [16, 8, 32],
        'n_hidden_layers': [3, 2, 3, 4],
        'lr':              [1e-1, 1e-2, 1e-3, 1e-4],
    }
    breakpoints_x = [-1, 0, 0.5, 1]
    breakpoints_y = [-2, 0.5, 4, -2]

    keys, values = zip(*grid.items())
    for vals in itertools.product(*values):
        cfg = dict(zip(keys, vals))
        cfg['breakpoints_x'] = breakpoints_x
        cfg['breakpoints_y'] = breakpoints_y
        cfg['repeats'] = repeats
        yield cfg

def main():
    experiments_dir = 'experiments_results'
    os.makedirs(experiments_dir, exist_ok=True)
    csv_path = os.path.join(experiments_dir, 'runs.csv')

    # Якщо лог уже є — завантажуємо, інакше — створюємо порожній
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(columns=[
            'in_dim','hidden','n_hidden_layers','lr','fix_first','seed'
        ])

    for cfg in get_experiment_configs():
        # підраховуємо, скільки запусків вже зроблено
        mask = (
            (df['in_dim']          == cfg['in_dim']) &
            (df['hidden']          == cfg['hidden']) &
            (df['n_hidden_layers'] == cfg['n_hidden_layers']) &
            (df['lr']              == cfg['lr']) &
            (df['fix_first']       == cfg['fix_first'])
        )
        completed = df.loc[mask].shape[0]
        remaining = cfg['repeats'] - completed

        if remaining <= 0:
            print(f"[in_dim={cfg['in_dim']}, hidden={cfg['hidden']}, "
                  f"layers={cfg['n_hidden_layers']}, lr={cfg['lr']}, "
                  f"fix_first={cfg['fix_first']}] – all done.")
            continue

        print(f"\n[in_dim={cfg['in_dim']}, hidden={cfg['hidden']}, "
              f"layers={cfg['n_hidden_layers']}, lr={cfg['lr']}, "
              f"fix_first={cfg['fix_first']}] – {remaining} runs remaining.")

        for run_idx in range(completed, cfg['repeats']):
            cmd = [
                'python', 'main.py',
                # breakpoints — потрібні тільки для PWLData(in_dim≠2), але
                # ми передаємо завжди, головне, щоб код прийняв їх
                '--breakpoints_x', *map(str, cfg['breakpoints_x']),
                '--breakpoints_y', *map(str, cfg['breakpoints_y']),
                # керуючі гіперпараметри
                '--in_dim',          str(cfg['in_dim']),
                '--hidden',          str(cfg['hidden']),
                '--n_hidden_layers', str(cfg['n_hidden_layers']),
                '--lr',              str(cfg['lr']),
                '--fix_first',       cfg['fix_first'],
                # фіксовані параметри тренування
                '--n_train',         '1000',
                '--n_val',           '700',
                '--n_test',          '700',
                '--n_samples',       '1000',
                '--inject',          '20',
                '--l1_lambda',       '0',
                '--optimizer',       'adam',
                '--batch',           '1024',
                '--epochs',          '7000',
                '--sched_f',         '0.1',
                '--sched_p',         '50',
                '--early_stop_patience', '1000',
                '--min_delta',       '1e-6',
                # різний seed для кожного запуску
                '--seed',            str(run_idx),
                '--outdir',          experiments_dir
            ]
            print(f"  RUN {run_idx+1}/{cfg['repeats']}: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)

            # після повернення — перечитати лог, щоб оновити df
            df = pd.read_csv(csv_path)

    print("\nAll experiments completed.")

if __name__ == '__main__':
    main()












'''
import os
import itertools
import subprocess
import pandas as pd
from pwl_data   import PWLData
from pwl_data2  import TriangularPWLData



def get_experiment_configs():
    """
    Generate all hyperparameter combinations plus fixed breakpoints and repeat count.
    """
    repeats = 10
    grid = {
        'fix_first':       ['fix', 'nofix'],
        'in_dim':          [2, 1, 3, 4],
        'hidden':          [2, 8, 32],
        'n_hidden_layers': [0, 1, 2, 3, 4],
        'lr':              [1e-1, 1e-2, 1e-3, 1e-4],
    }
    breakpoints_x = [-1, 0, 0.5, 1]
    breakpoints_y = [-2, 0.5, 4, -2]

    keys, values = zip(*grid.items())
    for vals in itertools.product(*values):
        cfg = dict(zip(keys, vals))
        cfg['breakpoints_x'] = breakpoints_x
        cfg['breakpoints_y'] = breakpoints_y
        cfg['repeats'] = repeats
        yield cfg

def main():
    experiments_dir = 'experiments_results'
    os.makedirs(experiments_dir, exist_ok=True)
    csv_path = os.path.join(experiments_dir, 'runs.csv')

    # Load existing run log or initialize empty DataFrame
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(columns=['in_dim','hidden','n_hidden_layers','lr','fix_first','seed'])

    for cfg in get_experiment_configs():
        # Count already-completed runs for this configuration
        mask = (
            (df['in_dim']          == cfg['in_dim']) &
            (df['hidden']          == cfg['hidden']) &
            (df['n_hidden_layers'] == cfg['n_hidden_layers']) &
            (df['lr']              == cfg['lr']) &
            (df['fix_first']       == cfg['fix_first'])
        )
        completed = df.loc[mask].shape[0]
        remaining = cfg['repeats'] - completed

        if remaining <= 0:
            print(f"[in_dim={cfg['in_dim']}, hidden={cfg['hidden']}, "
                  f"layers={cfg['n_hidden_layers']}, lr={cfg['lr']}, "
                  f"fix_first={cfg['fix_first']}] – all {cfg['repeats']} runs completed.")
            continue

        print(f"\n[in_dim={cfg['in_dim']}, hidden={cfg['hidden']}, "
              f"layers={cfg['n_hidden_layers']}, lr={cfg['lr']}, "
              f"fix_first={cfg['fix_first']}] – {remaining} runs remaining.")

        # Launch each missing run with seed=0 and fixed parameters
        for run_idx in range(completed, cfg['repeats']):
            cmd = [
                'python', 'main.py',
                # breakpoints
                '--breakpoints_x', *map(str, cfg['breakpoints_x']),
                '--breakpoints_y', *map(str, cfg['breakpoints_y']),
                # dynamic hyperparameters
                '--in_dim',          str(cfg['in_dim']),
                '--hidden',          str(cfg['hidden']),
                '--n_hidden_layers', str(cfg['n_hidden_layers']),
                '--lr',              str(cfg['lr']),
                '--fix_first',       cfg['fix_first'],
                # fixed training parameters
                '--n_train',         '1000',
                '--n_val',           '700',
                '--n_test',          '700',
                '--n_samples',       '1000',
                '--inject',          '20',
                '--l1_lambda',       '0',
                '--optimizer',       'adam',
                '--batch',           '1024',
                '--epochs',          '5000',
                '--sched_f',         '0.1',
                '--sched_p',         '50',
                '--early_stop_patience', '600',
                '--min_delta',       '1e-6',
                # seed and output directory
                '--seed',            '0',
                '--outdir',          experiments_dir
            ]
            print(f"  RUN {run_idx+1}/{cfg['repeats']}: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            # reload the log to update completed count
            df = pd.read_csv(csv_path)

    print("\nAll experiments completed.")

if __name__ == '__main__':
    main()
'''