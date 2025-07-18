#!/usr/bin/env python3
import os
import argparse
import csv                              # for logging runs
import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from pwl_data import PWLData
from pwl_model import NetPWL
import train_eval as u

def main():
    parser = argparse.ArgumentParser(
        description="PWL: generate → train → analyze → plot"
    )
    parser.add_argument('--breakpoints_x', type=float, nargs='+', required=True)
    parser.add_argument('--breakpoints_y', type=float, nargs='+', required=True)
    parser.add_argument('--in_dim', type=int, default=1)
    parser.add_argument('--n_train', type=int, default=1000)
    parser.add_argument('--n_val', type=int, default=200)
    parser.add_argument('--n_test', type=int, default=200)
    parser.add_argument('--n_samples', type=int, default=None,
                        help='number of samples for plotting')
    parser.add_argument('--inject', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--hidden', type=int, default=16)
    parser.add_argument('--n_hidden_layers', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--sched_f', type=float, default=0.8)
    parser.add_argument('--sched_p', type=int, default=10)
    parser.add_argument('--l1_lambda', type=float, default=0.01)
    parser.add_argument('--early_stop_patience', type=int, default=20)
    parser.add_argument('--min_delta', type=float, default=1e-6)
    parser.add_argument('--optimizer', choices=['adam','sgd','rmsprop','adagrad','adadelta'], default='adam')
    parser.add_argument('--fix_first', choices=['fix','nofix'], default='fix',
                        help='fix first PWL layer to true slopes or allow training')
    parser.add_argument('--outdir', type=str, default='results')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # datasets & loaders
    train_ds = PWLData(args.breakpoints_x, args.breakpoints_y,
                       n_samples=args.n_train, inject=args.inject,
                       seed=args.seed, in_dim=args.in_dim)
    val_ds   = PWLData(args.breakpoints_x, args.breakpoints_y,
                       n_samples=args.n_val, inject=0,
                       seed=args.seed+1, in_dim=args.in_dim)
    test_ds  = PWLData(args.breakpoints_x, args.breakpoints_y,
                       n_samples=args.n_test, inject=0,
                       seed=args.seed+2, in_dim=args.in_dim)
    loaders = [
        DataLoader(train_ds, batch_size=args.batch, shuffle=True),
        DataLoader(val_ds,   batch_size=args.batch, shuffle=False),
        DataLoader(test_ds,  batch_size=args.batch, shuffle=False)
    ]

    # optional dataset for plotting
    plot_ds = None
    if args.n_samples is not None:
        plot_ds = PWLData(args.breakpoints_x, args.breakpoints_y,
                          n_samples=args.n_samples, inject=args.inject,
                          seed=args.seed+3, in_dim=args.in_dim)

    # model, optimizer, scheduler
    fix_flag = (args.fix_first == 'fix')
    model = NetPWL(args.in_dim, args.hidden, args.n_hidden_layers,
                   train_ds.slopes, train_ds.intercepts,
                   fix_first_layer=fix_flag).to(device)
    optim_map = {
        'adam': optim.Adam,
        'sgd': optim.SGD,
        'rmsprop': optim.RMSprop,
        'adagrad': optim.Adagrad,
        'adadelta': optim.Adadelta
    }
    optimizer = optim_map[args.optimizer](model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=args.sched_f,
        patience=args.sched_p
    )

    # train & restore best
    model, history = u.train_model(model, loaders, optimizer, scheduler,
                                   args.epochs, args.l1_lambda,
                                   args.early_stop_patience, args.min_delta,
                                   args.outdir)

    # evaluation on train and test sets (capture metrics for logging)
    print("Train set evaluation:")
    train_mse, train_max_err = u.evaluate_model(model, loaders[0])
    print("Test set evaluation:")
    test_mse, test_max_err = u.evaluate_model(model, loaders[2])

    # --- CSV LOGGING ---
    log_path = os.path.join(args.outdir, 'runs.csv')
    is_new = not os.path.isfile(log_path)

    # determine run_id (1-based)
    if is_new:
        run_id = 1
    else:
        with open(log_path, 'r', newline='') as f_in:
            run_id = sum(1 for _ in f_in)

    # prepare argument names and values (convert lists to space-separated strings)
    args_dict  = vars(args)
    arg_names  = list(args_dict.keys())
    arg_values = []
    for k in arg_names:
        v = args_dict[k]
        if isinstance(v, list):
            arg_values.append(' '.join(map(str, v)))
        else:
            arg_values.append(str(v))

    # compute additional metrics from training history
    best_val_loss    = min(history['val'])
    best_epoch       = history['val'].index(best_val_loss) + 1
    final_train_loss = history['train'][-1]
    final_val_loss   = history['val'][-1]

    # write header if new, then append row
    with open(log_path, 'a', newline='') as f_out:
        writer = csv.writer(f_out)
        if is_new:
            header = (
                ['run_id'] + arg_names +
                ['best_val_loss', 'best_epoch',
                 'final_train_loss', 'final_val_loss',
                 'train_mse', 'train_max_err',
                 'test_mse', 'test_max_err']
            )
            writer.writerow(header)
        row = (
            [run_id] + arg_values +
            [f"{best_val_loss:.6e}", best_epoch,
             f"{final_train_loss:.6e}", f"{final_val_loss:.6e}",
             f"{train_mse:.6e}", f"{train_max_err:.6e}",
             f"{test_mse:.6e}", f"{test_max_err:.6e}"]
        )
        writer.writerow(row)
    print(f"Logged run #{run_id} → {log_path}")
    # --- end CSV LOGGING ---

    # plotting
    if args.in_dim == 1:
        ds = plot_ds if plot_ds is not None else train_ds
        u.plot_1d(ds, args.outdir)
    elif args.in_dim == 2:
        ds = plot_ds if plot_ds is not None else train_ds
        u.plot_2d(ds, args.outdir)
    print(f"Saved 2D plot to {args.outdir}")

    # activation patterns
    #u.analyze_activation_patterns(model, train_ds)
    #print('End of activation patterns')
    # save model
    #print("Before saving")
    torch.save(model.state_dict(), os.path.join(args.outdir, 'model.pt'))
    #print("Saved model")
if __name__ == '__main__':
    main()
