import numpy as np
from tabulate import tabulate
from .datasets import load_dataset

class Game:
    def __init__(self, model, baseline, explicand):
        self.model = model
        self.baseline = baseline
        self.explicand = explicand
    
    def value(self, S):
        # S is a m by n binary matrix
        inputs = self.baseline * (1 - S) + self.explicand * S
        return self.model.predict(inputs)
    
    def edge_cases(self):
        v0 = self.model.predict(self.baseline)
        v1 = self.model.predict(self.explicand)
        return v0, v1


def fancy_round(x, precision=3):
    return float(np.format_float_positional(x, precision=precision, unique=False, fractional=False, trim='k'))

def benchmark_table(results, filename=None, print_md=True, include_color=True):
    table = []
    for method in results:
        row = [method]
        values = results[method][list(results[method].keys())[0]]
        mean = np.mean(values)
        median = np.median(values)
        upper = np.percentile(values, 75)
        lower = np.percentile(values, 25)
        to_add = [mean, lower, median, upper]
        row += [fancy_round(x) for x in to_add]
        table.append(row)

    if print_md:
        print(tabulate(table, headers=['Method', 'Mean', '1st Quartile', '2nd Quartile', '3rd Quartile'], tablefmt="github"))    

    cols = []
    for i in range(1,len(table[0])):
        vals = [row[i] for row in table]
        cols += [sorted(vals)]
    if filename is not None:
        with open(filename, 'w') as f:
            f.write('\\begin{tabular}{lllll}\n')
            f.write('  \\toprule\n')
            f.write('  \\textbf{Method} & \\textbf{Mean} & \\textbf{1st Quartile} & \\textbf{2nd Quartile} & \\textbf{3rd Quartile} \\\\ \\midrule \n')

    for row in table:
        print_row = [row[0]]
        for idx in range(1, len(row)):
            color = ''
            if include_color:
                if row[idx] == cols[idx-1][0]:
                    color = '\\cellcolor{gold!60}'
                elif row[idx] == cols[idx-1][1]:
                    color = '\\cellcolor{silver!60}'
                elif row[idx] == cols[idx-1][2]:
                    color = '\\cellcolor{bronze!60}'
            val = "{:.2e}".format(row[idx])
            print_row.append(f'{color}{val}')

        to_print = ' & '.join(print_row) + r'\\'
        if filename is not None:
            with open(filename, 'a') as f:
                f.write(to_print + '\n')
    if filename is not None:
        with(open(filename, 'a')) as f:
            f.write('\\bottomrule\n')
            f.write('\\end{tabular}')

def one_big_table(results, filename, error_type):
    # Each column is a dataset
    # There are several groups of rows: one for each method
    # Each group has 4 rows: mean, 1st quartile, 2nd quartile, 3rd quartile
    num_methods = len(results[list(results.keys())[0]])
    table = np.zeros((num_methods*4, len(results)))
    for i, dataset in enumerate(results):
        for j, method in enumerate(results[dataset]):
            values = np.array(results[dataset][method][list(results[dataset][method].keys())[0]])
            if error_type == 'weighted_error': values = 1 - values
            mean = np.mean(values)
            median = np.median(values)
            upper = np.percentile(values, 75)
            lower = np.percentile(values, 25)
            to_add = np.array([mean, lower, median, upper])
            table[j*4:(j+1)*4, i] = to_add
    with open(filename, 'w') as f:
        f.write('\\resizebox{\\linewidth}{!}{ \n')
        f.write('\\begin{tabular} {l'+ 'c'*len(results) + '}\n')
        f.write('\\toprule\n')
        f.write(' & ' + ' & '.join([f'\\textbf{{{dataset}}}' for dataset in results]) + ' \\\\ \n')
        f.write('\\midrule\n')
        i = 0
        for method in results[dataset]:
            f.write('\\addlinespace[1ex] \n')
            f.write(f'\\textbf{{{method}}}' + ' & ' * len(results) + ' \\\\ \n')
            for metric in ['Mean', '1st Quartile', '2nd Quartile', '3rd Quartile']:
                row = [] 
                for j in range(len(results)):
                    # Color the best, second, third values with gold, silver, bronze
                    # Select every 4th row

                    relevant_col = sorted(table[(i%4)::4,j])
                    color = ''
                    if table[i,j] == relevant_col[0]:
                        color = '\\cellcolor{gold!60}'
                    elif table[i,j] == relevant_col[1]:
                        color = '\\cellcolor{silver!60}'
                    elif table[i,j] == relevant_col[2]:
                        color = '\\cellcolor{bronze!60}'
                    row += [color + str(fancy_round(table[i,j]))]
                start = '\\hspace{7pt}' + metric + ' & '
                f.write(start + ' & '.join([str(x) for x in row]) + ' \\\\ \n')
                i += 1
        f.write('\\bottomrule\n')
        f.write('\\end{tabular}}')            
        
def error_ratio_table(results, multipliers, filename, numerator='Leverage SHAP', denominator='Optimized Kernel SHAP'):
    # NEW (not in 0de0a80): reproduces overleaf_paper's Table \ref{tab:error2kshap}
    # (Appendix "Error Relative to Optimized Kernel SHAP"): one row per
    # dataset, one column per sample-size multiplier m/n, each cell the
    # ratio of `numerator`'s mean shap_error to `denominator`'s at that
    # sample size. `results` is exactly what `ls.load_results(datasets,
    # 'sample_size', 'shap_error', {'noise': 0}, ...)` returns, i.e.
    # results[dataset][estimator][sample_size] -> list of shap_error values
    # across runs (unconstrained sample_size, so every recorded multiplier
    # is present, keyed by the actual int sample size).
    #
    # Per the paper's caption: once m >= 2^n every non-trivial coalition is
    # enumerated, so both estimators recover Shapley values exact to machine
    # precision and the ratio is defined to be exactly 1 (computing it from
    # the data directly would instead divide two near-zero floating point
    # errors and produce meaningless noise). This threshold is applied
    # explicitly below rather than inferred from the data, and matches the
    # published table exactly (verified against overleaf_paper/tables/
    # error ratios: e.g. California n=8 is 1.00 starting at the 40n=320>=256
    # column, Diabetes n=10 only at 160n=1600>=1024, Adult n=12 never
    # reaches 1.00 since even 160n=1920 < 4096).
    rows = []
    for dataset in results:
        X, y = load_dataset(dataset)
        n = X.shape[1]
        row = [dataset]
        for mult in multipliers:
            sample_size = int(mult * n)
            if sample_size >= 2**n:
                ratio = 1.0
            else:
                try:
                    num_vals = results[dataset][numerator][sample_size]
                    den_vals = results[dataset][denominator][sample_size]
                    ratio = np.mean(num_vals) / np.mean(den_vals)
                except KeyError:
                    ratio = float('nan')
            row.append(ratio)
        rows.append(row)

    # The paper's caption quotes one pooled scalar ("averaging over all non-unit
    # entries, Leverage SHAP achieved 50.2% of the error"); write it (and the
    # per-column version) to a sidecar text file next to the table so the
    # caption numbers can be regenerated without editing the table itself.
    non_unit = [v for row in rows for v in row[1:] if not np.isnan(v) and not np.isclose(v, 1.0)]
    with open(filename.replace('.tex', '_summary.txt'), 'w') as f:
        f.write(f'pooled mean of non-unit ratios: {np.mean(non_unit):.4f} over {len(non_unit)} entries\n')
        f.write(f'pooled median of non-unit ratios: {np.median(non_unit):.4f}\n')
        f.write(f'min / max non-unit ratio: {np.min(non_unit):.4f} / {np.max(non_unit):.4f}\n')
        for col, mult in enumerate(multipliers, start=1):
            col_vals = [row[col] for row in rows if not np.isnan(row[col]) and not np.isclose(row[col], 1.0)]
            f.write(f'{mult}n: mean of non-unit ratios {np.mean(col_vals):.4f} over {len(col_vals)} datasets\n')
        for row in rows:
            f.write(row[0] + ': ' + ', '.join(f'{v:.4f}' for v in row[1:]) + '\n')

    def fmt(v):
        return '--' if np.isnan(v) else str(fancy_round(v))

    with open(filename, 'w') as f:
        f.write('\\begin{tabular}{l' + 'c' * len(multipliers) + '}\n')
        f.write('\\toprule\n')
        f.write('Dataset & ' + ' & '.join([f'${m}n$' for m in multipliers]) + ' \\\\\n')
        f.write('\\midrule\n')
        for row in rows:
            f.write(row[0] + ' & ' + ' & '.join(fmt(v) for v in row[1:]) + ' \\\\\n')
        f.write('\\bottomrule\n')
        f.write('\\end{tabular}\n')
