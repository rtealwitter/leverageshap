import numpy as np
from tabulate import tabulate

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
        
def one_big_table_old(results, filename, include_color=True):
    # Each row is a dataset
    # There are five groups of columns: one for each method
    # Each group has 4 columns: mean, 1st quartile, 2nd quartile, 3rd quartile
    table = []
    for dataset in results:
        row = [dataset]
        for method in results[dataset]:
            values = np.array(results[dataset][method][list(results[dataset][method].keys())[0]])
            mean = np.mean(values)
            median = np.median(values)
            upper = np.percentile(values, 75)
            lower = np.percentile(values, 25)
            to_add = [mean, lower, median, upper]
            row += [fancy_round(x) for x in to_add]
        table.append(row)

    with open(filename, 'w') as f:
        f.write('\\resizebox{\\linewidth}{!}{ \n')
        f.write('\\begin{tabular} {lcccc||cccc||cccc||cccc||cccc||cccc}\n')
        f.write('\\toprule\n')
        f.write('& \\multicolumn{4}{c}{\\textbf{Kernel SHAP}} & \\multicolumn{4}{c}{\\textbf{Kernel SHAP Paired}} & \\multicolumn{4}{c}{\\textbf{Optimized Kernel SHAP}} & \\multicolumn{4}{c}{\\textbf{Leverage SHAP}} & \\multicolumn{4}{c}{\\textbf{Leverage SHAP Paired}} \\\\ \n')
        cols = ['Mean', '1st', '2nd', '3rd']
        colnames = ' & '.join(['\\textbf{' + col + '}' for col in cols])
        f.write(f'\\textbf{{Approach}} & {colnames} & {colnames} & {colnames} & {colnames} & {colnames} \\\\ \\midrule \n')
        for row in table:
            print_row = [row[0]]
            for i in range(1, len(row)):
                color = ''
                if include_color:
                    vals = [row[j] for j in range(1, len(row))]
                    if row[i] == min(vals):
                        color = '\\cellcolor{gold!60}'
                    elif row[i] == max(vals):
                        color = '\\cellcolor{silver!60}'
                val = "{:.2e}".format(row[i])
                print_row.append(f'{color}{val}')
            to_print = ' & '.join(print_row) + r'\\'
            f.write(to_print + '\n')
        f.write('\\bottomrule\n')
        f.write('\\end{tabular}}')
