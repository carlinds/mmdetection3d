import pandas as pd

latex_table = """
& $32.2$          & $39.8$          & $100.0$          & $38.6$          & $43.1$          & $100.0$         & $38.4$          & $48.5$
& $13.5$          & $28.8$          & $46.3$           & $20.2$          & $31.6$          & $55.4$          & $29.1$          & $42.7$
& $0.00$          & $0.00$          & $0.00$           & $0.00$          & $0.00$          & $0.00$          & $0.00$          & $0.00$
& $32.5$          & $40.0$          & $100.0$          & $34.0$          & $38.9$          & $100.0$         & $38.9$          & $48.6$
& $13.5$          & $28.9$          & $46.5$           & $20.4$          & $30.0$          & $57.6$          & $31.0$          & $44.0$
& $0.00$          & $0.00$          & $0.00$           & $0.00$          & $0.00$          & $0.00$          & $0.00$          & $0.00$
& $31.2$          & $38.6$          & $100.0$          & $35.1$          & $40.0$          & $100.0$         & $38.5$          & $48.3$
& $23.5$          & $33.6$          & $58.7$           & $29.3$          & $37.3$          & $70.7$          & $31.7$          & $44.5$
& $0.00$          & $0.00$          & $0.00$           & $0.00$          & $0.00$          & $0.00$          & $0.00$          & $0.00$
& $32.5$          & $39.8$          & $100.0$          & $31.4$          & $37.2$          & $100.0$         & $37.5$          & $48.1$
& $24.5$          & $34.3$          & $57.3$           & $26.1$          & $35.1$          & $67.9$          & $33.0$          & $44.9$
& $0.00$          & $0.00$          & $0.00$           & $0.00$          & $0.00$          & $0.00$          & $0.00$          & $0.00$
"""

latex_table = latex_table.replace('&', '')
latex_table = latex_table.replace('$', '')
rows = latex_table.split('\n')
data = [[float(x) for x in row.split()] for row in rows if row]

df = pd.DataFrame(data)
ref_values_real = df.iloc[0, :]
ref_values_sim = df.iloc[1, :]

# For every third row, compute the gap as 1 - (the row before / ref_values_real)
for i in range(2, len(data), 3):
    row = df.iloc[i - 1, :]
    gap = round((1 - row / ref_values_real) * 100, 1)
    df.iloc[i, :] = gap

print(df)

new_latex_table = """"""
for i in range(df.shape[0]):
    new_latex_table += ' & '.join([f'${x:.1f}$' for x in df.iloc[i, :]]) + '\n'

print(new_latex_table)
