import pandas as pd
from pathlib import Path

THRESHOLDS = [0.25, 0.50, 0.75]
TAG_CHUNK_SIZES = [250, 500, 1000]

map_reduce = pd.read_csv(Path('results/map_reduce/results.csv'))
map_reduce = map_reduce.set_index('test_file')
results = pd.DataFrame()

for threshold in THRESHOLDS:
    for size in TAG_CHUNK_SIZES:
        try:
            tag = pd.read_csv(Path(f'results/tag_{threshold}_{size}_map_reduce/results.csv'))
            tag = tag.set_index('test_file')
            tag_diff = tag - map_reduce
            tag_diff['tokens_used'] = tag_diff['tokens_used'] / map_reduce['tokens_used']
            tag_diff = tag_diff.dropna()
            # print(f'{threshold}_{size}', len(tag_diff))
            results[f'{threshold}_{size}'] = tag_diff.mean()
        except:
            pass

results.index.name = 'metric'
print(results)
results.to_csv('tag_diff.csv')