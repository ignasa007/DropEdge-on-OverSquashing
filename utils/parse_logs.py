from collections import defaultdict


def parse_configs(fn):

    with open(fn, 'r') as f:
        configs = dict()
        for line in f.readlines():
            line = line.strip()
            if not line:
                continue
            elif 'Epoch' in line:
                break
            config, value = line.split(' = ')
            configs[config] = value

    return configs


def parse_metrics(fn):

    results = {
        'Training': defaultdict(list),
        'Validation': defaultdict(list),
        'Testing': defaultdict(list),
    }

    with open(fn, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if 'Epoch' in line:
                epoch = int(line.rsplit(' ', maxsplit=1)[1])
            elif any((key in line for key in results.keys())):
                split, metrics = line.split(': ')
                metrics = metrics.strip().split(', ')
                metrics = map(lambda x: x.split(' = '), metrics)
                results[split]['Epoch'].append(epoch)
                for metric, value in metrics:
                    results[split][metric].append(float(value))

    return dict(results['Training']), dict(results['Validation']), dict(results['Testing'])