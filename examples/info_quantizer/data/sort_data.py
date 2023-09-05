import os
import sys
from pathlib import Path

def sort(file):
    with open(file, 'r') as fd:
        lines = fd.read().splitlines()
    
    sources = {}
    actions = {}
    targets = {}
    for line in lines:
        if line.startswith('S-'):
            id = int(line.split('\t')[0].split('-')[1])
            sources[id] = line.split('\t')[1]
        elif line.startswith('T-'):
            id = int(line.split('\t')[0].split('-')[1])
            targets[id] = line.split('\t')[1]
        elif line.startswith('A-'):
            id = int(line.split('\t')[0].split('-')[1])
            actions[id] = line.split('\t')[1]
        else:
            print(line)
    
    assert len(list(sources.keys())) == len(list(targets.keys()))
    assert len(list(targets.keys())) == len(list(actions.keys()))
    sort_ids = sorted(list(actions.keys()))
    source = [sources[_id] for _id in sort_ids]
    action = [actions[_id] for _id in sort_ids]
    action = [a.replace('4', '0') for a in action]
    action = [a.replace('5', '1') for a in action]
    target = [targets[_id] for _id in sort_ids]

    return source, action, target

if __name__ == '__main__':
    file = sys.argv[1]
    prefix = file.split('/')[-1].replace('.txt', '')
    
    output_path = Path('/'.join(file.split('/')[:-1]))
    output_path = output_path / 'sorted'
    if not output_path.exists():
        os.mkdir(output_path)

    sources, actions, targets = sort(file)

    src_path = output_path / f"{prefix}.src"
    if src_path.exists():
        os.remove(src_path)
    with open(src_path, 'w') as sf:
        sf.write('\n'.join(sources))

    tgt_path = output_path / f"{prefix}.trg"
    if tgt_path.exists():
        os.remove(tgt_path)
    with open(tgt_path, 'w') as tf:
        tf.write('\n'.join(targets))

    act_path = output_path / f"{prefix}.act"
    if act_path.exists():
        os.remove(act_path)
    with open(act_path, 'w') as af:
        af.write('\n'.join(actions))