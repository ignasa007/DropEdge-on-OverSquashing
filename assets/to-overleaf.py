import os
import shutil

assets_dir = './sensitivity/model-store'

# for dataset in os.listdir(assets_dir):
#     dataset_dir = f'{assets_dir}/{dataset}'
#     for fn in os.listdir(dataset_dir):
#         fn = f'{dataset_dir}/{fn}'
#         if os.path.isfile(fn):
#             os.remove(fn)

# for dataset in os.listdir(assets_dir):
#     dataset_dir = f'{assets_dir}/{dataset}'
#     for which in os.listdir(dataset_dir):
#         which_dir = f'{dataset_dir}/{which}'
#         for metric in os.listdir(which_dir):
#             src = f'{which_dir}/{metric}'
#             dst = f"{which_dir}/{'-'.join(metric.lower().split())}"
#             if src != dst:
#                 if ' ' in src and os.path.isdir(dst):
#                     shutil.rmtree(dst)
#                 os.rename(src, dst)

import os
import shutil

dirs = os.listdir('.')

for dir in dirs:

    if dir == '__pycache__':
        continue

    src_names = [
        f'{dirpath}/{f}'.replace('\\', '/')
        for (dirpath, _, filenames) in os.walk(dir) 
        for f in filenames
    ]

    dst_names = [
        f"overleaf/{f.replace(' ', '-').replace('_', '-').replace('/', '_')}"
        for f in src_names
    ]

    for src_name, dst_name in zip(src_names, dst_names):
        
        if os.path.isfile(dst_name):
            print(dst_name)
            os.remove(dst_name)
        os.makedirs(os.path.dirname(dst_name), exist_ok=True)
        shutil.copy2(src_name, dst_name)