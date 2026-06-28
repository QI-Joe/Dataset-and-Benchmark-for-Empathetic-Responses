import sys
import os
import pickle
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple
from concurrent.futures import ProcessPoolExecutor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd

FIXED_DATA_PATH = r'./pre data/'
name_key_match = {
    'sbzjs': '上班这件事',
    'rswtyjs': '人生问题研究社',
    'rjqlgc': '人间情侣观察',
    'zctytlxz': '职场体验讨论小组',
    'zctcdh': '职场吐槽大会',
}


# ─── Raw CSV Loading ──────────────────────────────────────────────────────────

def load_gen_data(data_dir_name: str):
    """Load the three raw CSVs for a given forum dataset key.

    Returns:
        d_c2r       — (postID, comment, review)  rows that have a review
        d_c2l       — (postID, comment, label)   all comments with binary label
        d_c2rawtext — (postID, postkey)           post bodies
    """
    value, key = name_key_match[data_dir_name], data_dir_name

    comment2review  = f"{FIXED_DATA_PATH}{value}/{key}_id_c_r.csv"
    comment2label   = f"{FIXED_DATA_PATH}{value}/{key}_id_c_l.csv"
    comment2rawtext = f"{FIXED_DATA_PATH}{value}/{key}_postid_pre_final.csv"

    d_c2r       = pd.read_csv(comment2review)
    d_c2l       = pd.read_csv(comment2label)
    d_c2rawtext = pd.read_csv(comment2rawtext)

    return d_c2r, d_c2l, d_c2rawtext


# ─── Parallel chunk worker (top-level required for multiprocessing pickling) ──

def _process_chunk(
    chunk_c2rt: pd.DataFrame,
    c2r_by_post_dict: dict,
    c2l_by_post_dict: dict,
) -> Tuple[List[Tuple], dict]:
    """Process a slice of postID rows in a worker process.

    Uses local 0-based indices; caller rebases them after merging chunks.
    """
    all_data_local: List[Tuple] = []
    block_map_local: dict = defaultdict(list)
    local_idx = 0

    for _, row in chunk_c2rt.iterrows():
        pid, post_content = row['postID'], row['postkey']

        if pid not in c2r_by_post_dict or pid not in c2l_by_post_dict:
            continue

        c2r_dict = c2r_by_post_dict[pid].set_index('comment')['review'].to_dict()
        review_repeat_detection = set(c2r_dict.values())

        for _, c2l_row in c2l_by_post_dict[pid].iterrows():
            comment = c2l_row['comment']
            if comment in review_repeat_detection:
                continue
            label  = c2l_row['label']
            review = c2r_dict.get(comment, '') if label else ''
            all_data_local.append((pid, post_content, comment, review, label))
            block_map_local[pid].append(local_idx)
            local_idx += 1

    return all_data_local, block_map_local


# ─── Standalone builder ───────────────────────────────────────────────────────

def build_input_c2r(data_name: str) -> Tuple[List[Tuple], Dict]:
    """Parse raw CSVs into flat sample tuples and a postID → index mapping.

    Checks for a pickle cache at ``{FIXED_DATA_PATH}{value}/{key}_cache.pkl``
    and returns immediately on a hit.  On a miss, processes postID chunks in
    parallel via ``ProcessPoolExecutor(os.cpu_count())`` then persists the
    merged result to the same cache path.

    Each tuple: (pid, post_content, comment, review, label)
        pid          — postID (int)
        post_content — full post text (str)
        comment      — user comment (str)
        review       — author reply, empty string when label==0 (str)
        label        — 1 if author replied, 0 otherwise

    Returns:
        all_data  — list of the tuples above
        block_map — defaultdict(list) mapping pid → list of indices into all_data

    Note: ``global_idx`` is accepted for backward compatibility but ignored;
    indices always start at 0 and are rebased internally after chunk merging.
    """
    value      = name_key_match[data_name]
    cache_path = f"{FIXED_DATA_PATH}{value}/{data_name}_cache.pkl"

    if os.path.exists(cache_path):
        print(f"[build_input_c2r] Cache hit → {cache_path}")
        with open(cache_path, 'rb') as fh:
            return pickle.load(fh)

    print(f"[build_input_c2r] Cache miss — building '{data_name}' in parallel …")
    c2r, c2l, c2rt = load_gen_data(data_name)

    # Convert groupby to plain {pid: DataFrame} dicts — safe for multiprocessing pickling
    c2r_by_post_dict = {pid: grp.reset_index(drop=True) for pid, grp in c2r.groupby('postID')}
    c2l_by_post_dict = {pid: grp.reset_index(drop=True) for pid, grp in c2l.groupby('postID')}

    all_pids   = c2rt['postID'].unique()
    n_workers  = os.cpu_count() or 4
    pid_chunks = np.array_split(all_pids, n_workers)

    # Build per-chunk argument tuples (only ship relevant sub-dicts per worker)
    args_list = []
    for pid_chunk in pid_chunks:
        if len(pid_chunk) == 0:
            continue
        chunk_c2rt = c2rt[c2rt['postID'].isin(pid_chunk)].reset_index(drop=True)
        c2r_sub    = {pid: c2r_by_post_dict[pid] for pid in pid_chunk if pid in c2r_by_post_dict}
        c2l_sub    = {pid: c2l_by_post_dict[pid] for pid in pid_chunk if pid in c2l_by_post_dict}
        args_list.append((chunk_c2rt, c2r_sub, c2l_sub))

    all_data: List[Tuple] = []
    block_map: Dict       = defaultdict(list)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(_process_chunk, *args) for args in args_list]
        for future in futures:
            chunk_data, chunk_bmap = future.result()
            offset = len(all_data)
            all_data.extend(chunk_data)
            for pid, idxs in chunk_bmap.items():
                block_map[pid].extend(offset + i for i in idxs)

    print(f"[build_input_c2r] Done — {len(all_data)} samples, {len(block_map)} posts. Saving cache …")
    with open(cache_path, 'wb') as fh:
        pickle.dump((all_data, block_map), fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[build_input_c2r] Cache saved → {cache_path}")

    return all_data, block_map

if __name__ == "__main__":
    # test_name = 'zctcdh'
    for name in name_key_match.keys():
        all_data, block_map = build_input_c2r(name)
        print(f"Sample output for '{name}':")
        for i in range(3):
            print(f"  --- Sample {i} ---")
            print(f"  postID      : {all_data[i][0]}")
            print(f"  post_content: {all_data[i][1][:80]!r}")
            print(f"  comment      : {all_data[i][2][:80]!r}")
            print(f"  review       : {all_data[i][3][:80]!r}")
            print(f"  label        : {all_data[i][4]}")
            print()
