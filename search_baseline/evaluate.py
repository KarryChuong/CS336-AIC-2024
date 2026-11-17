"""Danh gia he thong truy van tren bo cau hoi co dap an.

File cau hoi (.jsonl), moi dong 1 cau:
    {"id": "q1", "query": "a man in a red shirt ...", "video": "L01_V001", "start": 1200, "end": 1500}
    {"id": "q2", "query": "...", "query_b": "...", "window": 30, "video": "L01_V003", "start": 900, "end": 900}
  - start/end: khoang frame_idx dung (bao gom 2 dau). Nut "Luu lam GT" tren UI ghi dung dinh dang nay.
  - query_b (tuy chon): truy van thoi gian A -> B.

Mot ket qua la DUNG neu cung video va frame_idx nam trong [start - tol, end + tol] (tol = --tol giay).

    python evaluate.py --index <data>/index --queries queries/gt_ui.jsonl --out runs/baseline.csv
    python evaluate.py --index <data>/index --queries runs/gt/gt.jsonl --translate --out runs/b32.csv

--translate: gt.jsonl tu collect_gt van la TIENG VIET -> dich sang tieng Anh truoc khi tim (giong o
"Dich Viet->Anh" tren UI). Ban dich luu cache (queries/translate_cache.json), chay lai khong goi mang.

--caption <index tit> (build_caption_index.py): cong diem khop nghia voi dong tit. --caption-weight nhan
nhieu gia tri -> do lan luot trong 1 lan chay (chi tai model 1 lan), vd --caption-weight 0 0.3 0.6 1
"""
import argparse
import json
import os
import re

import numpy as np
import pandas as pd

from engine import SearchEngine
from video import sample_frames

KS = (1, 5, 10, 20, 50, 100)
VI_CHARS = re.compile(r'[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', re.I)


def load_queries(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip() and not l.lstrip().startswith('//')]


def first_hit(results, q, tol_frames):
    """(rank khung dung dau tien, rank video dung dau tien), 1-based; None neu khong co."""
    frame_rank = video_rank = None
    for rank, r in enumerate(results, 1):
        if r['video'] != q['video']:
            continue
        video_rank = video_rank or rank
        if q['start'] - tol_frames <= r['frame_idx'] <= q['end'] + tol_frames:
            frame_rank = rank
            break
    return frame_rank, video_rank


class Translator:
    """Viet -> Anh qua endpoint cong khai cua Google (giong UI), co cache file. Cau khong dau giu nguyen."""

    def __init__(self, cache_path='queries/translate_cache.json'):
        self.path = cache_path
        self.cache = json.load(open(cache_path, encoding='utf-8')) if os.path.exists(cache_path) else {}
        self.dirty = False

    def __call__(self, text):
        if not text or not VI_CHARS.search(text):
            return text
        if text not in self.cache:
            import requests
            r = requests.get('https://translate.googleapis.com/translate_a/single',
                             params={'client': 'gtx', 'sl': 'vi', 'tl': 'en', 'dt': 't', 'q': text}, timeout=20)
            r.raise_for_status()
            self.cache[text] = ''.join(x[0] for x in r.json()[0] if x and x[0])
            self.dirty = True
        return self.cache[text]

    def save(self):
        if self.dirty:
            os.makedirs(os.path.dirname(self.path) or '.', exist_ok=True)
            with open(self.path, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=1)


def report(df, title):
    rk = df['rank'].astype('float')
    vrk = df['video_rank'].astype('float')
    print(f'\n{title} ({len(df)} cau)')
    print('           ' + ''.join(f'R@{k:<6d}' for k in KS) + 'MRR')
    print('khung dung ' + ''.join(f'{(rk <= k).mean():<8.3f}' for k in KS) + f'{np.nan_to_num(1 / rk).mean():.3f}')
    print('video dung ' + ''.join(f'{(vrk <= k).mean():<8.3f}' for k in KS) + f'{np.nan_to_num(1 / vrk).mean():.3f}')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--index', required=True, nargs='+', help='thu muc index (nhieu cai neu engine ho tro gop)')
    p.add_argument('--weights', type=float, nargs='*', default=None, help='trong so tung --index (khi gop)')
    p.add_argument('--data', default=None)
    p.add_argument('--queries', required=True)
    p.add_argument('--k', type=int, default=100, help='so ket qua lay ra moi cau')
    p.add_argument('--tol', type=float, default=2.0, help='dung sai (giay) quanh khoang dap an')
    p.add_argument('--max-per-video', type=int, default=0)
    p.add_argument('--translate', action='store_true', help='dich cau tieng Viet sang tieng Anh truoc khi tim')
    p.add_argument('--out', default=None, help='CSV ket qua tung cau (de so sanh giua cac lan cai tien)')
    p.add_argument('--caption', default=None, help='thu muc index tit (build_caption_index.py)')
    p.add_argument('--caption-weight', type=float, nargs='+', default=[0.5],
                   help='trong so diem tit; nhieu gia tri -> do lan luot')
    a = p.parse_args()

    # 1 index -> goi nhu cu (chay duoc voi engine.py goc); nhieu index / tit -> can engine.py ban gop
    if len(a.index) == 1 and not a.weights and not a.caption:
        eng = SearchEngine(a.index[0], a.data)
    else:
        eng = SearchEngine(a.index, a.data, weights=a.weights, caption_dir=a.caption)
    tr = Translator() if a.translate else (lambda t: t)
    qs = load_queries(a.queries)
    weights = a.caption_weight if a.caption else [None]
    summary = []
    for w in weights:
        if w is not None:
            eng.caption_weight = w
        df = run_all(eng, qs, a, tr)
        if len(weights) == 1:
            full_report(eng, df, a)
            save(df, a.out)
        else:
            rk, vrk = df['rank'].astype('float'), df['video_rank'].astype('float')
            summary.append({'caption_weight': w, **{f'R@{k}': (rk <= k).mean() for k in (1, 5, 10, 20)},
                            'MRR': np.nan_to_num(1 / rk).mean(), 'video_R@1': (vrk <= 1).mean(),
                            'video_MRR': np.nan_to_num(1 / vrk).mean()})
            save(df, a.out and f'{os.path.splitext(a.out)[0]}_w{w:g}.csv')
    if a.translate:
        tr.save()
    if summary:
        print(f'\n{len(df)} cau | model {eng.info["model"]} | tit {eng.info.get("caption_model")} | khung dung:')
        print(pd.DataFrame(summary).to_string(index=False, float_format=lambda x: f'{x:.3f}'))


def run_all(eng, qs, a, tr):
    fps_of = eng.meta.groupby('video')['fps'].first().to_dict()
    rows = []
    for q in qs:
        q.setdefault('query', '')
        if q['video'] not in fps_of:
            print(f'  !! bo qua {q.get("id")}: video {q["video"]} khong co trong index')
            continue
        texts = [tr(q.get(f)) for f in ('query', 'query_b', 'query_c')]
        if q.get('clip'):          # vkis: truy van bang clip de bai, khong co chu
            if not os.path.exists(q['clip']):
                print(f'  !! bo qua {q.get("id")}: khong thay clip {q["clip"]}')
                continue
            res = eng.search_frames(sample_frames(open(q['clip'], 'rb').read()), a.k,
                                    a.max_per_video or None)
        elif q.get('query_b'):
            res = eng.temporal(texts, q.get('window', 30.0), a.k, max_per_video=a.max_per_video or None)
        else:
            res = eng.search(texts[0], a.k, a.max_per_video or None)
        fr, vr = first_hit(res, q, a.tol * fps_of[q['video']])
        used = ' | '.join(t for t in texts if t) or os.path.basename(q.get('clip', ''))
        rows.append({'id': q.get('id'), 'kind': q.get('kind') or ('vkis' if q.get('clip') else 'text'),
                     'query': used, 'video': q['video'], 'rank': fr, 'video_rank': vr,
                     'top1': f'{res[0]["video"]}@{res[0]["frame_idx"]}' if res else ''})
    df = pd.DataFrame(rows)
    if not len(df):
        raise SystemExit('Khong co cau hoi hop le')
    return df


def save(df, path):
    if path:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        df.to_csv(path, index=False)
        print('->', path)


def full_report(eng, df, a):
    cap = f' | tit {eng.info.get("caption_model")} x{eng.caption_weight:g}' if getattr(eng, 'cap', None) else ''
    print(f'\n{len(df)} cau hoi | model {eng.info["model"]} ({eng.info["pretrained"]}) | '
          f'{eng.info["n_keyframes"]:,} keyframe | tol {a.tol}s | dich {"co" if a.translate else "KHONG"}{cap}')
    report(df, 'TAT CA')
    if df['kind'].nunique() > 1:
        for kind, g in df.groupby('kind'):
            report(g, kind)
    miss = df[df['rank'].isna()]
    if len(miss):
        print(f'\n{len(miss)} cau khong tim thay trong top {a.k}:')
        for _, r in miss.head(15).iterrows():
            print(f'  {r["id"]} [{r["kind"]}]: {r["query"][:80]}  (dap an {r["video"]}, '
                  f'video dung o rank {r["video_rank"]})')


if __name__ == '__main__':
    main()
