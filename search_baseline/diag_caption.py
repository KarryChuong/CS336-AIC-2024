"""Chan doan nhanh tit: voi moi cau tkis, dong tit nao khop nhat (cosine THO, khong z-score),
va dong tit thuoc doan dap an dung o dau.

Muc dich: biet nhanh tit co tin hieu that khong, va nguong cosine nao tach duoc tit lien quan / khong
lien quan -> chon cach gop an toan. Khong can GPU, khong nap model hinh anh (chi nap BGE-M3).

    python diag_caption.py --index D:/.../index --caption D:/.../Data/index_caption --queries D:/.../gt_std/gt_tkis.jsonl
"""
import argparse
import json

import numpy as np

from engine import SearchEngine


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--index', required=True, help='1 index bat ky (chi lay meta)')
    p.add_argument('--caption', required=True)
    p.add_argument('--queries', required=True)
    p.add_argument('--pad', type=float, default=15.0, help='giay noi rong quanh doan dap an khi tim tit cua no')
    a = p.parse_args()

    eng = SearchEngine(a.index, load_model=False, caption_dir=a.caption)
    c = eng.cap
    emb = c['emb'].float()
    cap_video = {}
    for i in np.where(c['cap_of'] >= 0)[0]:
        cap_video.setdefault(int(c['cap_of'][i]), set()).add(str(eng.video[i]))
    qs = [json.loads(l) for l in open(a.queries, encoding='utf-8') if l.strip() and not l.lstrip().startswith('//')]

    rows = []
    for q in qs:
        fps = float(eng.meta.loc[eng.video == q['video'], 'fps'].iloc[0])
        t0, t1 = q['start'] / fps - a.pad, q['end'] / fps + a.pad
        in_seg = (eng.video == q['video']) & (eng.pts >= t0) & (eng.pts <= t1) & (c['cap_of'] >= 0)
        gt_caps = sorted({int(x) for x in c['cap_of'][in_seg]})
        texts = {'A': q['query'], 'ABC': ' '.join(q.get(k, '') for k in ('query', 'query_b', 'query_c')).strip()}
        out = {'id': q['id'], 'gt_caps': len(gt_caps)}
        for name, t in texts.items():
            sims = (emb @ eng._caption_query_vec(t).float()).cpu().numpy()    # GPU -> phai ve CPU truoc
            order = np.argsort(-sims)
            best = int(order[0])
            out[f'{name}_top'] = float(sims[best])
            out[f'{name}_top_gt'] = q['video'] in cap_video.get(best, set())
            if gt_caps:
                g = max(gt_caps, key=lambda j: sims[j])
                out[f'{name}_gt'] = float(sims[g])
                out[f'{name}_gt_rank'] = int((sims > sims[g]).sum()) + 1
                if name == 'ABC':
                    out['gt_text'] = c['texts'][g][:70]
            if name == 'ABC':
                out['top_text'] = c['texts'][best][:70]
        rows.append(out)

    print(f'\n{len(rows)} cau | {len(c["texts"]):,} dong tit | cosine THO (BGE-M3), dap an +-{a.pad:g}s\n')
    print(f'{"cau":16s} {"tit_GT":>6s} | {"A:GT":>6s} {"hang":>6s} {"A:top":>6s} | {"ABC:GT":>6s} {"hang":>6s} {"ABC:top":>7s} top_dung_video')
    for r in rows:
        f = lambda k, fmt: (fmt % r[k]) if k in r else '-'
        print(f'{r["id"][:16]:16s} {r["gt_caps"]:6d} | {f("A_gt", "%6.3f")} {f("A_gt_rank", "%6d")} {r["A_top"]:6.3f} | '
              f'{f("ABC_gt", "%6.3f")} {f("ABC_gt_rank", "%6d")} {r["ABC_top"]:7.3f} {"CO" if r["ABC_top_gt"] else ""}')
    print('\nTit trong doan dap an khop nhat (ABC) / tit khop nhat toan bo:')
    for r in rows:
        print(f'  {r["id"][:16]:16s} GT : {r.get("gt_text", "(khong co tit)")}')
        print(f'  {"":16s} TOP: {r["top_text"]}')
    has = [r for r in rows if 'ABC_gt' in r]
    if has:
        g = np.array([r['ABC_gt'] for r in has])
        t = np.array([r['ABC_top'] for r in rows])
        print(f'\n{len(has)}/{len(rows)} cau co tit trong doan dap an | cosine tit dap an: TB {g.mean():.3f}, '
              f'min {g.min():.3f}, max {g.max():.3f} | cosine tit top-1 moi cau: TB {t.mean():.3f}')
        print(f'hang tit dap an (ABC) <= 10: {sum(r["ABC_gt_rank"] <= 10 for r in has)} cau, '
              f'<= 100: {sum(r["ABC_gt_rank"] <= 100 for r in has)} cau')


if __name__ == '__main__':
    main()
