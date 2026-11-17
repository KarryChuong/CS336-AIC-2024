"""Chuyen bo GT chuan (gt_std/) sang .jsonl cho evaluate.py.

Cau truc dau vao:
    gt_std/index.csv            folder,kind,id,video,start_frame,end_frame,start_s,end_s,...
    gt_std/tkis/tNN/query.txt      3 cau tieng Viet (A / B / C)
    gt_std/tkis/tNN/query_en.txt   3 cau tieng Anh tuong ung
    gt_std/vkis/vNN/query.mp4      clip de bai

Dau ra (mac dinh ngay trong --gt-dir):
    gt_tkis.jsonl     (--lang vi: gt_tkis_vi.jsonl -> chay evaluate them --translate)
    gt_vkis.jsonl

    python gt_std_to_jsonl.py --gt-dir D:/.../gt_std
    python evaluate.py --index <index> --queries D:/.../gt_std/gt_tkis.jsonl --out runs/std_b32_tkis.csv
    python evaluate.py --index <index> --queries D:/.../gt_std/gt_vkis.jsonl --out runs/std_b32_vkis.csv

Tu kiem tra: python gt_std_to_jsonl.py --selftest
"""
import argparse
import csv
import json
import os


def read_lines(path):
    with open(path, encoding='utf-8-sig') as f:
        return [l.strip() for l in f if l.strip()]


def convert(gt_dir, lang='en', min_window=30.0):
    """-> (tkis, vkis, canh_bao). Lay dap an tu index.csv, cau hoi/clip tu thu muc tung cau."""
    tkis, vkis, warn = [], [], []
    with open(os.path.join(gt_dir, 'index.csv'), encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        folder = os.path.join(gt_dir, *r['folder'].replace('\\', '/').split('/'))
        name = r['folder'].replace('\\', '/')
        base = {'id': f"{name.split('/')[-1]}_{r['id']}", 'video': r['video'],
                'start': int(float(r['start_frame'])), 'end': int(float(r['end_frame'])),
                'source': r.get('source', '')}
        if r['kind'] == 'vkis':
            clip = os.path.join(folder, 'query.mp4')
            if not os.path.exists(clip):
                warn.append(f'{name}: thieu query.mp4')
                continue
            vkis.append({**base, 'kind': 'vkis', 'clip': os.path.abspath(clip).replace(os.sep, '/')})
            continue
        qf = os.path.join(folder, 'query_en.txt' if lang == 'en' else 'query.txt')
        if lang == 'en' and not os.path.exists(qf) and os.path.exists(os.path.join(folder, 'query.txt')):
            warn.append(f'{name}: thieu query_en.txt -> dung query.txt (nho --translate)')
            qf = os.path.join(folder, 'query.txt')
        if not os.path.exists(qf):
            warn.append(f'{name}: thieu file cau hoi')
            continue
        lines = read_lines(qf)
        if not lines:
            warn.append(f'{name}: file cau hoi rong')
            continue
        span = float(r['end_s']) - float(r['start_s'])
        item = {**base, 'kind': 'tkis', 'query': lines[0]}
        if len(lines) > 1:
            # A -> B -> C deu nam trong doan dap an: cua so it nhat bang do dai doan (toi thieu 30s)
            item['query_b'] = lines[1]
            item['window'] = round(max(min_window, span), 1)
        if len(lines) > 2:
            item['query_c'] = lines[2]
        tkis.append(item)
    return tkis, vkis, warn


def write_jsonl(path, items, note):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f'// {note}\n')
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--gt-dir', help='thu muc gt_std (co index.csv)')
    p.add_argument('--lang', choices=('en', 'vi'), default='en', help='en: query_en.txt; vi: query.txt')
    p.add_argument('--out', default=None, help='thu muc ghi .jsonl (mac dinh = --gt-dir)')
    p.add_argument('--selftest', action='store_true')
    a = p.parse_args()
    if a.selftest:
        return selftest()
    if not a.gt_dir:
        p.error('can --gt-dir')
    tkis, vkis, warn = convert(a.gt_dir, a.lang)
    out = a.out or a.gt_dir
    os.makedirs(out, exist_ok=True)
    t_path = os.path.join(out, 'gt_tkis.jsonl' if a.lang == 'en' else 'gt_tkis_vi.jsonl')
    v_path = os.path.join(out, 'gt_vkis.jsonl')
    write_jsonl(t_path, tkis, f'tkis tu {a.gt_dir} ({a.lang})')
    write_jsonl(v_path, vkis, f'vkis tu {a.gt_dir}: truy van bang clip query.mp4')
    print(f'tkis: {len(tkis)} cau -> {t_path}')
    print(f'vkis: {len(vkis)} cau -> {v_path}')
    for w in warn:
        print('  !!', w)


def selftest():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        for sub in ('tkis/t01', 'tkis/t02', 'tkis/t03', 'vkis/v01', 'vkis/v02'):
            os.makedirs(os.path.join(tmp, *sub.split('/')))
        with open(os.path.join(tmp, 'index.csv'), 'w', encoding='utf-8-sig', newline='') as f:
            f.write('folder,kind,id,video,start_frame,end_frame,start_s,end_s,match_score,source,n_hint,has_en,task_id,task_dir\n'
                    'vkis/v01,vkis,f2cbb5cf,L24_V021,8730,9375,291,312.5,0.9609,practice,,,f2cbb5cf,x\n'
                    'vkis/v02,vkis,aaaa,L01_V001,0,10,0,1,,practice,,,aaaa,y\n'
                    'tkis/t01,tkis,0befa6e4,L22_V009,11424,14460,456.96,578.4,0.96,practice,3,TRUE,0befa6e4,\n'
                    'tkis/t02,tkis,clip10,L03_V015,22860,23544,914.4,941.76,,notebooklm,3,TRUE,clip10,\n'
                    'tkis/t03,tkis,clip11,L03_V016,100,200,4,8,,notebooklm,3,TRUE,clip11,\n')
        open(os.path.join(tmp, 'vkis', 'v01', 'query.mp4'), 'wb').write(b'x')     # v02 thieu clip
        for t, vi, en in (('t01', 'Mở đầu A\nSau đó B\nCuối cùng C\n', 'A\nB\nC\n'),
                          ('t02', '\ufeffMở đầu là người đàn ông\n\nSau đó khách\nCuối cùng cận cảnh\n', None)):
            open(os.path.join(tmp, 'tkis', t, 'query.txt'), 'w', encoding='utf-8').write(vi)
            if en:
                open(os.path.join(tmp, 'tkis', t, 'query_en.txt'), 'w', encoding='utf-8').write(en)
        tkis, vkis, warn = convert(tmp)
        assert [v['id'] for v in vkis] == ['v01_f2cbb5cf'] and vkis[0]['clip'].endswith('vkis/v01/query.mp4')
        assert (vkis[0]['video'], vkis[0]['start'], vkis[0]['end']) == ('L24_V021', 8730, 9375)
        t1 = tkis[0]
        assert (t1['query'], t1['query_b'], t1['query_c']) == ('A', 'B', 'C') and t1['window'] == 121.4, t1
        assert (t1['start'], t1['end']) == (11424, 14460)
        t2 = tkis[1]                                     # thieu query_en -> tieng Viet, bo BOM/dong trong
        assert t2['query'] == 'Mở đầu là người đàn ông' and t2['query_c'] == 'Cuối cùng cận cảnh'
        assert t2['window'] == 30.0                      # doan 27s < 30 -> giu toi thieu 30s
        assert len(tkis) == 2 and len(warn) == 3, warn   # v02 thieu clip, t02 thieu en, t03 thieu cau hoi
        tv, _, _ = convert(tmp, 'vi')
        assert tv[0]['query'] == 'Mở đầu A'
        out = os.path.join(tmp, 'out')
        os.makedirs(out)
        write_jsonl(os.path.join(out, 'gt_tkis.jsonl'), tkis, 'test')
        lines = open(os.path.join(out, 'gt_tkis.jsonl'), encoding='utf-8').read().splitlines()
        assert lines[0].startswith('//') and json.loads(lines[1])['id'] == 't01_0befa6e4'
    print('gt_std_to_jsonl.py selftest OK')


if __name__ == '__main__':
    main()
