"""OCR chu tren man hinh cho keyframe DA TRICH (doc thang tar), EasyOCR tieng Viet tren GPU.

Dau ra: <out>/<vid>.jsonl, moi dong 1 keyframe CO CHU (keyframe khong co chu thi khong ghi):
    {"frame_idx": 12345, "text": "GIAI PHONG KHI HIDRO ...", "lines": [{"text": "...", "box": [x1, y1, x2, y2], "conf": 0.93}]}
Video khong co chu nao van co file rong = da xong (de chay tiep khong lam lai).

Dung trong engine: chep thu muc <out> vao <data>/ocr/, hoac chay app.py --ocr <out>.
Engine so khop KHONG DAU (norm_vi) nen OCR sai dau it anh huong; quan trong la bat duoc chu.

    pip install easyocr
    python ocr_keyframes.py --data .../Batch1 .../Batch2 --out ocr --limit 2      # do toc do
    python ocr_keyframes.py --data .../Batch1 .../Batch2 --out ocr
Kaggle: --out /kaggle/working/ocr, --resume <output lan truoc>, --max-hours 11. Xem kaggle_embed_ocr.ipynb.
Chu nho (ticker, bien hieu) bi sot: thu --mag 1.5 (detector phong to anh, cham hon ~2 lan).

Tu kiem tra (khong can GPU/model):  python ocr_keyframes.py --selftest
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from embed_keyframes import check_writable, list_videos, read_video, seed_from


def jpeg_size(b):
    """(w, h) cua jpg tu header, khong giai ma ca anh."""
    import struct
    i = 2
    while i < len(b):
        if b[i] != 0xFF:
            i += 1
            continue
        marker = b[i + 1]
        if marker in (0xC0, 0xC1, 0xC2):
            h, w = struct.unpack('>HH', b[i + 5:i + 9])
            return w, h
        i += 2 + struct.unpack('>H', b[i + 2:i + 4])[0]
    raise ValueError('khong doc duoc kich thuoc jpg')


def to_lines(res, min_conf):
    """Ket qua EasyOCR [(4 diem, text, conf)] -> dong da loc, sap tren xuong, trai qua phai."""
    out = []
    for pts, text, conf in res:
        text = ' '.join(str(text).split())
        if not text or conf < min_conf or (len(text) == 1 and not text.isalnum()):
            continue
        xs, ys = [float(p[0]) for p in pts], [float(p[1]) for p in pts]
        out.append({'text': text, 'box': [round(min(xs)), round(min(ys)), round(max(xs)), round(max(ys))],
                    'conf': round(float(conf), 3)})
    # cung hang neu tam y lech < nua chieu cao dong -> gom hang theo bac 12px roi xet x
    out.sort(key=lambda l: (round((l['box'][1] + l['box'][3]) / 2 / 12), l['box'][0]))
    return out


def ocr_video(raw, frame_idx, reader_fn, batch):
    """-> list dong jsonl (chi keyframe co chu). reader_fn(list byte jpg, (w, h)) -> list ket qua tung anh."""
    rows = []
    for s in range(0, len(raw), batch):
        chunk, fis = raw[s:s + batch], frame_idx[s:s + batch]
        size = jpeg_size(chunk[0])
        for fi, lines in zip(fis, reader_fn(chunk, size)):
            if lines:
                rows.append({'frame_idx': fi, 'text': ' '.join(l['text'] for l in lines), 'lines': lines})
    return rows


def save_jsonl(path, rows):
    tmp = path + '.part'
    with open(tmp, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    os.replace(tmp, path)


def make_reader(langs, mag, min_conf, rec_batch, text_threshold, gpu=True):
    import easyocr
    reader = easyocr.Reader(langs, gpu=gpu, verbose=False)

    def fn(chunk, size):
        w, h = size
        # cung kich thuoc trong 1 lo (1 video) -> detector CRAFT chay theo lo tren GPU
        res = reader.readtext_batched(list(chunk), n_width=w, n_height=h, batch_size=rec_batch,
                                      mag_ratio=mag, text_threshold=text_threshold, paragraph=False)
        return [to_lines(r, min_conf) for r in res]

    print(f'EasyOCR {langs} | gpu={gpu} | mag {mag} | min_conf {min_conf}')
    return fn


def run(a, reader_fn):
    vids = list_videos(a.data, a.only, a.shard)
    check_writable(a.out)
    seed_from(a.resume, a.out, '.jsonl')
    todo = [(v, d) for v, d in vids if not os.path.exists(os.path.join(a.out, f'{v}.jsonl'))]
    print(f'{len(vids)} video, con {len(todo)} video chua OCR')
    if a.limit:
        todo = todo[:a.limit]

    errors, done_kf, t_all = [], 0, time.time()
    with ThreadPoolExecutor(1) as io_pool:
        nxt = io_pool.submit(read_video, *todo[0]) if todo else None
        for k, (vid, d) in enumerate(todo):
            if a.max_hours and time.time() - t_all > a.max_hours * 3600:
                print(f'>>> Da chay {a.max_hours}h -> dung som, con {len(todo) - k} video. Chay lai voi --resume.')
                break
            cur = nxt
            nxt = io_pool.submit(read_video, *todo[k + 1]) if k + 1 < len(todo) else None
            t0 = time.time()
            try:
                frame_idx, raw = cur.result()
                rows = ocr_video(raw, frame_idx, reader_fn, a.batch)
                save_jsonl(os.path.join(a.out, f'{vid}.jsonl'), rows)
            except Exception as e:
                errors.append((vid, f'{type(e).__name__}: {e}'))
                print(f'!! {vid}: {errors[-1][1][:300]}')
                continue
            dt = time.time() - t0
            done_kf += len(raw)
            speed = done_kf / (time.time() - t_all)
            left_v = len(todo) - k - 1
            avg_kf = done_kf / (k + 1 - len(errors)) if k + 1 > len(errors) else 0
            print(f'[{k + 1}/{len(todo)}] {vid}: {len(raw):5d} kf, {len(rows):5d} co chu, '
                  f'{len(raw) / max(dt, 1e-6):5.1f} anh/s | TB {speed:5.1f} anh/s | '
                  f'ETA ~{left_v * avg_kf / max(speed, 1e-6) / 3600:.2f}h')
    print(f'Xong {len(todo) - len(errors)}/{len(todo)} video, {done_kf:,} keyframe trong '
          f'{(time.time() - t_all) / 60:.1f} phut | loi {len(errors)}')
    for vid, msg in errors:
        print('  ', vid, msg[:200])
    return errors


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', nargs='+', help='thu muc co keyframes/ + map-keyframes/')
    p.add_argument('--out', default='ocr', help='thu muc ghi <vid>.jsonl')
    p.add_argument('--langs', nargs='+', default=['vi'], help='ngon ngu EasyOCR (vi da gom chu Latin/Anh)')
    p.add_argument('--batch', type=int, default=16, help='so anh / lo detector')
    p.add_argument('--rec-batch', type=int, default=64, help='so dong chu / lo recognizer')
    p.add_argument('--mag', type=float, default=1.0, help='phong to anh cho detector (1.5: bat chu nho hon)')
    p.add_argument('--text-threshold', type=float, default=0.7)
    p.add_argument('--min-conf', type=float, default=0.2, help='bo dong chu co do tin cay thap hon')
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--only', nargs='*', default=None, help='chi lay video co tien to nay, vd L01 L02')
    p.add_argument('--shard', default=None, help='i/n: may thu i (tu 0) trong n may chia nhau')
    p.add_argument('--limit', type=int, default=0, help='chi chay N video dau (do toc do)')
    p.add_argument('--resume', nargs='*', default=None, help='thu muc .jsonl cua lan chay truoc')
    p.add_argument('--max-hours', type=float, default=0, help='dung som sau N gio (Kaggle gioi han 12h)')
    p.add_argument('--selftest', action='store_true')
    a = p.parse_args(argv)
    if a.shard:
        i, n = map(int, a.shard.split('/'))
        assert 0 <= i < n
        a.shard = (i, n)
    if not a.selftest and not a.data:
        p.error('can --data')
    a.out = os.path.abspath(a.out)
    return a


def main():
    a = parse_args()
    if a.selftest:
        return selftest()
    run(a, make_reader(a.langs, a.mag, a.min_conf, a.rec_batch, a.text_threshold, gpu=not a.cpu))


def selftest():
    """Reader gia -> kiem tra dinh dang dau ra, loc/sap dong, resume, va engine doc duoc file OCR."""
    import csv
    import io
    import shutil
    import tarfile
    import tempfile

    import numpy as np
    from PIL import Image

    # 1. to_lines: loc conf thap / ky tu rac, sap tren xuong trai qua phai
    res = [([[300, 400], [500, 400], [500, 430], [300, 430]], 'Hà Nội', 0.9),
           ([[10, 402], [200, 402], [200, 428], [10, 428]], 'THỜI SỰ', 0.95),
           ([[10, 20], [90, 20], [90, 40], [10, 40]], 'VTV1', 0.8),
           ([[0, 0], [5, 0], [5, 5], [0, 5]], '|', 0.9),
           ([[0, 0], [50, 0], [50, 9], [0, 9]], 'mo', 0.05)]
    assert [l['text'] for l in to_lines(res, 0.2)] == ['VTV1', 'THỜI SỰ', 'Hà Nội']
    assert to_lines(res, 0.2)[0]['box'] == [10, 20, 90, 40]

    buf = io.BytesIO()
    Image.new('RGB', (854, 480)).save(buf, 'JPEG')
    assert jpeg_size(buf.getvalue()) == (854, 480)

    with tempfile.TemporaryDirectory() as tmp:
        b1 = os.path.join(tmp, 'Batch1')
        for sub in ('keyframes', 'map-keyframes'):
            os.makedirs(os.path.join(b1, sub))
        vids = {'L01_V001': [0, 50, 100], 'L01_V002': [10, 20]}
        for vid, fis in vids.items():
            kd = os.path.join(tmp, '_kf', vid)
            os.makedirs(kd)
            for fi in fis:
                Image.new('RGB', (64, 36), (fi % 256, 0, 0)).save(os.path.join(kd, f'{fi:06d}.jpg'))
            with tarfile.open(os.path.join(b1, 'keyframes', f'{vid}.tar'), 'w') as tf:
                tf.add(kd, arcname=vid)
            with open(os.path.join(b1, 'map-keyframes', f'{vid}.csv'), 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['n', 'pts_time', 'fps', 'frame_idx'])
                w.writerows([[n + 1, fi / 25, 25.0, fi] for n, fi in enumerate(fis)])

        calls = []

        def fake(chunk, size):
            calls.append((len(chunk), size))
            out = []
            for b in chunk:
                red = int(np.asarray(Image.open(io.BytesIO(b)))[0, 0, 0])
                # anh frame 50 co chu, cac anh khac khong
                out.append(to_lines([([[0, 0], [9, 0], [9, 9], [0, 9]], f'GIẢI PHÓNG {red}', 0.9)], 0.2)
                           if 45 <= red <= 55 else [])
            return out

        out = os.path.join(tmp, 'ocr')
        a = parse_args(['--data', b1, '--out', out, '--batch', '2'])
        assert run(a, fake) == []
        assert calls[0] == (2, (64, 36)) and sum(c[0] for c in calls) == 5
        rows = [json.loads(l) for l in open(os.path.join(out, 'L01_V001.jsonl'), encoding='utf-8')]
        assert len(rows) == 1 and rows[0]['frame_idx'] == 50 and rows[0]['text'].startswith('GIẢI PHÓNG')
        assert open(os.path.join(out, 'L01_V002.jsonl')).read() == ''          # khong chu van danh dau xong

        calls.clear()
        run(a, fake)
        assert calls == [], 'lan 2 phai bo qua video da xong'
        out2 = os.path.join(tmp, 'ocr2')
        run(parse_args(['--data', b1, '--out', out2, '--resume', out]), fake)
        assert calls == [] and sorted(os.listdir(out2)) == ['L01_V001.jsonl', 'L01_V002.jsonl']

        # engine doc duoc: dat vao <data>/ocr roi loc khong dau (Kaggle khong co engine/build_index -> bo qua)
        here = os.path.dirname(os.path.abspath(__file__))
        if not all(os.path.exists(os.path.join(here, f)) for f in ('build_index.py', 'engine.py')):
            print('ocr_keyframes.py selftest OK (bo qua kiem tra engine)')
            return
        shutil.copytree(out, os.path.join(b1, 'ocr'))
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import subprocess
        idx = os.path.join(tmp, 'index')
        os.makedirs(os.path.join(b1, 'features-vit-b-32'))
        for vid, fis in vids.items():
            np.save(os.path.join(b1, 'features-vit-b-32', f'{vid}.npy'), np.eye(len(fis), 4, dtype=np.float16))
        r = subprocess.run([sys.executable, os.path.join(here, 'build_index.py'), '--data', b1, '--out', idx],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        try:
            from engine import SearchEngine
        except ImportError:          # may khong co torch: bo qua buoc nay
            print('  (bo qua kiem tra engine: thieu torch)')
        else:
            eng = SearchEngine(idx, load_model=False)
            res = eng.search('', ocr='giai phong')
            assert [(x['video'], x['frame_idx']) for x in res] == [('L01_V001', 50)], res
    print('ocr_keyframes.py selftest OK')


if __name__ == '__main__':
    main()
