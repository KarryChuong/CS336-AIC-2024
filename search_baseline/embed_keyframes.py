"""Encode lai keyframe DA TRICH bang model manh hon (SigLIP2, DFN ViT-H...), doc thang tu file tar.

Khong trich keyframe lai: dung nguyen keyframes/<vid>.tar + map-keyframes/<vid>.csv cua notebook.

Dau ra: <data>/features-<model>/<vid>.npy  float16, L2-norm, CUNG THU TU voi map-keyframes/<vid>.csv
-> build_index.py dung duoc ngay, khong phai sua:
    python build_index.py --data .../Batch1 .../Batch2 --out .../index_siglip2 \
        --feat-dir features-vit-so400m-16-siglip2-384 --model ViT-SO400M-16-SigLIP2-384 --pretrained webli

Checkpoint: video da co .npy dung so dong voi csv thi bo qua -> bi ngat thi chay lai dung lenh cu.
Chia nhieu tai khoan Colab/Kaggle: --shard 0/2 o may 1, --shard 1/2 o may 2 (ghi chung 1 thu muc Drive).

    pip install -U open_clip_torch timm
    python embed_keyframes.py --data /content/drive/MyDrive/Batch1 /content/drive/MyDrive/Batch2 --limit 2
    python embed_keyframes.py --data /content/drive/MyDrive/Batch1 /content/drive/MyDrive/Batch2

Kaggle (dataset chi doc): bat buoc --out /kaggle/working/...; --resume <output lan truoc> de chay tiep,
--max-hours 11 de tu dung truoc gioi han 12h ma van giu ket qua. Xem kaggle_embed_ocr.ipynb.

Tu kiem tra (khong can GPU, khong can tai model):  python embed_keyframes.py --selftest
"""
import argparse
import csv
import io
import json
import os
import re
import shutil
import sys
import tarfile
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

DEFAULT_MODEL = 'ViT-SO400M-16-SigLIP2-384'
DEFAULT_PRETRAINED = 'webli'


def feat_dir_name(model):
    return 'features-' + re.sub(r'[^a-z0-9]+', '-', model.lower()).strip('-')


def list_videos(data_dirs, only=None, shard=None):
    """[(vid, data_dir)] sap theo ten. Trung ten giua cac thu muc thi lay thu muc dau, giong build_index."""
    seen, out = set(), []
    for d in data_dirs:
        mdir = os.path.join(d, 'map-keyframes')
        if not os.path.isdir(mdir):
            raise SystemExit(f'Khong thay map-keyframes trong {d}')
        for f in sorted(os.listdir(mdir)):
            v = f[:-4]
            if not f.endswith('.csv') or v in seen or (only and not any(v.startswith(p) for p in only)):
                continue
            seen.add(v)
            out.append((v, d))
    out.sort()
    if shard:
        i, n = shard
        out = out[i::n]
    return out


def _jpg_key(name):
    stem = os.path.splitext(os.path.basename(name))[0]
    return int(stem) if stem.isdigit() else None


def read_video(vid, d):
    """(frame_idx theo thu tu csv, byte jpg cung thu tu). Thu tu csv = thu tu build_index ghep feature.

    Anh lay tu keyframes/<vid>.tar (output notebook), hoac thu muc da giai nen
    (keyframes/<vid>.tar/ hay keyframes/<vid>/ -- Kaggle co the tu giai nen tar khi upload dataset).
    """
    with open(os.path.join(d, 'map-keyframes', f'{vid}.csv'), newline='') as f:
        frame_idx = [int(r['frame_idx']) for r in csv.DictReader(f)]
    tar = os.path.join(d, 'keyframes', f'{vid}.tar')
    if os.path.isfile(tar):
        with tarfile.open(tar) as tf:
            members = {_jpg_key(m.name): m for m in tf.getmembers() if m.isfile() and m.name.endswith('.jpg')}
            missing = [fi for fi in frame_idx if fi not in members]
            if missing:
                raise RuntimeError(f'tar thieu {len(missing)}/{len(frame_idx)} anh, vd frame {missing[:3]}')
            return frame_idx, [tf.extractfile(members[fi]).read() for fi in frame_idx]
    folder = next((p for p in (tar, os.path.join(d, 'keyframes', vid)) if os.path.isdir(p)), None)
    if folder is None:
        raise FileNotFoundError(f'khong thay {tar} (file hoac thu muc) hay keyframes/{vid}/')
    paths = {}
    for root, _, files in os.walk(folder):
        for fn in files:
            if fn.endswith('.jpg') and _jpg_key(fn) is not None:
                paths[_jpg_key(fn)] = os.path.join(root, fn)
    missing = [fi for fi in frame_idx if fi not in paths]
    if missing:
        raise RuntimeError(f'thu muc thieu {len(missing)}/{len(frame_idx)} anh, vd frame {missing[:3]}')
    raw = []
    for fi in frame_idx:
        with open(paths[fi], 'rb') as f:
            raw.append(f.read())
    return frame_idx, raw


def seed_from(resume_dirs, out_dir, ext):
    """Chep ket qua lan chay truoc (Kaggle: output version cu, them vao lam Input) sang out_dir de bo qua."""
    n = 0
    for rd in resume_dirs or []:
        for fn in os.listdir(rd):
            dst = os.path.join(out_dir, fn)
            if fn.endswith(ext) and not os.path.exists(dst):
                shutil.copyfile(os.path.join(rd, fn), dst + '.part')
                os.replace(dst + '.part', dst)
                n += 1
    if resume_dirs:
        print(f'resume: chep {n} file {ext} tu {resume_dirs}')


def check_writable(path):
    try:
        os.makedirs(path, exist_ok=True)
        probe = os.path.join(path, '.write_test')
        open(probe, 'w').close()
        os.remove(probe)
    except OSError as e:
        raise SystemExit(f'Khong ghi duoc vao {path} ({e}). Kaggle: dataset chi doc -> dung --out /kaggle/working/...')


def is_done(path, n):
    if not os.path.exists(path):
        return False
    try:
        return np.load(path, mmap_mode='r').shape[0] == n
    except Exception:          # file hong do bi ngat giua chung -> lam lai
        return False


def save_npy(path, arr):
    """Ghi .part roi doi ten: file o dich luon hoan chinh."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.part'
    with open(tmp, 'wb') as f:
        np.save(f, arr)
    os.replace(tmp, path)


def encode_video(raw, load, encode, batch, pool, dim, ahead=3):
    """byte jpg -> [N, D] float16. Giai ma/preprocess song song tren CPU, giu toi da ahead lo cho GPU."""
    futs, it, out = deque(), iter(raw), []

    def fill():
        while len(futs) < batch * ahead:
            b = next(it, None)
            if b is None:
                return
            futs.append(pool.submit(load, b))

    fill()
    while futs:
        x = [futs.popleft().result() for _ in range(min(batch, len(futs)))]
        fill()
        out.append(encode(x))
    return np.concatenate(out).astype(np.float16) if out else np.zeros((0, dim), np.float16)


def load_model(name, pretrained, device=None, fp16=True):
    """(preprocess, encode, dim). Preprocess lay theo pretrained cfg: SigLIP la 'squash' ca khung 16:9,
    khong cat 2 ben nhu center-crop cua ViT-B-32 hien tai."""
    import torch
    import torch.nn.functional as F
    import open_clip

    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    fp16 = fp16 and device == 'cuda'
    model, _, preprocess = open_clip.create_model_and_transforms(name, pretrained=pretrained, device=device)
    model.eval()
    if fp16:
        model = model.half()

    @torch.inference_mode()
    def encode(tensors):
        x = torch.stack(tensors).to(device, non_blocking=True)
        e = model.encode_image(x.half() if fp16 else x)
        return F.normalize(e.float(), dim=-1).cpu().numpy()

    dim = encode([preprocess(Image.new('RGB', (64, 64)))]).shape[1]
    print(f'{name} ({pretrained}) tren {device}{" fp16" if fp16 else ""} | dim {dim}\n  preprocess: '
          + ' -> '.join(type(t).__name__ for t in getattr(preprocess, 'transforms', [])))
    return preprocess, encode, dim


def run(a, preprocess, encode, dim):
    vids = list_videos(a.data, a.only, a.shard)
    out_of = (lambda d: a.out) if a.out else (lambda d: os.path.join(d, feat_dir_name(a.model)))

    for d in sorted({d for _, d in vids}):
        check_writable(out_of(d))
        info = os.path.join(out_of(d), 'info.json')
        if not os.path.exists(info):
            with open(info, 'w', encoding='utf-8') as f:
                json.dump({'model': a.model, 'pretrained': a.pretrained, 'dim': dim,
                           'created': time.strftime('%Y-%m-%d %H:%M:%S')}, f, indent=2)

    if a.out:
        seed_from(a.resume, a.out, '.npy')
    todo = []
    for vid, d in vids:
        p = os.path.join(out_of(d), f'{vid}.npy')
        with open(os.path.join(d, 'map-keyframes', f'{vid}.csv'), newline='') as f:
            n = sum(1 for _ in csv.DictReader(f))
        if not is_done(p, n):
            todo.append((vid, d, p, n))
    n_all = sum(t[3] for t in todo)
    print(f'{len(vids)} video, con {len(todo)} video / {n_all:,} keyframe chua encode')
    if a.limit:
        todo = todo[:a.limit]

    load = (lambda b: preprocess(Image.open(io.BytesIO(b)).convert('RGB')))
    errors, done_kf, t_all = [], 0, time.time()
    with ThreadPoolExecutor(a.workers) as pool, ThreadPoolExecutor(1) as io_pool:
        # doc tar cua video ke tiep trong luc GPU dang chay video hien tai (Drive doc cham)
        nxt = io_pool.submit(read_video, *todo[0][:2]) if todo else None
        for k, (vid, d, path, n) in enumerate(todo):
            if a.max_hours and time.time() - t_all > a.max_hours * 3600:
                print(f'>>> Da chay {a.max_hours}h -> dung som, con {len(todo) - k} video. Chay lai voi --resume.')
                break
            cur = nxt
            nxt = io_pool.submit(read_video, *todo[k + 1][:2]) if k + 1 < len(todo) else None
            t0 = time.time()
            try:
                _, raw = cur.result()
                emb = encode_video(raw, load, encode, a.batch, pool, dim)
                if len(emb) != n:
                    raise RuntimeError(f'lech so dong: csv={n} emb={len(emb)}')
                save_npy(path, emb)
            except Exception as e:
                errors.append((vid, f'{type(e).__name__}: {e}'))
                print(f'!! {vid}: {errors[-1][1][:300]}')
                continue
            dt = time.time() - t0
            done_kf += n
            speed = done_kf / (time.time() - t_all)
            left = sum(t[3] for t in todo[k + 1:])
            print(f'[{k + 1}/{len(todo)}] {vid}: {n:5d} kf, {n / max(dt, 1e-6):6.1f} anh/s | '
                  f'TB {speed:6.1f} anh/s | ETA {left / max(speed, 1e-6) / 3600:.2f}h')

    print(f'Xong {len(todo) - len(errors)}/{len(todo)} video, {done_kf:,} keyframe trong '
          f'{(time.time() - t_all) / 60:.1f} phut | loi {len(errors)}')
    for vid, msg in errors:
        print('  ', vid, msg[:200])
    feat = a.out or feat_dir_name(a.model)
    print('\nTao index:\n  python build_index.py --data ' + ' '.join(a.data)
          + f' --out <thu muc index moi> --feat-dir {feat} --model {a.model} --pretrained {a.pretrained}')
    return errors


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', nargs='+', help='thu muc co keyframes/*.tar + map-keyframes/*.csv')
    p.add_argument('--model', default=DEFAULT_MODEL)
    p.add_argument('--pretrained', default=DEFAULT_PRETRAINED)
    p.add_argument('--out', default=None,
                   help='ghi moi .npy vao 1 thu muc nay (mac dinh <data>/features-<model>/). '
                        'Dung --out thi build_index can --feat-dir <duong dan tuyet doi nay>')
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--workers', type=int, default=min(8, os.cpu_count() or 2), help='so thread giai ma jpg')
    p.add_argument('--device', default=None)
    p.add_argument('--fp32', action='store_true', help='tat fp16 tren GPU')
    p.add_argument('--only', nargs='*', default=None, help='chi lay video co tien to nay, vd L01 L02')
    p.add_argument('--shard', default=None, help='i/n: may thu i (tu 0) trong n may chia nhau')
    p.add_argument('--limit', type=int, default=0, help='chi chay N video dau (do toc do)')
    p.add_argument('--resume', nargs='*', default=None, help='thu muc .npy cua lan chay truoc (chi dung voi --out)')
    p.add_argument('--max-hours', type=float, default=0, help='dung som sau N gio (Kaggle gioi han 12h)')
    p.add_argument('--selftest', action='store_true')
    a = p.parse_args(argv)
    if a.shard:
        i, n = map(int, a.shard.split('/'))
        assert 0 <= i < n, '--shard phai dang i/n voi 0 <= i < n'
        a.shard = (i, n)
    if not a.selftest and not a.data:
        p.error('can --data')
    return a


def main():
    a = parse_args()
    if a.selftest:
        return selftest()
    if a.out:
        a.out = os.path.abspath(a.out)
        check_writable(a.out)
    pre, enc, dim = load_model(a.model, a.pretrained, a.device, not a.fp32)
    run(a, pre, enc, dim)


def selftest():
    """Du lieu gia giong output notebook + encoder gia (mau trung binh) -> kiem tra thu tu, resume,
    shard, video loi khong lam dung ca lo, va build_index doc duoc ket qua."""
    import shutil
    import subprocess
    import tempfile

    def make_batch(root, vids):
        for sub in ('keyframes', 'map-keyframes'):
            os.makedirs(os.path.join(root, sub), exist_ok=True)
        for vid, fis in vids.items():
            tmp = os.path.join(root, '_kf', vid)
            os.makedirs(tmp)
            for fi in reversed(fis):           # ghi tar nguoc thu tu -> phai doc theo csv, khong theo tar
                Image.new('RGB', (64, 36), color_of(vid, fi)).save(os.path.join(tmp, f'{fi:06d}.jpg'), quality=95)
            with tarfile.open(os.path.join(root, 'keyframes', f'{vid}.tar'), 'w') as tf:
                tf.add(tmp, arcname=vid)
            with open(os.path.join(root, 'map-keyframes', f'{vid}.csv'), 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['n', 'pts_time', 'fps', 'frame_idx'])
                for n, fi in enumerate(fis, 1):
                    w.writerow([n, fi / 25, 25.0, fi])
        shutil.rmtree(os.path.join(root, '_kf'))

    def color_of(vid, fi):
        return (int(vid[-1]) * 40 % 256, fi % 256, 200)

    pre = lambda im: np.asarray(im.resize((8, 8)), np.float32)
    calls = []

    def enc(xs):
        calls.append(len(xs))
        m = np.stack([x.mean(axis=(0, 1)) for x in xs])
        return m / np.linalg.norm(m, axis=1, keepdims=True)

    with tempfile.TemporaryDirectory() as tmp:
        b1, b2 = os.path.join(tmp, 'Batch1'), os.path.join(tmp, 'Batch2')
        make_batch(b1, {'L01_V001': [0, 12, 50, 99, 130], 'L01_V002': [3, 7]})
        make_batch(b2, {'L02_V001': list(range(0, 400, 4)), 'L01_V002': [1]})   # trung ten -> lay Batch1

        a = parse_args(['--data', b1, b2, '--batch', '16', '--workers', '2'])
        assert run(a, pre, enc, 3) == []
        assert max(calls) <= 16 and sum(calls) == 5 + 2 + 100, calls
        fd = feat_dir_name(DEFAULT_MODEL)
        e1 = np.load(os.path.join(b1, fd, 'L01_V001.npy'))
        assert e1.dtype == np.float16 and e1.shape == (5, 3), e1.shape
        exp = enc([pre(Image.new('RGB', (64, 36), color_of('L01_V001', fi))) for fi in [0, 12, 50, 99, 130]])
        assert np.abs(e1.astype(np.float32) - exp).max() < 0.02, 'sai thu tu so voi csv'
        assert not os.path.exists(os.path.join(b2, fd, 'L01_V002.npy')), 'video trung ten phai lay Batch1'
        assert json.load(open(os.path.join(b1, fd, 'info.json')))['dim'] == 3

        # resume: lan 2 khong encode gi; xoa 1 file / lam hong 1 file -> chi lam lai dung file do
        calls.clear()
        run(a, pre, enc, 3)
        assert calls == [], calls
        os.remove(os.path.join(b1, fd, 'L01_V002.npy'))
        open(os.path.join(b2, fd, 'L02_V001.npy'), 'wb').write(b'hong')
        run(a, pre, enc, 3)
        assert sum(calls) == 2 + 100, calls
        assert not any(f.endswith('.part') for d in (b1, b2) for f in os.listdir(os.path.join(d, fd)))

        # shard: 2 may chia nhau, khong trung, du ca bo
        s0 = [v for v, _ in list_videos([b1, b2], shard=(0, 2))]
        s1 = [v for v, _ in list_videos([b1, b2], shard=(1, 2))]
        assert sorted(s0 + s1) == ['L01_V001', 'L01_V002', 'L02_V001'] and not set(s0) & set(s1)

        # video thieu anh trong tar -> bao loi video do, cac video khac van chay
        with open(os.path.join(b1, 'map-keyframes', 'L01_V001.csv'), 'a', newline='') as f:
            csv.writer(f).writerow([6, 9.0, 25.0, 225])
        errs = run(a, pre, enc, 3)
        assert len(errs) == 1 and errs[0][0] == 'L01_V001' and 'thieu' in errs[0][1], errs
        with open(os.path.join(b1, 'map-keyframes', 'L01_V001.csv'), newline='') as f:
            rows = list(csv.reader(f))[:-1]
        with open(os.path.join(b1, 'map-keyframes', 'L01_V001.csv'), 'w', newline='') as f:
            csv.writer(f).writerows(rows)

        # --out: gom 1 thu muc; build_index doc duoc bang --feat-dir tuyet doi
        out = os.path.join(tmp, 'feat_all')
        run(parse_args(['--data', b1, b2, '--out', out, '--batch', '8']), pre, enc, 3)
        assert sorted(os.listdir(out)) == ['L01_V001.npy', 'L01_V002.npy', 'L02_V001.npy', 'info.json']
        bi = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'build_index.py')
        if os.path.exists(bi):         # Kaggle chi co 2 script -> bo qua buoc nay
            idx = os.path.join(tmp, 'index')
            r = subprocess.run([sys.executable, bi, '--data', b1, b2, '--out', idx, '--feat-dir', out,
                                '--model', DEFAULT_MODEL, '--pretrained', DEFAULT_PRETRAINED],
                               capture_output=True, text=True)
            assert r.returncode == 0, r.stderr
            feats = np.load(os.path.join(idx, 'features.npy'))
            assert feats.shape == (107, 3), feats.shape
            assert '!! bo qua' not in r.stdout or 'trung ten' in r.stdout, r.stdout

        # Kaggle tu giai nen tar: keyframes/<vid>.tar/<vid>/*.jpg va keyframes/<vid>/*.jpg -> van doc dung thu tu
        fis, raw = read_video('L01_V001', b1)
        for layout in ('tar_dir', 'plain_dir'):
            kd = os.path.join(tmp, layout, 'keyframes')
            shutil.copytree(os.path.join(b1, 'map-keyframes'), os.path.join(tmp, layout, 'map-keyframes'))
            dst = os.path.join(kd, 'L01_V001.tar') if layout == 'tar_dir' else kd
            with tarfile.open(os.path.join(b1, 'keyframes', 'L01_V001.tar')) as tf:
                tf.extractall(dst)
            fis2, raw2 = read_video('L01_V001', os.path.join(tmp, layout))
            assert fis2 == fis and raw2 == raw, layout

        # resume: output lan truoc -> chep sang --out moi, khong encode lai
        out2 = os.path.join(tmp, 'feat_run2')
        calls.clear()
        run(parse_args(['--data', b1, b2, '--out', out2, '--resume', out]), pre, enc, 3)
        assert calls == [] and sorted(f for f in os.listdir(out2) if f.endswith('.npy')) == \
            ['L01_V001.npy', 'L01_V002.npy', 'L02_V001.npy'], calls

        # max-hours rat nho -> dung ngay, khong loi
        out3 = os.path.join(tmp, 'feat_run3')
        a3 = parse_args(['--data', b1, '--out', out3, '--max-hours', '1e-9'])
        assert run(a3, pre, enc, 3) == [] and not [f for f in os.listdir(out3) if f.endswith('.npy')]
    print('embed_keyframes.py selftest OK')


if __name__ == '__main__':
    main()
