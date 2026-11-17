"""Tach DONG TIT tu ket qua OCR roi ma hoa theo NGHIA (BGE-M3) -> tim theo nghia trong engine.

Vi sao: OCR doc tot chu chen tren ban tin (tit, dia danh) nhung cau hoi ta bang chu KHAC
("traffic accident" <-> "XE TAI HUC DAI PHAN CACH TONG SAP NHA DAN"), so khop tung chu khong noi duoc.
Model van ban da ngon ngu so khop duoc theo nghia, ca Anh <-> Viet.

Luat tach (do tren anh 854x480 cua bo du lieu):
  - bo dong chu chay o day man hinh (mep tren >= 0.9 chieu cao): toan tin KHAC, khong lien quan hinh
  - bo goc tren ben phai (logo dai, dong ho)
  - giu dong >= 2 tu, >= 6 chu cai, conf >= 0.3: tit, dia danh, chu trong canh doc duoc

Lam sach truoc khi ma hoa (mac dinh, tat bang --no-clean): tit IN HOA + OCR doc 'I' thanh 'l' (GlỮ, THÁl)
lam model cat chu thanh manh vun, mat nghia -> sua l->I trong tu in hoa roi chuyen dong in hoa ve chu thuong.

Dau ra <out>/: captions.json (chuoi goc de hien), captions_clean.json (chuoi da lam sach, dung de ma hoa),
emb.npy (float16, L2-norm), keys.csv (video,frame_idx,cap_id), info.json

    python build_caption_index.py --ocr D:/.../Data/ocr --index D:/.../index --out D:/.../Data/index_caption
CPU cham (BGE-M3 ~568M tham so): xem ETA in ra; bi ngat thi chay lai, phan da ma hoa duoc giu lai.
May yeu/RAM it: --model intfloat/multilingual-e5-small (nhe hon ~5 lan, kem hon).

Tu kiem tra (khong tai model): python build_caption_index.py --selftest
"""
import argparse
import csv
import glob
import json
import os
import re
import time

import numpy as np

# model -> (pooling, tien to cau hoi, tien to van ban)
PRESETS = {
    'BAAI/bge-m3': ('cls', '', ''),
    'intfloat/multilingual-e5-small': ('mean', 'query: ', 'passage: '),
    'intfloat/multilingual-e5-base': ('mean', 'query: ', 'passage: '),
    'intfloat/multilingual-e5-large': ('mean', 'query: ', 'passage: '),
}
LETTERS = re.compile(r'[^\W\d_]', re.UNICODE)


def extract_caption(lines, width=854, height=480, min_conf=0.3):
    """Dong OCR cua 1 keyframe -> chuoi tit (rong neu khong co). Xem luat o dau file."""
    keep = []
    for l in lines:
        x1, y1, x2, y2 = l['box']
        text = ' '.join(str(l['text']).split())
        if y1 >= 0.9 * height:                                # dong chu chay
            continue
        if x1 >= 0.8 * width and y2 <= 0.2 * height:          # logo, dong ho
            continue
        if l.get('conf', 1) < min_conf or len(text.split()) < 2 or len(LETTERS.findall(text)) < 6:
            continue
        keep.append((y1, x1, text))
    keep.sort()
    return ' | '.join(t for _, _, t in keep)


def norm_key(s):
    return ' '.join(s.lower().split())


def clean_caption(text):
    """'PHÚ YÊN: TaM GlỮ 16 TẨN | NHẬP LẬU TỪ THÁl LAN' -> 'phú yên: tam giữ 16 tẩn. nhập lậu từ thái lan'.
    Tung dong (ngan bang ' | '): sua 'l' -> 'I' trong tu ma moi chu cai khac deu in hoa (OCR nham),
    dong ma >= 60% chu cai in hoa thi chuyen ve chu thuong (model hieu tieng Viet chu thuong tot hon)."""
    out = []
    for line in text.split(' | '):
        toks = []
        for t in line.split():
            other = [ch for ch in t if ch.isalpha() and ch != 'l']
            if 'l' in t and len(other) >= 1 and all(ch.isupper() for ch in other):
                t = t.replace('l', 'I')
            toks.append(t)
        line = ' '.join(toks)
        letters = [ch for ch in line if ch.isalpha()]
        if letters and sum(ch.isupper() for ch in letters) >= 0.6 * len(letters):
            line = line.lower()
        out.append(line)
    return '. '.join(out)


def collect(ocr_dirs, width, height, clean=True):
    """-> (keys [(video, frame_idx, cap_id)], captions [chuoi goc], captions_clean [de ma hoa], so file).
    Gop chuoi trung theo ban da lam sach (cac bien the doc sai l/I cua cung 1 tit gop lai)."""
    files = sorted({f for d in ocr_dirs for f in glob.glob(os.path.join(d, '*.jsonl'))})
    if not files:
        raise SystemExit(f'Khong thay file .jsonl nao trong {ocr_dirs}')
    caps, clean_caps, idx, keys = [], [], {}, []
    for fp in files:
        vid = os.path.splitext(os.path.basename(fp))[0]
        with open(fp, encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                c = extract_caption(r.get('lines', []), width, height)
                if not c:
                    continue
                cc = clean_caption(c) if clean else c
                k = norm_key(cc)
                if k not in idx:
                    idx[k] = len(caps)
                    caps.append(c)
                    clean_caps.append(cc)
                keys.append((r.get('video', vid), int(r['frame_idx']), idx[k]))
    return keys, caps, clean_caps, len(files)


class TextEncoder:
    """BGE-M3 / e5 bang transformers thuan (khong can sentence-transformers)."""

    def __init__(self, model, device=None, pooling=None, fp16=True):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        preset = PRESETS.get(model, ('cls', '', ''))
        self.pooling = pooling or preset[0]
        self.q_prefix, self.p_prefix = preset[1], preset[2]
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model).to(self.device).eval()
        if fp16 and self.device == 'cuda':
            self.model = self.model.half()

    def encode(self, texts, is_query=False, batch=64, max_len=128):
        torch = self.torch
        pre = self.q_prefix if is_query else self.p_prefix
        out = []
        with torch.inference_mode():
            for s in range(0, len(texts), batch):
                b = self.tok([pre + t for t in texts[s:s + batch]], padding=True, truncation=True,
                             max_length=max_len, return_tensors='pt').to(self.device)
                h = self.model(**b).last_hidden_state
                if self.pooling == 'cls':
                    v = h[:, 0]
                else:
                    m = b['attention_mask'].unsqueeze(-1).to(h.dtype)
                    v = (h * m).sum(1) / m.sum(1).clamp(min=1)
                out.append(torch.nn.functional.normalize(v.float(), dim=-1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 1), np.float32)


def embed_resumable(caps, encode, out_dir, chunk=4096, batch=64):
    """Ma hoa theo tung khuc, luu tung khuc -> bi ngat chay lai khong mat phan da xong.
    Sap theo do dai de it dem (nhanh hon nhieu tren CPU), cuoi cung tra ve dung thu tu caps."""
    order = np.argsort([len(c) for c in caps], kind='stable')
    parts = os.path.join(out_dir, '_parts')
    os.makedirs(parts, exist_ok=True)
    n_chunk = (len(caps) + chunk - 1) // chunk
    t0, done = time.time(), 0
    for ci in range(n_chunk):
        p = os.path.join(parts, f'{ci:05d}.npy')
        ids = order[ci * chunk:(ci + 1) * chunk]
        if os.path.exists(p):
            continue
        e = encode([caps[i] for i in ids], batch).astype(np.float16)
        np.save(p + '.tmp.npy', e)
        os.replace(p + '.tmp.npy', p)
        done += len(ids)
        speed = done / (time.time() - t0)
        left = sum(len(order[j * chunk:(j + 1) * chunk]) for j in range(ci + 1, n_chunk)
                   if not os.path.exists(os.path.join(parts, f'{j:05d}.npy')))
        print(f'  khuc {ci + 1}/{n_chunk}: {speed:6.1f} cau/s | ETA {left / max(speed, 1e-6) / 60:5.1f} phut', flush=True)
    emb = None
    for ci in range(n_chunk):
        e = np.load(os.path.join(parts, f'{ci:05d}.npy'))
        if emb is None:
            emb = np.zeros((len(caps), e.shape[1]), np.float16)
        emb[order[ci * chunk:(ci + 1) * chunk]] = e
    return emb


def build(a, encoder=None):
    keys, caps, clean_caps, n_files = collect(a.ocr, a.width, a.height, not getattr(a, 'no_clean', False))
    print(f'{n_files} file OCR -> {len(keys):,} keyframe co tit, {len(caps):,} chuoi tit khac nhau')
    os.makedirs(a.out, exist_ok=True)
    if encoder is None:
        print(f'Tai model {a.model} ...')
        encoder = TextEncoder(a.model, a.device)
        print(f'  thiet bi {encoder.device}, pooling {encoder.pooling}')
    # phan ma hoa do dang cua model KHAC (hoac du lieu khac) -> xoa, khong tron vector 2 model
    import hashlib
    import shutil
    digest = hashlib.md5('\n'.join(clean_caps).encode()).hexdigest()
    sig = f'{a.model}|{len(caps)}|{digest}'
    sig_file = os.path.join(a.out, '_parts', 'sig.txt')
    if os.path.exists(sig_file) and open(sig_file, encoding='utf-8').read() != sig:
        print('  phan do dang thuoc model/du lieu khac -> lam lai tu dau')
        shutil.rmtree(os.path.join(a.out, '_parts'))
    os.makedirs(os.path.join(a.out, '_parts'), exist_ok=True)
    with open(sig_file, 'w', encoding='utf-8') as f:
        f.write(sig)
    emb = embed_resumable(clean_caps, lambda t, b: encoder.encode(t, False, b), a.out, batch=a.batch)
    np.save(os.path.join(a.out, 'emb.npy'), emb)
    with open(os.path.join(a.out, 'captions.json'), 'w', encoding='utf-8') as f:
        json.dump(caps, f, ensure_ascii=False)
    with open(os.path.join(a.out, 'captions_clean.json'), 'w', encoding='utf-8') as f:
        json.dump(clean_caps, f, ensure_ascii=False)
    with open(os.path.join(a.out, 'keys.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['video', 'frame_idx', 'cap_id'])
        w.writerows(keys)
    pooling, qp, pp = PRESETS.get(a.model, (getattr(encoder, 'pooling', 'cls'), '', ''))
    with open(os.path.join(a.out, 'info.json'), 'w', encoding='utf-8') as f:
        json.dump({'model': a.model, 'pooling': getattr(encoder, 'pooling', pooling), 'query_prefix': qp,
                   'passage_prefix': pp, 'n_captions': len(caps), 'n_keyframes': len(keys),
                   'dim': int(emb.shape[1]), 'width': a.width, 'height': a.height,
                   'clean': not getattr(a, 'no_clean', False)}, f, indent=2)
    shutil.rmtree(os.path.join(a.out, '_parts'), ignore_errors=True)
    print(f'Xong -> {a.out} (dim {emb.shape[1]})')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ocr', nargs='+', help='thu muc chua <vid>.jsonl cua ocr_keyframes.py')
    p.add_argument('--out', help='thu muc index tit')
    p.add_argument('--model', default='BAAI/bge-m3')
    p.add_argument('--device', default=None)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--width', type=int, default=854, help='kich thuoc anh keyframe luc chay OCR')
    p.add_argument('--height', type=int, default=480)
    p.add_argument('--index', default=None, help='(tuy chon) chi de in nhac lenh chay app')
    p.add_argument('--no-clean', action='store_true', help='ma hoa nguyen chuoi OCR (khong sua l/I, khong ha chu)')
    p.add_argument('--selftest', action='store_true')
    a = p.parse_args()
    if a.selftest:
        return selftest()
    if not (a.ocr and a.out):
        p.error('can --ocr va --out')
    build(a)


def selftest():
    import tempfile
    W, H = 854, 480
    lines = [
        {'text': 'XE TẢl HÚC DẢl PHÂN CÁCH TÔNG SẬP NHÀ DÂN', 'box': [238, 392, 790, 420], 'conf': 0.8},
        {'text': 'QUẬN 6, TPHCM', 'box': [238, 406, 356, 430], 'conf': 0.87},
        {'text': 'Ngành y tế TP HCM dể xuất loạt giải pháp', 'box': [474, 434, 854, 462], 'conf': 0.65},  # chu chay
        {'text': '18:46:54', 'box': [697, 63, 759, 77], 'conf': 0.4},                                   # dong ho
        {'text': 'IHỂ THAO', 'box': [734, 44, 816, 68], 'conf': 0.83},                                  # logo
        {'text': 'XĂNG E5 RON 92', 'box': [79, 167, 319, 203], 'conf': 0.98},                            # chu trong canh
        {'text': 'quanse', 'box': [56, 20, 112, 44], 'conf': 0.4},                                      # 1 tu
        {'text': 'mờ quá đọc sai', 'box': [300, 300, 400, 320], 'conf': 0.1},                           # conf thap
    ]
    # lam sach: vi du lay tu ket qua chan doan that
    assert clean_caption('PHÚ YÊN: TaM GlỮ 16 TẨN ĐƯỜNG CÁT | NHẬP LẬU TỪ THÁl LAN') == \
        'phú yên: tam giữ 16 tẩn đường cát. nhập lậu từ thái lan'
    assert clean_caption('HỘP SỌ KHỦNG LONG 68 TRIỆU NĂM TUỔl') == 'hộp sọ khủng long 68 triệu năm tuổi'
    assert clean_caption('XE TẢl HÚC DẢl PHÂN CÁCH') == 'xe tải húc dải phân cách'
    assert clean_caption('diffuseur officiel') == 'diffuseur officiel'          # chu thuong: giu nguyen
    assert clean_caption('Distance 0.10 miles') == 'Distance 0.10 miles'
    assert clean_caption('Bà TRẨN THI NỞ') == 'bà trẩn thi nở'

    cap = extract_caption(lines, W, H)
    assert cap == 'XĂNG E5 RON 92 | XE TẢl HÚC DẢl PHÂN CÁCH TÔNG SẬP NHÀ DÂN | QUẬN 6, TPHCM', cap
    assert extract_caption([lines[2], lines[3], lines[4]], W, H) == ''

    def fake_encode(texts, batch=64):           # vector theo tu: chung tu -> cosine cao
        vocab = {}
        out = np.zeros((len(texts), 64), np.float32)
        for i, t in enumerate(texts):
            for wd in norm_key(t).split():
                out[i, hash(wd) % 64] += 1
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-6)

    class Fake:
        pooling = 'cls'

        def encode(self, texts, is_query=False, batch=64):
            return fake_encode(texts)

    with tempfile.TemporaryDirectory() as tmp:
        od = os.path.join(tmp, 'ocr')
        os.makedirs(od)
        with open(os.path.join(od, 'L01_V001.jsonl'), 'w', encoding='utf-8') as f:
            for fi, ls in ((0, [lines[0], lines[2]]), (30, [lines[0]]), (60, [lines[3]]), (90, [lines[5]])):
                f.write(json.dumps({'frame_idx': fi, 'text': '', 'lines': ls}, ensure_ascii=False) + '\n')
        with open(os.path.join(od, 'L01_V002.jsonl'), 'w', encoding='utf-8') as f:
            # cung tit nhung OCR doc dung chu I (khong nham l) -> lam sach xong phai gop chung 1 chuoi
            f.write(json.dumps({'frame_idx': 5, 'text': '', 'lines': [dict(lines[0], text=lines[0]['text'].replace('l', 'I'))]},
                               ensure_ascii=False) + '\n')
        keys, caps, clean_caps, n = collect([od], W, H)
        assert clean_caps[0] == 'xe tải húc dải phân cách tông sập nhà dân', clean_caps
        # frame 0 va 30 cung tit (chu chay bi bo); V002 cung tit (doc dung I thay vi l) -> gop 1 chuoi
        assert n == 2 and len(caps) == 2, caps
        assert [k[:2] for k in keys] == [('L01_V001', 0), ('L01_V001', 30), ('L01_V001', 90), ('L01_V002', 5)]
        assert keys[0][2] == keys[1][2] == keys[3][2] != keys[2][2]

        out = os.path.join(tmp, 'cap')
        a = argparse.Namespace(ocr=[od], out=out, model='fake', device='cpu', batch=2, width=W, height=H)
        build(a, encoder=Fake())
        emb = np.load(os.path.join(out, 'emb.npy'))
        assert emb.shape == (2, 64) and emb.dtype == np.float16
        assert np.allclose(emb.astype(np.float32), fake_encode(json.load(open(os.path.join(out, 'captions_clean.json'),
                                                                                encoding='utf-8'))), atol=1e-3)
        assert not os.path.exists(os.path.join(out, '_parts'))

        # phan do dang cua model khac -> bi xoa, khong tron
        os.makedirs(os.path.join(out, '_parts'))
        open(os.path.join(out, '_parts', 'sig.txt'), 'w').write('model-cu|1|x')
        np.save(os.path.join(out, '_parts', '00000.npy'), np.zeros((2, 7), np.float16))
        build(a, encoder=Fake())
        assert np.load(os.path.join(out, 'emb.npy')).shape == (2, 64)

        # resume: khuc da co thi khong ma hoa lai
        calls = []
        os.makedirs(os.path.join(tmp, 'r'))
        embed_resumable(['a b', 'c d e', 'f'], lambda t, b: (calls.append(len(t)), fake_encode(t))[1],
                        os.path.join(tmp, 'r'), chunk=2)
        n1 = len(calls)
        calls.clear()
        e2 = embed_resumable(['a b', 'c d e', 'f'], lambda t, b: (calls.append(len(t)), fake_encode(t))[1],
                             os.path.join(tmp, 'r'), chunk=2)
        assert n1 == 2 and calls == [] and e2.shape == (3, 64)
        assert np.allclose(e2.astype(np.float32), fake_encode(['a b', 'c d e', 'f']), atol=1e-3)
    print('build_caption_index.py selftest OK')


if __name__ == '__main__':
    main()
