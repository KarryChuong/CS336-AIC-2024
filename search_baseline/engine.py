"""Loi truy van: text/anh/clip -> embedding -> cosine voi toan bo keyframe (brute force), nhieu encoder.

Vai tram nghin vector chi la 1 phep nhan ma tran (~10-100 ms ca tren CPU), nen khong can Milvus/FAISS.

Nhieu encoder (vd ViT-B-32 + SigLIP2): moi encoder 1 thu muc index rieng (build_index.py), truyen ca
list vao SearchEngine. Thu muc DAU la chinh (meta, anh). Cac index khac duoc can hang theo
(video, frame_idx) nen lech vai video cung khong sao. Diem gop = trung binh co trong so cua z-score
tung encoder (diem tho moi model mot thang khac nhau). Chi 1 encoder -> giu diem cosine tho nhu cu.

OCR (tuy chon): <data>/ocr/<vid>.jsonl, moi dong {"frame_idx": ..., "text": ...} -> loc ket qua theo cum chu.
Dong tit (tuy chon): index tu build_caption_index.py -> cong diem khop NGHIA giua cau hoi va dong tit.
"""
import glob
import json
import os
import re
import threading
import unicodedata

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------- tien ich thoi gian
def window_max_after(vals, tkey, window_s):
    """best[i] = max vals[j] voi i < j va tkey[j] - tkey[i] <= window_s; arg[i] = j do (-1 neu rong).

    tkey tang dan, video sau cach video truoc 1e6 giay -> cua so khong bao gio sang video khac.
    Vector hoa theo do lech chi so m = 1..M (M = so keyframe nhieu nhat trong 1 cua so).
    """
    n = len(vals)
    best = np.full(n, -np.inf, dtype=np.float64)
    arg = np.full(n, -1, dtype=np.int64)
    if n == 0:
        return best, arg
    idx = np.arange(n)
    hi = np.searchsorted(tkey, tkey + window_s, side='right')     # j < hi
    m_max = int((hi - idx - 1).max())
    for m in range(1, m_max + 1):
        j = idx + m
        ok = j < hi
        cand = np.where(ok, vals[np.minimum(j, n - 1)], -np.inf)
        better = cand > best
        best[better] = cand[better]
        arg[better] = j[better]
    return best, arg


def chain_scores(scores, tkey, window_s):
    """Chuoi su kien dung thu tu bang quy hoach dong (thay cho tham lam ban dau).

    total[i] = s0[i] + max_{j in (i, i+W]} (s1[j] + max_{k in (j, j+W]} (s2[k] + ...)), cung video.
    Cua so rong (het video) -> su kien do lay diem thap nhat cua chinh no: chuoi dut van xep hang
    duoc nhung luon thua chuoi day du. Tra ve (total, picks) voi picks[i] = [j, k, ...].
    """
    n_ev = len(scores)
    scores = [np.asarray(s, dtype=np.float64) for s in scores]
    if n_ev == 1:
        return scores[0].copy(), lambda i: []
    args = [None] * n_ev
    val = scores[-1]
    for e in range(n_ev - 2, -1, -1):
        best, arg = window_max_after(val, tkey, window_s)
        miss = float(sum(s.min() for s in scores[e + 1:]))
        best = np.where(arg >= 0, best, miss)
        args[e] = arg
        val = scores[e] + best

    def picks(i):
        out, cur = [], int(i)
        for e in range(n_ev - 1):
            nxt = int(args[e][cur])
            if nxt < 0:
                break
            out.append(nxt)
            cur = nxt
        return out

    return val, picks


def chain_rep(ev, tkey):
    """Keyframe dai dien cho 1 chuoi su kien ev = [A, B, C, ...] (chi so keyframe, tang dan).

    So le -> su kien giua (3 cau -> B); chan -> keyframe gan diem giua thoi gian nhat (trong [dau, cuoi]).
    Ly do: A ('mo dau la...') hay khop som hon doan dap an vai giay; su kien giua nam trong doan
    chac hon -> nop DRES an toan hon.
    """
    if len(ev) <= 1:
        return ev[0]
    if len(ev) % 2:
        return ev[len(ev) // 2]
    lo, hi = ev[0], ev[-1]
    mid = (tkey[lo] + tkey[hi]) / 2
    j = int(np.searchsorted(tkey, mid))
    cands = [c for c in (j - 1, j) if lo <= c <= hi]
    return min(cands, key=lambda c: abs(tkey[c] - mid)) if cands else lo


def range_max(vals, lo, hi):
    """max(vals[lo[i]:hi[i]]) cho moi i (hi > lo), vector hoa bang sparse table O(N log N)."""
    st = [np.asarray(vals, dtype=np.float64)]       # st[j][i] = max vals[i : i + 2^j]
    need, step = max(int((hi - lo).max()), 1), 1
    while step * 2 <= need:
        prev = st[-1]
        st.append(np.maximum(prev[:-step], prev[step:]))
        step *= 2
    length = hi - lo
    k = np.floor(np.log2(np.maximum(length, 1))).astype(int)
    out = np.empty(len(lo))
    for lv in np.unique(k):
        m = k == lv
        span = 1 << lv
        out[m] = np.maximum(st[lv][lo[m]], st[lv][hi[m] - span])
    return out


def window_max_center(vals, tkey, window_s):
    """max vals trong [t - w, t + w] cung video (video cach nhau 1e6 giay trong tkey)."""
    lo = np.searchsorted(tkey, tkey - window_s, side='left')
    hi = np.searchsorted(tkey, tkey + window_s, side='right')
    return range_max(vals, lo, hi)


def zscore_finite(s):
    """z-score tren cac gia tri huu han; -inf/nan giu -inf (vd keyframe bi loai)."""
    z = np.full(len(s), -np.inf)
    f = np.isfinite(s)
    if f.any():
        z[f] = (s[f] - s[f].mean()) / (s[f].std() + 1e-6)
    return z


def attr_parts(attrs):
    """'Mo ta theo o' tren UI -> danh sach (cau tieng Anh ngan, trong so).

    Moi o thanh 1 cau RIENG, cham diem rieng roi moi cong (xem SearchEngine._attr_total). Khong noi
    thanh 1 cau dai: text encoder SigLIP2 chi doc 64 token (B/32: 77), phan sau bi cat mat im lang.
    Trong so am = o 'Loai tru' (tru diem nhung anh giong cau do).
    attrs: dict who/act/where/objs/color/neg (da dich sang tieng Anh o UI), thieu o nao bo qua.
    """
    a = {k: str(v or '').strip().strip('.').strip() for k, v in (attrs or {}).items()}
    low = {k: v[:1].lower() + v[1:] for k, v in a.items()}     # Google dich hay viet hoa chu dau
    parts = []
    subj = ' '.join(x for x in (a.get('who'), low.get('act')) if x)   # "workers in blue" + "stacking wood"
    if subj:
        parts.append((subj, 1.0))
    if a.get('where'):
        parts.append((f'a photo taken in {low["where"]}', 0.7))
    if a.get('objs'):
        parts.append((f'a photo full of {low["objs"]}', 0.5))
    if a.get('color'):
        parts.append((f'a photo with mostly {low["color"]} colors', 0.3))
    if a.get('neg'):
        parts.append((a['neg'], -0.5))
    return parts


def dilate_mask(mask, tkey, window_s):
    """Keyframe qua neu co keyframe nao trong +-window_s giay (cung video) qua mask."""
    if window_s <= 0:
        return mask
    c = np.concatenate([[0], np.cumsum(mask)])
    lo = np.searchsorted(tkey, tkey - window_s, side='left')
    hi = np.searchsorted(tkey, tkey + window_s, side='right')
    return (c[hi] - c[lo]) > 0


# ---------------------------------------------------------------------- chu tieng Viet
def norm_vi(s):
    """Thuong, bo dau, d/D -> d, chi giu chu so: 'Giải phóng khí Hiđro' -> 'giai phong khi hidro'."""
    s = unicodedata.normalize('NFD', str(s)).replace('đ', 'd').replace('Đ', 'D')
    s = ''.join(ch for ch in s if unicodedata.category(ch) != 'Mn').lower()
    return re.sub(r'[^0-9a-z]+', ' ', s).strip()


def norm_ocr(s):
    """norm_vi + gop ky tu OCR hay nham: 'I' in hoa thuong bi doc thanh 'l' (GIẢl PHÓNG).
    Ap dung CA 2 phia (chu OCR va cau tim) nen 'kim long' van khop 'KIM LONG'."""
    return norm_vi(s).replace('l', 'i')


def ocr_mask(texts_norm, query):
    """texts_norm: pd.Series da norm_vi. query: '"cum chinh xac"' hoac cac tu (AND, khop chuoi con).
    Nhieu dieu kien cach nhau bang dau ';' -> OR."""
    out = np.zeros(len(texts_norm), dtype=bool)
    for part in [p for p in query.split(';') if p.strip()]:
        phrases = re.findall(r'"([^"]+)"', part)
        words = norm_ocr(re.sub(r'"[^"]*"', ' ', part)).split()
        terms = [norm_ocr(p) for p in phrases if norm_ocr(p)] + words
        if not terms:
            continue
        m = np.ones(len(texts_norm), dtype=bool)
        for t in terms:
            m &= texts_norm.str.contains(t, regex=False).to_numpy()
        out |= m
    return out


# ---------------------------------------------------------------------- 1 encoder
class Encoder:
    """1 thu muc index (features.npy + meta.csv + info.json) + model open_clip tuong ung."""

    def __init__(self, index_dir, device, ref_meta=None, load_model=True):
        with open(os.path.join(index_dir, 'info.json'), encoding='utf-8') as f:
            self.info = json.load(f)
        self.index_dir = index_dir
        self.name = self.info['model']
        self.device = device
        feats = np.load(os.path.join(index_dir, 'features.npy'))
        self.missing = None
        if ref_meta is not None:
            # can hang theo index chinh: (video, frame_idx) -> vi tri; thieu thi vector 0 + danh dau
            meta = pd.read_csv(os.path.join(index_dir, 'meta.csv'), usecols=['video', 'frame_idx'])
            pos = pd.Series(np.arange(len(meta)), index=pd.MultiIndex.from_frame(meta))
            want = pd.MultiIndex.from_frame(ref_meta[['video', 'frame_idx']])
            src = pos.reindex(want).to_numpy()
            have = ~np.isnan(src)
            aligned = np.zeros((len(ref_meta), feats.shape[1]), dtype=feats.dtype)
            aligned[have] = feats[src[have].astype(np.int64)]
            feats = aligned
            if not have.all():
                self.missing = ~have
                print(f'  !! {self.name}: {int((~have).sum()):,} keyframe khong co trong {index_dir}')
        dtype = torch.float16 if device == 'cuda' else torch.float32
        self.feats = torch.from_numpy(feats).to(device, dtype)
        self._lock = threading.Lock()   # model khong an toan khi goi tu nhieu thread cung luc
        self._cache = {}
        self.model = None
        if load_model:
            import open_clip
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                self.name, pretrained=self.info['pretrained'], device=device)
            self.model.eval()
            self.tokenizer = open_clip.get_tokenizer(self.name)

    def text_vec(self, text):
        if text not in self._cache:
            with self._lock, torch.no_grad():
                e = self.model.encode_text(self.tokenizer([text]).to(self.device))
            if len(self._cache) > 4096:
                self._cache.clear()
            self._cache[text] = F.normalize(e.float(), dim=-1)[0]
        return self._cache[text]

    def text_scores(self, text):
        q = self.text_vec(text).to(self.feats.dtype)
        return (self.feats @ q).float().cpu().numpy()

    def id_scores(self, kf_id):
        return (self.feats @ self.feats[kf_id]).float().cpu().numpy()

    def vec_scores(self, vec):
        q = F.normalize(torch.as_tensor(vec, dtype=torch.float32), dim=-1).to(self.device, self.feats.dtype)
        return (self.feats @ q).float().cpu().numpy()

    def image_matrix(self, imgs, chunk=32):
        """[N, so anh] cosine cua moi keyframe voi tung anh ngoai. Encode theo lo cho nhanh."""
        cols = []
        with self._lock, torch.no_grad():
            for s in range(0, len(imgs), chunk):
                batch = torch.stack([self.preprocess(im) for im in imgs[s:s + chunk]]).to(self.device)
                e = F.normalize(self.model.encode_image(batch).float(), dim=-1).to(self.feats.dtype)
                cols.append((self.feats @ e.T).float().cpu().numpy())
        return np.concatenate(cols, axis=1)


def zscore(s, missing=None, axis=0):
    z = (s - s.mean(axis=axis, keepdims=True)) / (s.std(axis=axis, keepdims=True) + 1e-6)
    if missing is not None:
        z[missing] = 0.0          # keyframe encoder nay khong co -> trung tinh
    return z


class SearchEngine:
    def __init__(self, index_dir, data_dir=None, device=None, weights=None, ocr_dirs=None, load_model=True,
                 caption_dir=None, caption_weight=0.5, caption_window=15.0):
        dirs = [index_dir] if isinstance(index_dir, str) else list(index_dir)
        with open(os.path.join(dirs[0], 'info.json'), encoding='utf-8') as f:
            info0 = json.load(f)
        # Uu tien --data truyen vao; neu khong thi lay luc build, cuoi cung la thu muc cha cua index
        cands = [data_dir, info0.get('data_dir'), os.path.dirname(os.path.abspath(dirs[0]))]
        self.data_dir = next((d for d in cands if d and os.path.isdir(os.path.join(d, 'keyframes'))), None)
        if self.data_dir is None:
            raise SystemExit(f'Khong thay thu muc keyframes/ trong {[d for d in cands if d]} -> them --data <Batch1>')

        self.meta = pd.read_csv(os.path.join(dirs[0], 'meta.csv'))
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoders = [Encoder(dirs[0], self.device, load_model=load_model)]
        for d in dirs[1:]:
            self.encoders.append(Encoder(d, self.device, ref_meta=self.meta, load_model=load_model))
        self.weights = list(weights) if weights else [1.0] * len(self.encoders)
        assert len(self.weights) == len(self.encoders), 'so --weights phai bang so --index'

        self.video = self.meta['video'].to_numpy()
        self.frame_idx = self.meta['frame_idx'].to_numpy()
        self.pts = self.meta['pts_time'].to_numpy()
        codes, self.video_names = pd.factorize(self.meta['video'], sort=True)
        self.vcode = codes
        # Khoa tang dan theo (video, thoi gian) -- index da sap san theo thu tu nay
        self.tkey = codes.astype(np.float64) * 1e6 + self.pts
        assert np.all(np.diff(self.tkey) >= 0), 'meta.csv chua sap theo (video, frame_idx) -- build lai index'

        names = [e.name for e in self.encoders]
        self.info = {**info0, 'model': ' + '.join(names),
                     'pretrained': ' + '.join(str(e.info['pretrained']) for e in self.encoders),
                     'encoders': names, 'weights': self.weights}
        self.ocr_raw = self.ocr_norm = None
        self._load_ocr(ocr_dirs)
        self.cap = None
        self.caption_weight, self.caption_window = caption_weight, caption_window
        if caption_dir:
            self._load_caption(caption_dir)

    # ------------------------------------------------------------------ OCR
    def _load_ocr(self, ocr_dirs=None):
        dirs = list(ocr_dirs or [])
        for d in self.info.get('data_dirs') or [self.data_dir]:
            if os.path.isdir(os.path.join(d, 'ocr')):
                dirs.append(os.path.join(d, 'ocr'))
        files = sorted({f for d in dirs for f in glob.glob(os.path.join(d, '*.jsonl'))})
        if not files:
            return
        pos = {(v, int(f)): i for i, (v, f) in enumerate(zip(self.video, self.frame_idx))}
        raw = [''] * len(self.meta)
        n_line = n_miss = 0
        for fp in files:
            vid = os.path.splitext(os.path.basename(fp))[0]
            with open(fp, encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    n_line += 1
                    i = pos.get((r.get('video', vid), int(r['frame_idx'])))
                    if i is None:
                        n_miss += 1
                        continue
                    t = r.get('text') or ' '.join(x.get('text', '') for x in r.get('lines', []))
                    raw[i] = (raw[i] + ' ' + t).strip() if raw[i] else t
        self.ocr_raw = pd.Series(raw, dtype=object)
        self.ocr_norm = self.ocr_raw.map(norm_ocr)
        self.info['ocr_keyframes'] = int((self.ocr_norm != '').sum())
        print(f'OCR: {len(files)} file, {n_line:,} dong, {self.info["ocr_keyframes"]:,} keyframe co chu'
              + (f', {n_miss:,} dong khong khop keyframe nao' if n_miss else ''))

    # ------------------------------------------------------------------ dong tit (theo nghia)
    def _load_caption(self, caption_dir):
        with open(os.path.join(caption_dir, 'info.json'), encoding='utf-8') as f:
            info = json.load(f)
        keys = pd.read_csv(os.path.join(caption_dir, 'keys.csv'))
        pos = pd.Series(np.arange(len(self.meta)), index=pd.MultiIndex.from_arrays([self.video, self.frame_idx]))
        at = pos.reindex(pd.MultiIndex.from_frame(keys[['video', 'frame_idx']])).to_numpy()
        ok = ~np.isnan(at)
        cap_of = np.full(len(self.meta), -1, dtype=np.int64)
        cap_of[at[ok].astype(np.int64)] = keys['cap_id'].to_numpy()[ok]
        emb = np.load(os.path.join(caption_dir, 'emb.npy'))
        with open(os.path.join(caption_dir, 'captions.json'), encoding='utf-8') as f:
            texts = json.load(f)
        dtype = torch.float16 if self.device == 'cuda' else torch.float32
        self.cap = {'info': info, 'cap_of': cap_of, 'texts': texts, 'enc': None, 'cache': {},
                    'emb': torch.from_numpy(emb).to(self.device, dtype), 'lock': threading.Lock()}
        self.info['caption_keyframes'] = int((cap_of >= 0).sum())
        self.info['caption_model'] = info['model']
        print(f'Tit: {len(texts):,} chuoi, {self.info["caption_keyframes"]:,} keyframe co tit ({info["model"]})'
              + (f', {int((~ok).sum()):,} khoa khong khop index' if (~ok).any() else ''))

    def _caption_query_vec(self, text):
        c = self.cap
        if text not in c['cache']:
            with c['lock']:
                if c['enc'] is None:
                    from build_caption_index import TextEncoder
                    c['enc'] = TextEncoder(c['info']['model'], self.device, c['info'].get('pooling'))
                v = c['enc'].encode([text], is_query=True)[0]
            if len(c['cache']) > 4096:
                c['cache'].clear()
            c['cache'][text] = torch.as_tensor(v, dtype=c['emb'].dtype, device=c['emb'].device)
        return c['cache'][text]

    def caption_scores(self, text):
        """z-score khop nghia voi dong tit, lan +- caption_window giay (tit chi hien vai giay dau moi tin).
        Keyframe khong co tit nao gan, hoac tit kem hon trung binh -> 0 (trung tinh)."""
        c = self.cap
        per_cap = (c['emb'] @ self._caption_query_vec(text.strip())).float().cpu().numpy()
        s = np.full(len(self.meta), -np.inf)
        has = c['cap_of'] >= 0
        s[has] = per_cap[c['cap_of'][has]]
        if self.caption_window > 0:
            s = window_max_center(s, self.tkey, self.caption_window)
        z = zscore_finite(s)
        z[~np.isfinite(z)] = 0.0
        # chi CONG diem khi tit khop hon trung binh; tit khong lien quan khong bi tru diem
        # (neu tru, video it tit nhu the thao se duoc loi vo ly)
        return np.maximum(z, 0.0)

    def caption_of(self, i):
        if self.cap is None or self.cap['cap_of'][i] < 0:
            return ''
        return self.cap['texts'][self.cap['cap_of'][i]]

    def ocr_filter(self, query, window_s=0.0):
        if self.ocr_norm is None:
            raise ValueError('Chua co OCR: dat <data>/ocr/<vid>.jsonl hoac chay app voi --ocr')
        return dilate_mask(ocr_mask(self.ocr_norm, query), self.tkey, window_s)

    # ------------------------------------------------------------------ diem
    def _active(self, use):
        if not use:
            return list(range(len(self.encoders)))
        want = [u.strip().lower() for u in (use.split(',') if isinstance(use, str) else use) if u.strip()]
        act = [i for i, e in enumerate(self.encoders) if any(w in e.name.lower() for w in want)]
        if not act:
            raise ValueError(f'Khong co encoder {use}; dang co {[e.name for e in self.encoders]}')
        return act

    def _fuse(self, per_enc, act):
        """1 encoder -> diem tho; nhieu -> trung binh co trong so cua z-score (theo cot)."""
        if len(act) == 1:
            miss = self.encoders[act[0]].missing
            # chi 1 encoder ma encoder do thieu keyframe -> loai han, khong de diem 0 lot vao top
            return per_enc[0] if miss is None else np.where(
                miss.reshape(-1, *[1] * (per_enc[0].ndim - 1)), -np.inf, per_enc[0])
        w = [self.weights[i] for i in act]
        tot = sum(wi * zscore(s, self.encoders[i].missing) for wi, s, i in zip(w, per_enc, act))
        return tot / sum(w)

    def text_scores(self, text, use=None, cap=None):
        """Diem hinh anh (gop cac encoder); co index tit thi cong them cap * diem tit (z-score).
        cap=None -> self.caption_weight; cap=0 -> tat nhanh tit."""
        act = self._active(use)
        vis = self._fuse([self.encoders[i].text_scores(text.strip()) for i in act], act)
        w = self.caption_weight if cap is None else cap
        if self.cap is None or not w:
            return vis
        if len(act) == 1:                 # 1 encoder -> diem tho, doi sang z-score de cong duoc
            vis = zscore_finite(vis)
        return vis + w * self.caption_scores(text)

    def _attr_total(self, attrs, use=None):
        """(tong w * z-score cac o, tong trong so duong) hoac (None, 0) khi khong dien o nao.
        Tinh 1 lan roi dung chung cho moi su kien A/B/C: cac o mo ta ca clip, khong rieng canh nao."""
        parts = attr_parts(attrs)
        if not parts:
            return None, 0.0
        tot = 0.0
        for text, w in parts:
            z = zscore_finite(self.text_scores(text, use, cap=0))
            # o Loai tru: keyframe thieu diem (-inf) thi khong tru gi, tranh -w * -inf = +inf
            tot = tot + w * (z if w > 0 else np.where(np.isfinite(z), z, 0.0))
        return tot, float(sum(w for _, w in parts if w > 0))

    def _combine(self, text, attr, use=None, cap=None):
        """Diem 1 cau mo ta + cac o (attr tu _attr_total). Khong co o nao -> y het text_scores cu."""
        tot, wsum = attr
        if tot is None:
            return self.text_scores(text, use, cap)
        if text and text.strip():
            tot, wsum = tot + zscore_finite(self.text_scores(text, use, cap)), wsum + 1.0
        if wsum <= 0:
            raise ValueError('Chi co o Loai tru: can them cau mo ta hoac it nhat 1 o khac')
        return tot / wsum

    def video_frames(self, video):
        """Moi keyframe cua 1 video, theo thu tu thoi gian (de xem/chuyen ma video cho dong doi)."""
        v = str(video).strip().upper()
        if v not in self.video_names:
            raise ValueError(f'Khong co video {v} (vd dung: L24_V025)')
        c = self.video_names.get_loc(v)
        lo, hi = np.searchsorted(self.vcode, c, 'left'), np.searchsorted(self.vcode, c, 'right')
        return [self.row(i) for i in range(lo, hi)]

    def id_scores(self, kf_id, use=None):
        act = self._active(use)
        return self._fuse([self.encoders[i].id_scores(kf_id) for i in act], act)

    def image_matrix(self, imgs, use=None):
        act = self._active(use)
        return self._fuse([self.encoders[i].image_matrix(imgs) for i in act], act)

    # ------------------------------------------------------------------ ranking
    def _apply_ocr(self, scores, ocr, window_s):
        if not ocr or not ocr.strip():
            return scores
        m = self.ocr_filter(ocr, window_s)
        return np.where(m, scores, -np.inf)

    def _rank(self, scores, k, max_per_video=None, extra=None):
        finite = np.isfinite(scores)
        if not finite.any():
            return []
        if not max_per_video:
            k = min(k, int(finite.sum()))
            top = np.argpartition(-np.where(finite, scores, -np.inf), k - 1)[:k]
            top = top[np.argsort(-scores[top], kind='stable')]
        else:
            # Lay du rong roi loc, tranh 1 video chiem het top-k
            pool = np.argsort(-scores, kind='stable')[:max(k * 20, 2000)]
            count, top = {}, []
            for i in pool:
                v = self.video[i]
                if count.get(v, 0) < max_per_video:
                    count[v] = count.get(v, 0) + 1
                    top.append(i)
                    if len(top) == k:
                        break
            top = np.array(top, dtype=np.int64)
        top = top[np.isfinite(scores[top])]
        return [self.row(int(i), float(scores[i]), **(extra(int(i)) if extra else {})) for i in top]

    def row(self, i, score=None, **kw):
        i = int(i)
        r = {'id': i, 'video': str(self.video[i]), 'frame_idx': int(self.frame_idx[i]),
             'pts_time': round(float(self.pts[i]), 2)}
        if score is not None:
            r['score'] = round(score, 4)
        if self.ocr_raw is not None and self.ocr_raw.iat[i]:
            r['ocr'] = self.ocr_raw.iat[i][:300]
        if self.cap is not None and self.cap['cap_of'][i] >= 0:
            r['tit'] = self.caption_of(i)[:300]
        r.update(kw)
        return r

    def search(self, text, k=100, max_per_video=None, ocr=None, use=None, ocr_window=3.0, cap=None,
               attrs=None):
        """Van ban (+ cac o mo ta attrs); ocr (tuy chon) loc keyframe co chu do trong +-ocr_window giay.
        Bo trong text va attrs -> chi OCR."""
        attr = self._attr_total(attrs, use)
        if (text and text.strip()) or attr[0] is not None:
            scores = self._combine(text, attr, use, cap)
        elif ocr and ocr.strip():
            # chi loc OCR: dung keyframe co chu (khong noi cua so), xep theo thu tu video/thoi gian
            scores, ocr_window = 1.0 - np.arange(len(self.meta)) / len(self.meta), 0.0
        else:
            raise ValueError('Can cau truy van hoac bo loc OCR')
        return self._rank(self._apply_ocr(scores, ocr, ocr_window), k, max_per_video)

    def similar(self, kf_id, k=100, max_per_video=None, ocr=None, use=None):
        return self._rank(self._apply_ocr(self.id_scores(kf_id, use), ocr, 3.0), k, max_per_video)

    def feedback(self, pos, neg=(), k=100, max_per_video=None, ocr=None, use=None, text=None,
                 alpha=1.0, beta=0.8, gamma=0.3):
        """Rocchio tren tung encoder: q' = a*q_text + b*mean(dung) - c*mean(sai), roi gop nhu thuong."""
        act = self._active(use)
        per = []
        for i in act:
            e = self.encoders[i]
            v = torch.zeros(e.feats.shape[1], dtype=torch.float32, device=e.feats.device)
            if text and text.strip():
                v += alpha * e.text_vec(text.strip()).to(v.device)
            if pos:
                v += beta * F.normalize(e.feats[list(pos)].float().mean(0), dim=-1)
            if neg:
                v -= gamma * F.normalize(e.feats[list(neg)].float().mean(0), dim=-1)
            per.append(e.vec_scores(v))
        return self._rank(self._apply_ocr(self._fuse(per, act), ocr, 3.0), k, max_per_video)

    def image_scores(self, imgs, use=None):
        """Diem moi keyframe voi 1 hoac nhieu anh ngoai, lay MAX tren cac anh (giu de collect_gt dung)."""
        if not isinstance(imgs, (list, tuple)):
            imgs = [imgs]
        return self.image_matrix(imgs, use).max(axis=1)

    def search_image(self, img, k=100, max_per_video=None, ocr=None, use=None):
        return self._rank(self._apply_ocr(self.image_scores(img, use), ocr, 3.0), k, max_per_video)

    def search_frames(self, imgs, k=100, max_per_video=None, ocr=None, use=None):
        """Nhieu khung cua cung 1 clip, diem = MAX tren cac khung (cach cu, collect_gt van dung)."""
        return self._rank(self._apply_ocr(self.image_scores(imgs, use), ocr, 3.0), k, max_per_video)

    def clip_scores(self, S, times, tol=3.0, n_cand=3000, generic_delta=0.02):
        """VKIS: chua lai diem MAX bang cach can khung clip theo thoi gian trong cung video.

        S [N, n]: diem keyframe x khung clip; times [n]: giay cua tung khung trong clip.
        Voi moi ung vien c (top n_cand theo MAX): khung clip khop nhat f* -> moc t0 = t_c - times[f*];
        khung f phai co keyframe khop trong [t0 + times[f] +- tol] -> diem = trung binh co trong so.
        Ban tin khac chi trung canh studio/MC o dau clip se khong khop cac khung sau -> tut hang.
        Trong so khung: khung 'chung' (nhieu video khac cung khop gan bang top1) nhe hon.
        """
        N, n = S.shape
        times = np.asarray(times, dtype=np.float64) - float(np.min(times))
        smax = S.max(axis=1)
        if n == 1:
            return smax
        # trong so: so video co diem >= top1 cua cot - delta (tinh tren top 2000 cua cot)
        w = np.ones(n)
        for f in range(n):
            col = S[:, f]
            # delta theo do lech chuan cua cot: 0.4*std ~ 0.02 cosine voi ViT-B-32, dung ca cho z-score gop
            delta = generic_delta / 0.05 * col.std()
            top = np.argpartition(-col, min(2000, N) - 1)[:min(2000, N)]
            near = top[col[top] >= col[top].max() - delta]
            w[f] = 1.0 / (1.0 + np.log(len(np.unique(self.video[near]))))
        cand = np.argpartition(-smax, min(n_cand, N) - 1)[:min(n_cand, N)]
        fstar = S[cand].argmax(axis=1)
        t0 = self.tkey[cand] - times[fstar]
        acc = np.zeros(len(cand))
        floor = S.min(axis=0)
        for f in range(n):
            lo = np.searchsorted(self.tkey, t0 + times[f] - tol, side='left')
            hi = np.searchsorted(self.tkey, t0 + times[f] + tol, side='right')
            best = np.full(len(cand), floor[f])
            for m in range(int((hi - lo).max()) if len(cand) else 0):
                j = lo + m
                ok = j < hi
                v = np.where(ok, S[np.minimum(j, N - 1), f], -np.inf)
                best = np.maximum(best, v)
            acc += w[f] * best
        out = np.full(N, -np.inf)
        # cac keyframe cung doan can ra cung diem -> cong chut diem MAX rieng de keyframe khop nhat len dau
        out[cand] = acc / w.sum() + 1e-3 * smax[cand]
        return out

    def search_clip(self, imgs, times, k=100, max_per_video=None, ocr=None, use=None, tol=3.0):
        """VKIS bang ca clip: can theo thoi gian (xem clip_scores)."""
        S = self.image_matrix(imgs, use)
        return self._rank(self._apply_ocr(self.clip_scores(S, times, tol), ocr, 3.0), k, max_per_video)

    def temporal(self, texts, window_s=30.0, k=100, n_cand=None, max_per_video=None, ocr=None, use=None,
                 rep='mid', cap=None, attrs=None):
        """Chuoi su kien theo thu tu (tkis thuong co 3 cau: mo dau / sau do / cuoi cung).

        score = tong diem tung su kien (DP, xem chain_scores); moi su kien sau trong window_s giay sau
        su kien truoc, cung video. Moi ket qua kem 'chain' (A, B, C), 'first' (A), 'next' (B).
        rep='mid' (mac dinh): keyframe tra ve (va nop) la su kien GIUA (chain_rep); 'first': A nhu cu.
        ocr: chu do phai xuat hien o dau do trong ca chuoi (+- window_s * so su kien).
        n_cand: giu tham so cu cho tuong thich, DP chay tren toan bo keyframe nen khong can.
        """
        texts = [t for t in (texts if isinstance(texts, (list, tuple)) else [texts]) if t and t.strip()]
        attr = self._attr_total(attrs, use)     # attrs: cac o mo ta ca clip, cong vao MOI su kien
        if not texts:
            if attr[0] is None:
                raise ValueError('Khong co cau truy van nao')
            return self._rank(self._apply_ocr(self._combine('', attr, use, cap), ocr, 3.0), k, max_per_video)
        scores = [self._combine(t, attr, use, cap) for t in texts]
        if len(scores) == 1:
            return self._rank(self._apply_ocr(scores[0], ocr, 3.0), k, max_per_video)
        combined, picks = chain_scores(scores, self.tkey, window_s)
        combined = self._apply_ocr(combined, ocr, window_s * len(scores))

        # xep theo diem chuoi (moi neo A 1 diem), roi doi sang keyframe dai dien; nhieu neo cung ra
        # 1 keyframe dai dien -> giu cai diem cao nhat
        out, seen, per_video = [], set(), {}
        for i in np.argsort(-combined, kind='stable'):
            if not np.isfinite(combined[i]) or len(out) >= k:
                break
            ev = [int(i)] + picks(i)
            r = chain_rep(ev, self.tkey) if rep == 'mid' else int(i)
            v = self.video[r]
            if r in seen or (max_per_video and per_video.get(v, 0) >= max_per_video):
                continue
            seen.add(r)
            per_video[v] = per_video.get(v, 0) + 1
            row = self.row(r, float(combined[i]))
            if len(ev) > 1:
                row['chain'] = [self.row(j, float(scores[n][j])) for n, j in enumerate(ev)]
                row['first'], row['next'] = row['chain'][0], row['chain'][1]
            out.append(row)
        return out

    def neighbors(self, kf_id, radius=6):
        """Cac keyframe lien truoc/sau trong cung video."""
        lo, hi = max(0, kf_id - radius), min(len(self.video), kf_id + radius + 1)
        return [self.row(i) for i in range(lo, hi) if self.video[i] == self.video[kf_id]]

    # ------------------------------------------------------------------ anh
    def image_bytes(self, kf_id):
        r = self.meta.iloc[kf_id]
        with open(os.path.join(self.data_dir, r['tar']), 'rb') as f:
            f.seek(int(r['offset']))
            return f.read(int(r['size']))


if __name__ == '__main__':
    # 2 video, moi video 6 keyframe cach nhau 10s -> tkey = ma_video*1e6 + giay
    tkey = np.array([0, 10, 20, 30, 40, 50, 1e6, 1e6 + 10, 1e6 + 20, 1e6 + 30, 1e6 + 40, 1e6 + 50])
    n = len(tkey)

    def arr(**kw):
        a = np.zeros(n)
        for i, v in kw.items():
            a[int(i)] = v
        return a

    # A@0, B@20, C@40 trong video 1 -> chuoi dung, cua so 25s du de nhay tung buoc
    A, B, C = arr(**{'0': 1.0, '6': 0.9}), arr(**{'2': 1.0}), arr(**{'4': 1.0})
    tot, picks = chain_scores([A, B, C], tkey, 25.0)
    assert picks(0) == [2, 4], picks(0)
    assert abs(tot[0] - 3.0) < 1e-9, tot[0]
    # ung vien 6 (video 2) khong co B/C -> chi con diem A, phai thua ung vien 0
    assert abs(tot[6] - 0.9) < 1e-9 and tot[0] > tot[6]

    # cua so qua hep -> khong bat duoc B
    tot2, picks2 = chain_scores([A, B, C], tkey, 5.0)
    assert picks2(0) == [] and abs(tot2[0] - 1.0) < 1e-9, (picks2(0), tot2[0])

    # khong duoc nhay sang video khac: A@50 (video 1), B chi co o video 2
    A2, B2 = arr(**{'5': 1.0}), arr(**{'6': 1.0})
    tot3, picks3 = chain_scores([A2, B2], tkey, 100.0)
    assert picks3(5) == [] and abs(tot3[5] - 1.0) < 1e-9

    # phai dung THU TU: C nam TRUOC B thi khong duoc tinh diem
    C_truoc = arr(**{'1': 1.0})                        # C dinh o t=10, truoc B@20
    tot4, picks4 = chain_scores([A, B, C_truoc], tkey, 25.0)
    assert picks4(0)[0] == 2 and picks4(0)[1] > 2, picks4(0)
    assert abs(tot4[0] - 2.0) < 1e-9, tot4[0]          # 1.0(A) + 1.0(B) + 0(C bi loai)

    # 1 su kien -> giong het tim thuong
    tot5, picks5 = chain_scores([A], tkey, 25.0)
    assert picks5(0) == [] and tot5[0] == 1.0 and tot5[6] == 0.9

    # DP hon tham lam: B gan (diem 0.6) dan toi C rat tot; B xa (0.7) khong co C sau no
    # tham lam chon B=0.7 roi mat C; DP chon B=0.6 + C=1.0
    A6, B6, C6 = arr(**{'0': 1.0}), arr(**{'1': 0.6, '3': 0.7}), arr(**{'2': 1.0})
    tot6, picks6 = chain_scores([A6, B6, C6], tkey, 30.0)
    assert picks6(0) == [1, 2] and abs(tot6[0] - 2.6) < 1e-9, (picks6(0), tot6[0])

    # DP khop vet can (brute force) tren du lieu ngau nhien
    rng = np.random.default_rng(0)
    tk = np.sort(np.concatenate([rng.uniform(0, 200, 60), 1e6 + rng.uniform(0, 200, 50)]))
    ev = [rng.normal(size=len(tk)) for _ in range(3)]
    tot7, _ = chain_scores(ev, tk, 15.0)
    for i in rng.choice(len(tk), 25, replace=False):
        best = -np.inf
        for j in range(i + 1, len(tk)):
            if tk[j] - tk[i] > 15:
                break
            for kk in range(j + 1, len(tk)):
                if tk[kk] - tk[j] > 15:
                    break
                best = max(best, ev[1][j] + ev[2][kk])
        if best > -np.inf:
            assert abs(tot7[i] - (ev[0][i] + best)) < 1e-9

    # range_max / window_max_center khop vet can, ke ca o -inf
    rng2 = np.random.default_rng(1)
    v = rng2.normal(size=300)
    v[rng2.choice(300, 60, replace=False)] = -np.inf
    lo = rng2.integers(0, 299, 300)
    hi = np.minimum(lo + rng2.integers(1, 70, 300), 300)
    assert np.allclose(range_max(v, lo, hi), [v[a:b].max() for a, b in zip(lo, hi)])
    wm = window_max_center(np.array([0, 5, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0.]), tkey, 10.0)
    assert list(wm[:6]) == [5, 5, 5, 0, 0, 0] and list(wm[6:8]) == [1, 1], wm   # khong lan sang video 2
    zf = zscore_finite(np.array([1.0, 3.0, -np.inf]))
    assert np.isinf(zf[2]) and abs(zf[0] + zf[1]) < 1e-9

    # keyframe dai dien: 3 su kien -> B; 2 su kien -> gan diem giua; 1 -> chinh no
    assert chain_rep([0, 2, 4], tkey) == 2 and chain_rep([3], tkey) == 3
    assert chain_rep([0, 4], tkey) == 2                  # t=0 va t=40 -> giua t=20 (chi so 2)
    assert chain_rep([0, 1], tkey) in (0, 1)

    # loc OCR: khong dau, cum trong ngoac, OR bang ';'
    texts = pd.Series(['GIẢl PHÓNG KHÍ HIĐRO', 'phi 1 0 bru', '', 'Di tích Kim Long'])   # 'l' = OCR doc nham 'I'
    tn = texts.map(norm_ocr)
    assert list(ocr_mask(tn, 'giai phong hidro')) == [True, False, False, False]
    assert list(ocr_mask(tn, '"kim long"; bru')) == [False, True, False, True]
    assert list(ocr_mask(tn, '"phong hidro"')) == [False, False, False, False]
    assert norm_vi('Đường Hồ Chí Minh!') == 'duong ho chi minh'
    m = dilate_mask(np.array([False, False, True, False, False, False] + [False] * 6), tkey, 10.0)
    assert list(m[:6]) == [False, True, True, True, False, False] and not m[6:].any()
    assert attr_parts({}) == [] and attr_parts({'who': ' ', 'neg': ''}) == []
    assert attr_parts({'act': 'Stacking wood.', 'where': 'Wood workshop', 'neg': 'graphics'}) == [
        ('stacking wood', 1.0), ('a photo taken in wood workshop', 0.7), ('graphics', -0.5)]
    print('engine.py selftest OK')
