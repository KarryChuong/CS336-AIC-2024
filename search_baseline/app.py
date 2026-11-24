"""Web UI + API cho baseline truy van.

    python app.py --index D:/data/HCMAI24_keyframes/index --videos D:/4_Code/IR/batch1
    -> mo http://localhost:8000
    python app.py --index .../index_siglip2 .../index --weights 1 0.5 ...   # gop 2 encoder

--index: 1 hoac nhieu thu muc index (moi encoder 1 cai, cai DAU la chinh). --weights: trong so tung cai.
--ocr: thu muc chua <vid>.jsonl OCR (mac dinh tu tim <data>/ocr/).
--caption: index dong tit (build_caption_index.py) -> cong diem khop nghia; --caption-weight trong so.

--videos (tuy chon): thu muc chua .mp4 (tim de quy) de xem lai doan video quanh keyframe.
--gt: file jsonl luu cac cap (cau truy van, dap an) bam "Luu lam GT" tren UI -> dung cho evaluate.py.
"""
import argparse
import io
import json
import os
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from PIL import Image
from pydantic import BaseModel

import dres
from engine import SearchEngine, attr_parts
from video import sample_frames      # import truc tiep: route 'def video(...)' se che mat module

STATIC = Path(__file__).parent / 'static'


class GTItem(BaseModel):
    query: str
    query_b: str = ''
    query_c: str = ''
    video: str
    frame_idx: int
    pts_time: float
    note: str = ''


class FeedbackItem(BaseModel):
    pos: list[int] = []
    neg: list[int] = []
    text: str = ''
    ocr: str = ''
    enc: str = ''
    k: int = 100
    max_per_video: int = 0


class SubmitItem(BaseModel):
    kf_id: int
    text: str = ''      # khac rong -> nop dang VQA


def create_app(index_dir, data_dir=None, video_dir=None, gt_file='queries/gt_ui.jsonl', dres_client=None,
               weights=None, ocr_dirs=None, engine=None, caption_dir=None, caption_weight=0.5,
               sub_log='runs/submissions.jsonl'):
    eng = engine or SearchEngine(index_dir, data_dir, weights=weights, ocr_dirs=ocr_dirs,
                                 caption_dir=caption_dir, caption_weight=caption_weight)
    dres_eval_name = None
    if dres_client is not None:
        # kiem tra ngay luc khoi dong: sai DRES_EVAL / co 2 evaluation ma chua chon -> biet truoc khi nop
        try:
            eid, dres_eval_name = dres_client.active_evaluation()
            print(f'DRES evaluation: "{dres_eval_name}" ({eid})')
        except Exception as e:
            print(f'!! DRES: {e}')
    videos = {p.stem: p for p in Path(video_dir).rglob('*.mp4')} if video_dir else {}
    app = FastAPI(title='HCMAI24 search baseline')

    def _check(kf_id):
        if not 0 <= kf_id < len(eng.meta):
            raise HTTPException(404, f'id {kf_id} khong ton tai')

    def _timed(fn, **extra):
        """Chay 1 truy van, loi dau vao (ValueError) -> HTTP 400 thay vi 500."""
        t = time.time()
        try:
            res = fn()
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {'results': res, 'ms': round((time.time() - t) * 1000, 1), **extra}

    def _cap(v):
        """'' -> trong so mac dinh; '0' -> tat nhanh tit."""
        try:
            return float(v) if str(v).strip() != '' else None
        except ValueError:
            raise HTTPException(400, f'cap phai la so: {v!r}')

    @app.get('/')
    def index():
        return FileResponse(STATIC / 'index.html')

    @app.get('/api/info')
    def info():
        return {**eng.info, 'device': eng.device, 'n_videos_playable': len(videos), 'gt_file': gt_file,
                'dres': dres_client is not None, 'dres_eval': dres_eval_name}

    def _attrs(v):
        """attrs = JSON {"who": ..., "act": ..., "where": ..., "objs": ..., "color": ..., "neg": ...}."""
        if not v or not v.strip():
            return None
        try:
            d = json.loads(v)
        except ValueError:
            raise HTTPException(400, f'attrs khong phai JSON: {v[:100]!r}')
        if not isinstance(d, dict):
            raise HTTPException(400, 'attrs phai la object JSON')
        return d

    @app.get('/api/search')
    def search(q: str = '', k: int = 100, max_per_video: int = 0, ocr: str = '', enc: str = '', cap: str = '',
               attrs: str = ''):
        """q rong + ocr -> chi loc theo chu tren man hinh. attrs: cac o mo ta (tra lai 'parts' da gui model)."""
        at = _attrs(attrs)
        return _timed(lambda: eng.search(q, k, max_per_video or None, ocr=ocr, use=enc, cap=_cap(cap), attrs=at),
                      parts=attr_parts(at))

    @app.get('/api/temporal')
    def temporal(a: str = '', b: str = '', c: str = '', window: float = 30.0,
                 k: int = 100, max_per_video: int = 0, ocr: str = '', enc: str = '', cap: str = '',
                 attrs: str = ''):
        at = _attrs(attrs)
        return _timed(lambda: eng.temporal([a, b, c], window, k, max_per_video=max_per_video or None,
                                           ocr=ocr, use=enc, cap=_cap(cap), attrs=at), parts=attr_parts(at))

    @app.get('/api/video_frames')
    def video_frames(video: str):
        """Moi keyframe cua 1 video theo thoi gian (o 'Ma video' tren UI)."""
        return _timed(lambda: eng.video_frames(video))

    @app.post('/api/search_image')
    async def search_image(request: Request, k: int = 100, max_per_video: int = 0, frames_n: int = 0,
                           ocr: str = '', enc: str = '', align: int = 1, tol: float = 3.0):
        """Body la byte anh HOAC video tho.

        vkis: dan anh chup clip muc tieu, hoac tha thang file 00.mp4 ma capture.py luu ve.
        Video -> lay khung cach deu (frames_n=0: tu chon theo do dai clip). align=1 (mac dinh): can
        khung clip theo thoi gian trong cung video (engine.clip_scores); align=0: MAX tren khung nhu cu.
        """
        raw = await request.body()
        if not raw:
            raise HTTPException(400, 'Body rong, can byte anh hoac video')
        times = None
        try:
            img = Image.open(io.BytesIO(raw))
            img.load()
            frames = [img.convert('RGB')]
        except Exception as img_err:
            try:
                frames, times = sample_frames(raw, frames_n, return_times=True)   # vkis: ca clip de bai
            except Exception as vid_err:
                raise HTTPException(400, f'Khong doc duoc anh ({img_err}) cung khong phai video ({vid_err})')
        mpv = max_per_video or None
        if times and align and len(frames) > 1:
            return _timed(lambda: eng.search_clip(frames, times, k, mpv, ocr=ocr, use=enc, tol=tol),
                          frames=len(frames), mode='align')
        return _timed(lambda: eng.search_frames(frames, k, mpv, ocr=ocr, use=enc), frames=len(frames), mode='max')

    @app.get('/api/similar/{kf_id}')
    def similar(kf_id: int, k: int = 100, max_per_video: int = 0, ocr: str = '', enc: str = ''):
        _check(kf_id)
        return _timed(lambda: eng.similar(kf_id, k, max_per_video or None, ocr=ocr, use=enc))

    @app.post('/api/feedback')
    def feedback(item: FeedbackItem):
        """Rocchio: cac keyframe danh dau dung (pos) / sai (neg) + cau truy van hien tai."""
        for i in item.pos + item.neg:
            _check(i)
        if not item.pos and not (item.text or '').strip():
            raise HTTPException(400, 'Can it nhat 1 keyframe dung hoac cau truy van')
        return _timed(lambda: eng.feedback(item.pos, item.neg, item.k, item.max_per_video or None,
                                           ocr=item.ocr, use=item.enc, text=item.text))

    @app.get('/api/neighbors/{kf_id}')
    def neighbors(kf_id: int, r: int = 6):
        _check(kf_id)
        return {'results': eng.neighbors(kf_id, r)}

    @app.get('/img/{kf_id}')
    def img(kf_id: int):
        _check(kf_id)
        return Response(eng.image_bytes(kf_id), media_type='image/jpeg',
                        headers={'Cache-Control': 'public, max-age=86400'})

    @app.get('/video/{vid}')
    def video(vid: str):
        if vid not in videos:
            raise HTTPException(404, 'khong co video (chay app voi --videos)')
        return FileResponse(videos[vid], media_type='video/mp4')

    @app.post('/api/gt')
    def save_gt(item: GTItem):
        os.makedirs(os.path.dirname(gt_file) or '.', exist_ok=True)
        rec = {'id': f'ui_{int(time.time() * 1000)}', 'query': item.query, 'video': item.video,
               'start': item.frame_idx, 'end': item.frame_idx, 'pts_time': item.pts_time,
               'note': item.note, 'created': time.strftime('%Y-%m-%d %H:%M:%S')}
        for f in ('query_b', 'query_c'):
            if getattr(item, f):
                rec[f] = getattr(item, f)
        with open(gt_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        return {'ok': True, 'saved_to': os.path.abspath(gt_file)}

    # Server dung chung (--host 0.0.0.0): moi lan nop ghi 1 dong ra console + runs/submissions.jsonl
    # (ai nop, nop gi, DRES tra loi gi) -> loi nhu HTTP 500 xem lai duoc nguyen van.
    def _log_submit(rec):
        rec = {'time': time.strftime('%Y-%m-%d %H:%M:%S'), **rec}
        print('[NOP] ' + json.dumps(rec, ensure_ascii=False, default=str), flush=True)
        try:
            os.makedirs(os.path.dirname(sub_log) or '.', exist_ok=True)
            with open(sub_log, 'a', encoding='utf-8') as f:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + '\n')
        except OSError as e:
            print(f'!! khong ghi duoc {sub_log}: {e}', flush=True)

    # 2 nguoi bam Nop gan nhu cung luc -> 2 request song song toi DRES cung 1 tai khoan. Chi cho 1 luot
    # nop chay tai 1 thoi diem; luot den sau bi tu choi ngay (KHONG tu nop lai: DRES co the da ghi nhan).
    submit_lock = threading.Lock()

    @app.post('/api/submit')
    def submit_dres(item: SubmitItem, request: Request):
        if dres_client is None:
            raise HTTPException(400, 'Chua cau hinh DRES: chay app voi --dres-url/--dres-user/--dres-pass '
                                     'hoac dat bien moi truong DRES_URL/DRES_USER/DRES_PASS')
        _check(item.kf_id)
        row = eng.meta.iloc[item.kf_id]
        rec = {'from': request.client.host if request.client else '?', 'video': str(row['video']),
               'frame_idx': int(row['frame_idx']), 'text': item.text or None}
        if not submit_lock.acquire(blocking=False):
            _log_submit({**rec, 'result': 'BAN (dang co luot nop khac)'})
            raise HTTPException(409, 'Đang có một lượt nộp khác chưa xong. Đợi vài giây, xem kết quả rồi mới nộp lại.')
        try:
            res = dres_client.submit(row['video'], int(row['frame_idx']), float(row['fps']),
                                     text=item.text or None)
        except Exception as e:
            _log_submit({**rec, 'error': f'{type(e).__name__}: {e}'})
            raise HTTPException(502, f'{type(e).__name__}: {e}')
        finally:
            submit_lock.release()
        _log_submit({**rec, **res})
        return res

    return app


if __name__ == '__main__':
    import uvicorn

    p = argparse.ArgumentParser()
    p.add_argument('--index', required=True, nargs='+',
                   help='thu muc index (build_index.py); nhieu cai = nhieu encoder, cai dau la chinh')
    p.add_argument('--weights', type=float, nargs='*', default=None, help='trong so tung --index, vd 1 0.5')
    p.add_argument('--ocr', nargs='*', default=None, help='thu muc <vid>.jsonl OCR (mac dinh <data>/ocr/)')
    p.add_argument('--caption', default=None, help='thu muc index dong tit (build_caption_index.py)')
    p.add_argument('--caption-weight', type=float, default=0.5, help='trong so diem dong tit')
    p.add_argument('--data', default=None, help='OUT_DIR chua keyframes/*.tar (mac dinh lay tu info.json)')
    p.add_argument('--videos', default=None, help='thu muc .mp4 de xem lai video (tuy chon)')
    p.add_argument('--gt', default='queries/gt_ui.jsonl', help='file luu GT khi bam "Luu lam GT"')
    p.add_argument('--host', default='127.0.0.1', help='0.0.0.0 de may khac trong mang truy cap')
    p.add_argument('--port', type=int, default=8000)
    p.add_argument('--dres-url', default=None, help='vd https://dres.ban-to-chuc.vn (hoac bien DRES_URL)')
    p.add_argument('--dres-user', default=None, help='(hoac bien DRES_USER)')
    p.add_argument('--dres-pass', default=None, help='(hoac bien DRES_PASS) -- nen dung bien moi truong')
    p.add_argument('--dres-eval', default=None,
                   help='id / vai ky tu dau cua id / ten evaluation khi co nhieu cai cung chay (hoac DRES_EVAL)')
    a = p.parse_args()
    client = dres.from_env(a.dres_url, a.dres_user, a.dres_pass, a.dres_eval)
    print(f'DRES: {client.base if client else "chua cau hinh -> nut Nop bi tat"}')
    uvicorn.run(create_app(a.index, a.data, a.videos, a.gt, client, a.weights, a.ocr,
                           caption_dir=a.caption, caption_weight=a.caption_weight), host=a.host, port=a.port)
