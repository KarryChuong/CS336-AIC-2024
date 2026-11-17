"""Nop ket qua len DRES (theo oas-client.json 2.0.0-RC4).

3 call HTTP:
    POST /api/v2/login                        {username, password}  -> sessionId
    GET  /api/v2/client/evaluation/list       ?session=             -> lay cai status == ACTIVE
    POST /api/v2/submit/{evaluationId}        ?session=             -> ket qua cham

Session cache lai; gap 401 thi login lai 1 lan.

Nhieu evaluation cung ACTIVE (vd "De thi thu" + "De thi"): chon bang DRES_EVAL (hoac --dres-eval)
= id day du, vai ky tu dau cua id (vd a64a8da4), hoac dung ten. Khong chon ma co >1 cai ACTIVE
-> bao loi, KHONG tu chon (nop nham evaluation la mat diem that).

Tu kiem tra (khong can mang):  python dres.py
"""
import os

import requests


def frame_to_ms(frame_idx, fps):
    """DRES nhan start/end bang milliseconds.

    fps phai la fps cua chinh video do, khong duoc hard-code 25: batch2 co 187 video
    o 30fps va 1 video o 26.44fps -> lech 20% tuong duong hang tram giay.
    """
    return round(frame_idx / fps * 1000)


def build_body(video, frame_idx, fps, text=None):
    """ApiClientSubmission. text != None -> nop dang VQA thay vi media item."""
    ms = frame_to_ms(frame_idx, fps)
    answer = {'text': text} if text else {'mediaItemName': video, 'start': ms, 'end': ms}
    # taskName/taskId la optional: server tu suy ra task dang chay. Neu bi 412 thi phai
    # goi /api/v2/client/evaluation/currentTask/{id} roi dien taskName vao day.
    return {'answerSets': [{'answers': [answer]}]}, ms


def select_evaluation(evs, want=None):
    """(id, name) cua evaluation can dung trong list tu /client/evaluation/list.

    want: id day du / tien to id / dung ten (khong phan biet hoa thuong). None -> chi hop le khi
    dung 1 cai ACTIVE. Chon theo want thi khong bat buoc ACTIVE (de bao loi ro hon o buoc nop).
    """
    def show(es):
        return '; '.join(f'{e.get("id")} "{e.get("name")}" {e.get("status")}' for e in es) or '(khong co)'

    if want:
        w = want.strip().lower()
        hit = [e for e in evs if str(e.get('id', '')).lower() == w]
        hit = hit or [e for e in evs if str(e.get('name', '')).strip().lower() == w]
        hit = hit or [e for e in evs if str(e.get('id', '')).lower().startswith(w)]
        if len(hit) != 1:
            why = 'khong thay' if not hit else f'trung {len(hit)} cai'
            raise RuntimeError(f'DRES_EVAL="{want}" {why}. Dang co: {show(evs)}')
        return hit[0]['id'], hit[0].get('name', hit[0]['id'])
    act = [e for e in evs if str(e.get('status')).upper() == 'ACTIVE']
    if not act:
        raise RuntimeError('Khong co evaluation nao dang ACTIVE')
    if len(act) > 1:
        raise RuntimeError(f'Co {len(act)} evaluation ACTIVE -> dat DRES_EVAL=<id hoac ten>. '
                           f'Dang co: {show(act)}')
    return act[0]['id'], act[0].get('name', act[0]['id'])


class Dres:
    def __init__(self, base_url, username, password, timeout=15, evaluation=None):
        self.base = base_url.rstrip('/')
        self.user, self.pwd, self.timeout = username, password, timeout
        self.evaluation = evaluation or None    # id / tien to id / ten; None = cai ACTIVE duy nhat
        self.session_id = None

    def _login(self):
        r = requests.post(f'{self.base}/api/v2/login',
                          json={'username': self.user, 'password': self.pwd}, timeout=self.timeout)
        if not r.ok:
            raise RuntimeError(f'Login that bai: HTTP {r.status_code} {r.text[:200]}')
        try:
            self.session_id = r.json().get('sessionId')
        except ValueError:  # thuong do DRES_URL tro vao trang web chu khong phai goc API
            raise RuntimeError(f'Login tra ve khong phai JSON ({r.url}): {r.text[:200]!r}')
        if not self.session_id:
            raise RuntimeError(f'Login khong tra ve sessionId: {r.text[:200]}')
        return self.session_id

    def _req(self, method, path, **kw):
        for attempt in (1, 2):
            kw.setdefault('params', {})['session'] = self.session_id or self._login()
            r = requests.request(method, f'{self.base}{path}', timeout=self.timeout, **kw)
            if r.status_code == 401 and attempt == 1:
                self.session_id = None      # het han -> login lai 1 lan
                continue
            return r

    def active_evaluation(self):
        r = self._req('GET', '/api/v2/client/evaluation/list')
        if not r.ok:
            raise RuntimeError(f'Lay evaluation list that bai: HTTP {r.status_code} {r.text[:200]}')
        return select_evaluation(r.json(), self.evaluation)

    def submit(self, video, frame_idx, fps, text=None):
        eid, name = self.active_evaluation()
        body, ms = build_body(video, frame_idx, fps, text)
        r = self._req('POST', f'/api/v2/submit/{eid}', json=body)
        try:
            data = r.json()
        except ValueError:
            data = {'description': r.text[:300]}
        return {'http': r.status_code, 'ok': r.ok, 'evaluation': name, 'ms': ms,
                'status': data.get('status'), 'submission': data.get('submission'),
                'description': data.get('description')}


def from_env(url=None, user=None, pwd=None, evaluation=None):
    """Tra ve Dres neu du 3 thong tin (uu tien tham so, sau do bien moi truong), nguoc lai None.
    evaluation / DRES_EVAL: tuy chon, chon evaluation khi co nhieu cai cung chay."""
    url = url or os.getenv('DRES_URL')
    user = user or os.getenv('DRES_USER')
    pwd = pwd or os.getenv('DRES_PASS')
    evaluation = evaluation or os.getenv('DRES_EVAL')
    return Dres(url, user, pwd, evaluation=evaluation) if (url and user and pwd) else None


if __name__ == '__main__':
    assert frame_to_ms(30000, 25.0) == 1200000
    assert frame_to_ms(30000, 30.0) == 1000000, 'video 30fps bi tinh theo 25fps -> lech 200s'
    assert frame_to_ms(933, 26.44) == 35287
    assert frame_to_ms(0, 25.0) == 0
    body, ms = build_body('L01_V003', 11400, 25.0)
    assert body == {'answerSets': [{'answers': [
        {'mediaItemName': 'L01_V003', 'start': 456000, 'end': 456000}]}]}, body
    assert ms == 456000
    body, _ = build_body('L01_V003', 11400, 25.0, text='QA-abc')
    assert body == {'answerSets': [{'answers': [{'text': 'QA-abc'}]}]}, body
    assert from_env(None, None, None) is None or os.getenv('DRES_URL')
    assert from_env('http://x:5000/', 'u', 'p').base == 'http://x:5000'

    # 2 evaluation cung chay nhu tren aiclub: phai chon dung, khong duoc lay cai dau tien
    evs = [{'id': 'bcb482c6-2b9e-4908-877a-e5ccd7c04640', 'name': 'Đề thi thử', 'status': 'ACTIVE'},
           {'id': 'a64a8da4-9b83-4431-965e-262667dfe0d1', 'name': 'Đề thi', 'status': 'ACTIVE'},
           {'id': 'ffff0000-old', 'name': 'Cu', 'status': 'ENDED'}]
    for w in ('a64a8da4', 'A64A8DA4-9b83-4431-965e-262667dfe0d1', 'Đề thi', ' đề thi '):
        assert select_evaluation(evs, w) == ('a64a8da4-9b83-4431-965e-262667dfe0d1', 'Đề thi'), w
    assert select_evaluation(evs, 'Đề thi thử')[0].startswith('bcb482c6')
    for w, msg in ((None, 'Co 2 evaluation ACTIVE'), ('zzz', 'khong thay'), ('', 'Co 2')):
        try:
            select_evaluation(evs, w)
            raise AssertionError(f'phai bao loi voi {w!r}')
        except RuntimeError as e:
            assert msg in str(e) and 'a64a8da4' in str(e), str(e)
    try:
        select_evaluation([{'id': 'ab1', 'name': 'x'}, {'id': 'ab2', 'name': 'y'}], 'ab')
        raise AssertionError('tien to trung phai bao loi')
    except RuntimeError as e:
        assert 'trung 2' in str(e)
    assert select_evaluation(evs[2:] + evs[:1]) == (evs[0]['id'], 'Đề thi thử')   # 1 ACTIVE -> tu chon
    assert from_env('http://x', 'u', 'p', 'a64a8da4').evaluation == 'a64a8da4'

    class FakeR:
        ok, status_code = True, 200
        def json(self):
            return evs

    class Fake(Dres):
        def _req(self, method, path, **kw):
            assert path == '/api/v2/client/evaluation/list'
            return FakeR()
    assert Fake('http://x', 'u', 'p', evaluation='a64a8da4').active_evaluation()[1] == 'Đề thi'
    print('dres.py selftest OK')
