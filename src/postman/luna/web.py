"""Loopback operator interface and deadman lease for Luna teleoperation."""

from __future__ import annotations

import json
import secrets
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from .control import HEAD_OFFSET, POSITION_OFFSET, ROTATION_OFFSET
from .dds import vector
from .model import BASE_RATE

DEADMAN_SECONDS = 0.24
COMMAND_SCALE = np.r_[np.tile(np.r_[POSITION_OFFSET, [ROTATION_OFFSET] * 3], 2),
                      BASE_RATE, [HEAD_OFFSET] * 2]


@dataclass(frozen=True)
class Jog:
    values: np.ndarray
    expires: float
    reanchor: int


class Operator:
    def __init__(self, port: int):
        self._lock = threading.Lock()
        self._jog = Jog(np.zeros(len(COMMAND_SCALE)), 0., 0)
        self._status = {"status": "Waiting for native DDS feedback"}
        self._token = secrets.token_urlsafe(24)
        self._sequence = 0
        self.server = ThreadingHTTPServer(("127.0.0.1", port), self._handler())
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def update(self, body: dict, token: str) -> bool:
        values = vector(body["values"], len(COMMAND_SCALE), "operator input")
        if np.any(np.abs(values) > 1.) or type(body["enable"]) is not bool:
            raise ValueError("invalid operator input")
        sequence = body["sequence"]
        if type(sequence) is not int or sequence <= 0:
            raise ValueError("invalid operator request sequence")
        with self._lock:
            if not secrets.compare_digest(token, self._token):
                return False
            if sequence <= self._sequence:
                return True
            self._sequence = sequence
            enable = body["enable"]
            now = time.monotonic()
            if enable and 0. < self._jog.expires <= now:
                # A stalled client must release the deadman before it can take control again.
                return True
            reanchor = self._jog.reanchor + int(body.get("reanchor") is True and not enable)
            self._jog = Jog(values * COMMAND_SCALE, now + DEADMAN_SECONDS if enable else 0.,
                            reanchor)
            return True

    def jog(self) -> Jog:
        with self._lock:
            return self._jog

    def status(self, **fields) -> None:
        with self._lock:
            self._status = fields

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def _handler(self):
        operator = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def send(self, status: int, content: bytes, kind: str):
                try:
                    self.send_response(status)
                    self.send_header("Content-Type", kind)
                    self.send_header("Content-Length", str(len(content)))
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("X-Frame-Options", "DENY")
                    self.end_headers()
                    self.wfile.write(content)
                except (BrokenPipeError, ConnectionResetError):
                    # The browser or SSH forward can close during a status poll.
                    self.close_connection = True

            def local_host(self):
                host = self.headers.get("Host", "").partition(":")[0]
                return host in {"localhost", "127.0.0.1"}

            def do_GET(self):
                if not self.local_host():
                    self.send_error(403)
                elif self.path == "/":
                    with operator._lock:
                        operator._sequence += 1
                        sequence = operator._sequence
                        last = operator._jog
                        operator._jog = Jog(last.values, 0., last.reanchor)
                        operator._token = secrets.token_urlsafe(24)
                        token = operator._token
                    page = PAGE.replace("__TOKEN__", json.dumps(token))
                    page = page.replace("__SEQUENCE__", str(sequence))
                    self.send(200, page.encode(), "text/html; charset=utf-8")
                elif self.path == "/status":
                    with operator._lock:
                        content = json.dumps(operator._status, allow_nan=False).encode()
                    self.send(200, content, "application/json")
                else:
                    self.send_error(404)

            def do_POST(self):
                if self.path != "/jog" or not self.local_host():
                    self.send_error(404)
                    return
                self.connection.settimeout(0.5)
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < length <= 2048:
                        raise ValueError("invalid input length")
                    body = json.loads(self.rfile.read(length))
                    if not operator.update(body, self.headers.get("X-Postman-Session", "")):
                        self.send_error(403)
                        return
                except (OSError, KeyError, TypeError, ValueError):
                    self.send_error(400)
                    return
                self.send(204, b"", "text/plain")

        return Handler


PAGE = """<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport"
content="width=device-width,initial-scale=1"><title>POSTMAN · Luna 680</title>
<style>
:root{color-scheme:dark;font-family:system-ui,sans-serif;background:#10151c;color:#e8edf3}
body{margin:2rem auto;max-width:960px;padding:0 1rem}h1{font-size:1.8rem}p{line-height:1.5}
main{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:1rem}
fieldset{border:1px solid #42536a;border-radius:8px;padding:.75rem}legend{padding:0 .4rem}
label{display:grid;grid-template-columns:3rem 1fr 3.7rem;align-items:center;
gap:.3rem;margin:.4rem 0}
output{font-variant-numeric:tabular-nums;text-align:right}button{padding:.65rem 1rem;margin:.4rem;
border:1px solid #557091;border-radius:6px;background:#202e40;color:inherit;font-size:1rem}
#hold{background:#214d55;touch-action:none}#hold.active{background:#637f38}pre{white-space:pre-wrap}
</style>
<h1>POSTMAN · Luna 680</h1>
<p id="status" role="status">等待 DDS 反馈</p>
<main id="controls"></main>
<p><button id="hold">按住运行（空格）</button><button id="reset">以当前姿态为零点</button></p>
<pre id="feedback"></pre>
<script>
const token=__TOKEN__;
const controls=document.querySelector('#controls'), hold=document.querySelector('#hold');
const labels=[['左手 · 米 / 弧度','X','Y','Z','Roll','Pitch','Yaw'],
 ['右手 · 米 / 弧度','X','Y','Z','Roll','Pitch','Yaw'],
 ['底盘 · 米/秒 / 弧度/秒','前后','左右','转向'],['头部 · 弧度','轴 1','轴 2']];
const limits=[.15,.12,.12,.30,.30,.30,.15,.12,.12,.30,.30,.30,.22,.18,.35,.25,.25];
const sliders=[];
for(const [title,...fields] of labels){
 const panel=document.createElement('fieldset'), legend=document.createElement('legend');
 legend.textContent=title;panel.append(legend);controls.append(panel);
 for(const name of fields){
  const label=document.createElement('label'), text=document.createElement('span');
  const slider=document.createElement('input'), value=document.createElement('output');
  const scale=limits[sliders.length];text.textContent=name;slider.type='range';
  slider.min=-1;slider.max=1;slider.step=.01;slider.value=0;
  const render=()=>value.textContent=(Number(slider.value)*scale).toFixed(3);
  slider.addEventListener('input',render);render();label.append(text,slider,value);
  sliders.push(slider);panel.append(label);
 }
}
let active=false,busy=false,sequence=__SEQUENCE__;
function send(reanchor=false){
 const body={values:sliders.map(x=>Number(x.value)),enable:active,reanchor,sequence:++sequence};
 return fetch('/jog',{method:'POST',headers:{'Content-Type':'application/json',
  'X-Postman-Session':token},body:JSON.stringify(body),keepalive:!active}).then(r=>{
   if(!r.ok) throw Error(`HTTP ${r.status}`);
 });
}
function enabled(next){
 if(next===active) return;
 active=next;hold.classList.toggle('active',next);
 send().catch(()=>{active=false;hold.classList.remove('active');});
}
hold.onpointerdown=e=>{hold.setPointerCapture(e.pointerId);enabled(true);};
hold.onpointerup=hold.onpointercancel=hold.onlostpointercapture=()=>enabled(false);
window.addEventListener('keydown',e=>{if(e.code==='Space'&&!e.repeat){e.preventDefault();enabled(true);}});
window.addEventListener('keyup',e=>{if(e.code==='Space'){e.preventDefault();enabled(false);}});
window.addEventListener('blur',()=>enabled(false));
document.addEventListener('visibilitychange',()=>{if(document.hidden)enabled(false);});
document.querySelector('#reset').onclick=()=>{
 enabled(false);sliders.forEach(x=>{x.value=0;x.dispatchEvent(new Event('input'));});
 send(true).catch(()=>{});
};
setInterval(()=>{
 if(active&&!busy){busy=true;send().catch(()=>enabled(false)).finally(()=>{busy=false;});}
},45);
setInterval(async()=>{
 try{
  const json=await (await fetch('/status')).json();
  document.querySelector('#status').textContent=json.status;
  delete json.status;document.querySelector('#feedback').textContent=JSON.stringify(json,null,2);
 }catch{enabled(false);document.querySelector('#status').textContent='连接中断';}
},160);
</script></html>"""
