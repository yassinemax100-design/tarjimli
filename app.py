"""ترجملي — Tarjimli: ترجمة ودبلجة الفيديوهات إلى العربية (FastAPI + FFmpeg + OpenAI)."""
import json
import logging
import os
import shutil
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import openai
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

log = logging.getLogger("tarjimli")
logging.basicConfig(level=logging.INFO)

BASE = Path(__file__).parent
DATA = Path(os.getenv("DATA_DIR", "/tmp/tarjimli"))
UPLOADS, OUTPUTS = DATA / "uploads", DATA / "outputs"

MAX_MB = int(os.getenv("MAX_UPLOAD_MB", "100"))
MAX_SECONDS = int(os.getenv("MAX_DURATION_SECONDS", "600"))
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT_JOBS", "1"))
OUTPUT_TTL = int(os.getenv("OUTPUT_TTL_SECONDS", "3600"))
TRANSCRIBE_MODEL = os.getenv("OPENAI_TRANSCRIBE_MODEL", "whisper-1")  # يحتاج verbose_json
TRANSLATE_MODEL = os.getenv("OPENAI_TRANSLATE_MODEL", "gpt-4o-mini")
TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
SUB_FONT = os.getenv("SUBTITLE_FONT", "Noto Sans Arabic")

VOICES = {"alloy", "ash", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer"}
EXTS = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi", ".3gp"}

JOBS: dict = {}
LOCK = threading.Lock()
SLOTS = threading.Semaphore(MAX_CONCURRENT)


class JobError(Exception):
    """خطأ برسالة عربية جاهزة للعرض للمستخدم."""


def client() -> openai.OpenAI:
    return openai.OpenAI(timeout=180, max_retries=3)  # يقرأ OPENAI_API_KEY من البيئة


def api_key_ok() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


# ---------------------------------------------------------------- FFmpeg
def run(cmd, cwd=None):
    if cmd[0] == "ffmpeg":
        cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error"] + cmd[1:]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd,
                           stdin=subprocess.DEVNULL, timeout=1500)
    except subprocess.TimeoutExpired:
        raise JobError("استغرقت معالجة الفيديو وقتاً طويلاً جداً. جرّب فيديو أقصر.")
    if p.returncode != 0:
        log.error("فشل الأمر %s\n%s", cmd[0], p.stderr[-2000:])
        raise JobError("فشلت معالجة الفيديو. جرّب ملفاً بصيغة MP4.")
    return p.stdout


def probe_duration(path: Path) -> float:
    p = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True)
    try:
        return float(p.stdout.strip())
    except ValueError:
        return 0.0


def has_audio(path: Path) -> bool:
    p = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
         "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    return bool(p.stdout.strip())


def has_video(path: Path) -> bool:
    p = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "stream=index",
         "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    return bool(p.stdout.strip())


# ---------------------------------------------------------------- الترجمة المكتوبة
def ts(sec: float, sep: str) -> str:
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02}{sep}{ms:03}"


def write_subs(segs, srt: Path, vtt: Path):
    srt_lines, vtt_lines = [], ["WEBVTT", ""]
    for i, s in enumerate(segs, 1):
        srt_lines += [str(i), f"{ts(s['start'], ',')} --> {ts(s['end'], ',')}", s["ar"], ""]
        vtt_lines += [f"{ts(s['start'], '.')} --> {ts(s['end'], '.')}", s["ar"], ""]
    srt.write_text("\n".join(srt_lines), encoding="utf-8")
    vtt.write_text("\n".join(vtt_lines), encoding="utf-8")


# ---------------------------------------------------------------- OpenAI
def get(o, k):
    return o.get(k) if isinstance(o, dict) else getattr(o, k)


def transcribe(audio: Path):
    with open(audio, "rb") as f:
        r = client().audio.transcriptions.create(
            model=TRANSCRIBE_MODEL, file=f, response_format="verbose_json",
            timestamp_granularities=["segment"])
    segs = []
    for s in (getattr(r, "segments", None) or []):
        text = (get(s, "text") or "").strip()
        if text:
            segs.append({"start": float(get(s, "start")), "end": float(get(s, "end")), "text": text})
    if not segs:
        raise JobError("لم يتم العثور على كلام واضح في الفيديو.")
    # منع التداخل بين المقاطع
    for a, b in zip(segs, segs[1:]):
        a["end"] = min(a["end"], b["start"]) if b["start"] > a["start"] else a["end"]
    return segs


TRANSLATE_SYS = (
    "You are a professional subtitle and dubbing translator. Translate each segment into natural, "
    "simple Modern Standard Arabic. Be faithful but concise so the Arabic can be spoken in about the "
    "same time as the original. Use the neighbouring segments for context. Do not add notes. "
    'Return ONLY JSON: {"translations":[{"id":<int>,"text":"<arabic>"}]} with every id exactly once.')


def translate(segs):
    for i in range(0, len(segs), 40):
        chunk = segs[i:i + 40]
        payload = [{"id": i + k, "text": s["text"]} for k, s in enumerate(chunk)]
        resp = client().chat.completions.create(
            model=TRANSLATE_MODEL, temperature=0.2, response_format={"type": "json_object"},
            messages=[{"role": "system", "content": TRANSLATE_SYS},
                      {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}])
        try:
            data = json.loads(resp.choices[0].message.content)
            got = {int(x["id"]): str(x["text"]).strip() for x in data["translations"]}
        except (ValueError, KeyError, TypeError):
            raise JobError("تعذّر فهم ردّ الترجمة. أعد المحاولة.")
        for k, s in enumerate(chunk):
            if not got.get(i + k):
                raise JobError("أعادت خدمة الترجمة نتيجة ناقصة. أعد المحاولة.")
            s["ar"] = got[i + k]


def tts_one(text: str, dest: Path, voice: str):
    kw = {}
    if TTS_MODEL.startswith("gpt-4o"):
        kw["instructions"] = "تحدّث بالعربية الفصحى المبسطة بنبرة طبيعية وواضحة وإيقاع مناسب."
    with client().audio.speech.with_streaming_response.create(
            model=TTS_MODEL, voice=voice, input=text, response_format="mp3", **kw) as r:
        r.stream_to_file(dest)


def build_dub(src: Path, segs, work: Path, voice: str, volume: float, out: Path, progress):
    total = len(segs)
    done = [0]
    count_lock = threading.Lock()

    def make(idx_seg):
        idx, s = idx_seg
        raw = work / f"tts_{idx}.mp3"
        tts_one(s["ar"], raw, voice)
        dur = probe_duration(raw)
        nxt = segs[idx + 1]["start"] if idx + 1 < total else s["end"] + 2.0
        slot = max(s["end"] - s["start"], nxt - s["start"] - 0.05, 0.3)
        fit = raw
        if dur > slot * 1.02:  # تسريع الصوت ليلائم المدة المتاحة (حتى 1.6x)
            fit = work / f"fit_{idx}.mp3"
            run(["ffmpeg", "-y", "-i", str(raw), "-filter:a", f"atempo={min(dur / slot, 1.6):.3f}", str(fit)])
        with count_lock:
            done[0] += 1
            n = done[0]
        progress(n, total)
        return fit

    with ThreadPoolExecutor(max_workers=4) as ex:
        files = list(ex.map(make, enumerate(segs)))

    cmd = ["ffmpeg", "-y", "-i", str(src)]
    for f in files:
        cmd += ["-i", str(f)]
    parts, labels = [], []
    for k, s in enumerate(segs):
        ms = max(int(s["start"] * 1000), 0)
        parts.append(f"[{k + 1}:a]adelay={ms}|{ms}[d{k}]")
        labels.append(f"[d{k}]")
    parts.append("".join(labels) + f"amix=inputs={total}:normalize=0:dropout_transition=0[dub]")
    if volume > 0.01:
        parts.append(f"[0:a]volume={volume:.2f}[o]")
        parts.append("[o][dub]amix=inputs=2:duration=first:normalize=0[m]")
        parts.append("[m]apad[a]")
    else:
        parts.append("[dub]apad[a]")
    cmd += ["-filter_complex", ";".join(parts), "-map", "0:v:0", "-map", "[a]",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-shortest",
            "-movflags", "+faststart", str(out)]
    run(cmd)


def burn_subs(src: Path, srt: Path, work: Path, out: Path):
    shutil.copy(srt, work / "subs.srt")
    style = f"FontName={SUB_FONT},FontSize=16,Outline=1.2,Shadow=0,MarginV=24,Alignment=2"
    vf = f"scale=-2:'min(720,ih)',subtitles=subs.srt:force_style='{style}'"
    run(["ffmpeg", "-y", "-i", str(src), "-vf", vf, "-c:v", "libx264", "-preset", "veryfast",
         "-crf", "23", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
         "-movflags", "+faststart", str(out)], cwd=str(work))


# ---------------------------------------------------------------- المهمة
def update(jid, progress, message, status="running"):
    with LOCK:
        JOBS[jid].update(progress=progress, message=message, status=status)


def process(jid, src: Path, mode: str, volume: float, voice: str):
    work, out = DATA / f"work_{jid}", OUTPUTS / jid
    try:
        work.mkdir(parents=True, exist_ok=True)
        out.mkdir(parents=True, exist_ok=True)
        update(jid, 8, "في الانتظار...", "queued")
        with SLOTS:
            update(jid, 12, "جاري استخراج الكلام")
            audio = work / "audio.mp3"
            run(["ffmpeg", "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000", "-b:a", "48k", str(audio)])
            if audio.stat().st_size > 24 * 1024 * 1024:
                raise JobError("الصوت المستخرج كبير جداً. استخدم فيديو أقصر.")
            update(jid, 25, "جاري التعرّف على الكلام")
            segs = transcribe(audio)
            update(jid, 45, "جاري ترجمة النص")
            translate(segs)
            write_subs(segs, out / "tarjimli.srt", out / "tarjimli.vtt")
            video = out / "tarjimli.mp4"
            if mode == "dubbing":
                update(jid, 55, "جاري إنشاء الصوت العربي")
                build_dub(src, segs, work, voice, volume, video,
                          lambda d, t: update(jid, 55 + int(35 * d / t), "جاري إنشاء الصوت العربي"))
            else:
                update(jid, 70, "جاري دمج الترجمة مع الفيديو")
                burn_subs(src, out / "tarjimli.srt", work, video)
        files = {"video": "tarjimli.mp4", "srt": "tarjimli.srt", "vtt": "tarjimli.vtt"}
        with LOCK:
            JOBS[jid].update(status="done", progress=100, message="اكتملت الترجمة", files=files)
    except JobError as e:
        update(jid, JOBS[jid]["progress"], str(e), "error")
    except openai.AuthenticationError:
        update(jid, 0, "مفتاح OpenAI غير صالح. راجع إعداد OPENAI_API_KEY على الخادم.", "error")
    except openai.RateLimitError:
        update(jid, 0, "تجاوزت حدّ الاستخدام أو انتهى رصيد OpenAI. راجع حسابك ثم أعد المحاولة.", "error")
    except (openai.APIConnectionError, openai.APITimeoutError):
        update(jid, 0, "تعذّر الاتصال بخدمة OpenAI. حاول مرة أخرى بعد قليل.", "error")
    except openai.APIStatusError as e:
        log.error("OpenAI: %s", e)
        update(jid, 0, "رفضت خدمة OpenAI الطلب. قد يكون الملف غير مدعوم أو النموذج غير متاح لحسابك.", "error")
    except Exception:
        log.exception("خطأ غير متوقع")
        update(jid, 0, "حدث خطأ غير متوقع أثناء المعالجة. أعد المحاولة.", "error")
    finally:
        shutil.rmtree(work, ignore_errors=True)
        src.unlink(missing_ok=True)  # حذف الملف المرفوع بعد المعالجة
        with LOCK:
            if JOBS[jid]["status"] == "error":
                shutil.rmtree(out, ignore_errors=True)


def sweeper():
    while True:
        time.sleep(300)
        now = time.time()
        with LOCK:
            old = [j for j, v in JOBS.items() if now - v["created"] > OUTPUT_TTL and v["status"] in ("done", "error")]
            for j in old:
                JOBS.pop(j, None)
        for j in old:
            shutil.rmtree(OUTPUTS / j, ignore_errors=True)


# ---------------------------------------------------------------- التطبيق
app = FastAPI(title="Tarjimli", docs_url=None, redoc_url=None)


@app.on_event("startup")
def startup():
    for d in (UPLOADS, OUTPUTS):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)
    for d in DATA.glob("work_*"):
        shutil.rmtree(d, ignore_errors=True)
    threading.Thread(target=sweeper, daemon=True).start()
    if not api_key_ok():
        log.warning("OPENAI_API_KEY غير مضبوط — لن تعمل الترجمة.")


def err(msg: str, code: int):
    return JSONResponse({"error": msg}, status_code=code)


@app.get("/api/health")
def health():
    return {"ok": True, "api_key": api_key_ok(), "max_mb": MAX_MB, "max_minutes": MAX_SECONDS // 60}


@app.post("/api/jobs")
async def create_job(file: UploadFile = File(None), url: str = Form(""), mode: str = Form("subtitles"),
                     original_volume: float = Form(0.15), voice: str = Form("onyx")):
    if not api_key_ok():
        return err("الخادم غير متصل بخدمة OpenAI: لم يتم ضبط مفتاح OPENAI_API_KEY.", 503)
    if file is None or not file.filename:
        if url.strip():
            return err("تنزيل الفيديو من رابط إنستغرام مباشرةً غير مدعوم لأنه غير موثوق وكثيراً ما يُحظر. "
                       "نزّل الفيديو على هاتفك ثم ارفعه من زر «اختر الفيديو».", 400)
        return err("اختر ملف فيديو أولاً.", 400)
    if mode not in ("subtitles", "dubbing"):
        return err("وضع الإخراج غير صالح.", 400)
    voice = voice if voice in VOICES else "onyx"
    volume = min(max(original_volume, 0.0), 1.0)

    ext = Path(file.filename).suffix.lower()
    if ext not in EXTS and not (file.content_type or "").startswith("video/"):
        return err("نوع الملف غير مدعوم. ارفع فيديو بصيغة MP4 أو MOV أو WEBM.", 400)

    jid = uuid.uuid4().hex
    dest = UPLOADS / f"{jid}{ext if ext in EXTS else '.mp4'}"
    size, limit = 0, MAX_MB * 1024 * 1024
    with open(dest, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > limit:
                f.close()
                dest.unlink(missing_ok=True)
                return err(f"حجم الفيديو أكبر من الحد المسموح ({MAX_MB} ميغابايت).", 413)
            f.write(chunk)

    dur = probe_duration(dest)
    if dur <= 0 or not has_video(dest):
        dest.unlink(missing_ok=True)
        return err("الملف ليس فيديو صالحاً أو أنه تالف.", 400)
    if dur > MAX_SECONDS:
        dest.unlink(missing_ok=True)
        return err(f"الفيديو أطول من الحد المسموح ({MAX_SECONDS // 60} دقائق).", 400)
    if not has_audio(dest):
        dest.unlink(missing_ok=True)
        return err("لا يحتوي الفيديو على صوت لترجمته.", 400)

    with LOCK:
        JOBS[jid] = {"status": "queued", "progress": 5, "message": "جاري رفع الفيديو", "files": {},
                     "created": time.time()}
    threading.Thread(target=process, args=(jid, dest, mode, volume, voice), daemon=True).start()
    return {"id": jid}


@app.get("/api/jobs/{jid}")
def job_status(jid: str):
    with LOCK:
        j = JOBS.get(jid)
        if not j:
            return err("المهمة غير موجودة أو انتهت صلاحيتها.", 404)
        r = {"status": j["status"], "progress": j["progress"], "message": j["message"]}
        if j["status"] == "done":
            r["files"] = {k: f"/api/jobs/{jid}/file/{k}" for k in j["files"]}
        return r


@app.get("/api/jobs/{jid}/file/{kind}")
def job_file(jid: str, kind: str, dl: int = 0):
    with LOCK:
        j = JOBS.get(jid)
        name = j["files"].get(kind) if j and j["status"] == "done" else None
    if not name:
        return err("الملف غير متوفر.", 404)
    types = {"video": "video/mp4", "srt": "application/x-subrip", "vtt": "text/vtt"}
    return FileResponse(OUTPUTS / jid / name, media_type=types[kind], filename=name,
                        content_disposition_type="attachment" if dl else "inline")


app.mount("/", StaticFiles(directory=BASE / "static", html=True), name="static")
