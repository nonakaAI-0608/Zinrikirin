"""Audio segmentation and Japanese phonetic filenames. No GUI dependencies."""
from dataclasses import dataclass, asdict
from pathlib import Path
import math, re, unicodedata
import numpy as np
import soundfile as sf

@dataclass
class Clip:
    start: float
    end: float
    text: str = ''
    reading: str = ''
    name: str = ''
    excluded: bool = False
    raw_text: str = ""

@dataclass
class Settings:
    threshold: float = -30.0
    gap: float = .28
    before: float = .15
    after: float = .20


def runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def split_audio(audio, rate, settings):
    """RMS over all channels avoids cancellation of stereo voices."""
    if audio.ndim == 1:
        audio = audio[:, None]
    if not len(audio):
        return []
    hop = max(1, round(rate * .01))
    padded = np.pad(audio, ((0, (-len(audio)) % hop), (0, 0)))
    rms = np.sqrt(np.mean(padded.reshape(-1, hop, audio.shape[1]).astype(np.float64)**2, axis=(1, 2)))
    db = 20*np.log10(np.maximum(rms, 1e-12))
    quiet = db < settings.threshold
    active = ~quiet
    # Join interruptions shorter than the requested minimum silence.
    for s, e in runs(quiet):
        if s > 0 and e < len(quiet) and (e-s)*hop/rate < settings.gap:
            active[s:e] = True
    result = []
    for s, e in runs(active):
        if (e-s)*hop/rate < .05:
            continue
        result.append(Clip(max(0., s*hop/rate-settings.before),
                           min(len(audio)/rate, e*hop/rate+settings.after)))
    for left, right in zip(result, result[1:]):
        if left.end > right.start:
            left.end = right.start = (left.end + right.start) / 2
    return result


def hiragana(text):
    text = unicodedata.normalize('NFKC', text)
    return ''.join(chr(ord(c)-0x60) if 'ァ' <= c <= 'ヶ' else c for c in text)


def get_reading(text, overrides=None):
    from fugashi import Tagger
    import alkana
    import threading
    if not hasattr(get_reading, '_local'):
        get_reading._local = threading.local()
    if not hasattr(get_reading._local, 'tagger'):
        get_reading._local.tagger = Tagger()
    words = list(get_reading._local.tagger(unicodedata.normalize('NFKC', text)))
    dictionary = {unicodedata.normalize('NFKC', k).casefold(): hiragana(v)
                  for k, v in (overrides or {}).items()}
    parts = []; i = 0
    while i < len(words):
        # Match whole morphological tokens, longest registered phrase first.
        # This keeps 私 from replacing the first character of 私立.
        combined = ''; match = None
        for j in range(i, len(words)):
            if j > i and words[j].white_space: break
            combined += words[j].surface
            folded = combined.casefold()
            if folded in dictionary: match = (j + 1, dictionary[folded])
            if not any(k.startswith(folded) for k in dictionary): break
        if match:
            i, reading = match; parts.append(reading); continue
        w = words[i]; i += 1
        if re.fullmatch(r"[A-Za-z]+(?:['’-][A-Za-z]+)*", w.surface):
            parts.append(alkana.get_kana(w.surface.lower()) or w.surface)
        elif w.feature.pos1 == '助詞' and w.surface in ('は', 'へ', 'を'):
            parts.append({'は': 'わ', 'へ': 'え', 'を': 'お'}[w.surface])
        else:
            value = getattr(w.feature, 'kana', None)
            parts.append(value if value and value != '*' else w.surface)
    return hiragana(''.join(parts))


def resize_bounds(clips, index, edge, value, duration, rate):
    """Clamp the selected interval against all earlier/later rows, including exclusions."""
    low = max((c.end for c in clips[:index]), default=0.)
    high = min((c.start for c in clips[index+1:]), default=duration)
    minimum = 1 / rate
    if high - low < minimum:
        raise ValueError('隣接区間の間に空きがありません。開始・終了を確認してください。')
    clip = clips[index]
    start = max(low, min(clip.start, high-minimum))
    end = min(high, max(clip.end, start+minimum))
    value = round(value*rate)/rate
    if edge == 'start': start = max(low, min(value, end-minimum))
    else: end = min(high, max(value, start+minimum))
    return start, end

def resize_linked(clips, index, edge, value, duration, rate):
    """Move one edge and the facing edge of the neighbour together.

    clips are the positions at drag start. The neighbour follows when it was
    touching at drag start, or when the dragged edge crosses into it; it always
    keeps at least one sample. Returns {index: (start, end)} for changed rows.
    """
    minimum = 1 / rate
    touch = 1.5 / rate
    value = round(value * rate) / rate
    clip = clips[index]
    if edge == 'end':
        j = index + 1 if index + 1 < len(clips) else None
        high = clips[j].end - minimum if j is not None else duration
        end = min(high, max(clip.start + minimum, value))
        changes = {index: (clip.start, end)}
        if j is not None:
            other = clips[j]
            if abs(other.start - clip.end) <= touch or end > other.start:
                changes[j] = (end, other.end)
    else:
        j = index - 1 if index > 0 else None
        low = clips[j].start + minimum if j is not None else 0.
        start = max(low, min(clip.end - minimum, value))
        changes = {index: (start, clip.end)}
        if j is not None:
            other = clips[j]
            if abs(clip.start - other.end) <= touch or start < other.end:
                changes[j] = (other.start, start)
    return changes

BASE = {}
for kana, roma in [
    ('あいうえお', 'la li lu le lo'), ('かきくけこ', 'ka ki ku ke ko'),
    ('がぎぐげご', 'ga gi gu ge go'), ('さしすせそ', 'sa si su se so'),
    ('ざじずぜぞ', 'za zi zu ze zo'), ('たちつてと', 'ta ti tu te to'),
    ('だぢづでど', 'da zi zu de do'), ('なにぬねの', 'na ni nu ne no'),
    ('はひふへほ', 'ha hi hu he ho'), ('ばびぶべぼ', 'ba bi bu be bo'),
    ('ぱぴぷぺぽ', 'pa pi pu pe po'), ('まみむめも', 'ma mi mu me mo'),
    ('やゆよ', 'ya yu yo'), ('らりるれろ', 'ra ri ru re ro'),
    ('わゐゑをん', 'wa li le lo ln'), ('ぁぃぅぇぉ', 'la li lu le lo'),
    ('ゃゅょゔ', 'ya yu yo vu')]:
    BASE.update(zip(kana, roma.split()))
DIGRAPH = {}
for k, stem in [('き','ky'),('ぎ','gy'),('し','sy'),('じ','zy'),('ち','ty'),('ぢ','zy'),
                ('に','ny'),('ひ','hy'),('び','by'),('ぴ','py'),('み','my'),('り','ry')]:
    for small, vowel in zip('ゃゅょ', 'auo'):
        DIGRAPH[k+small] = stem+vowel
DIGRAPH.update({'てぃ':'ti','でぃ':'di','とぅ':'tu','どぅ':'du','ふぁ':'fa','ふぃ':'fi',
                'ふぇ':'fe','ふぉ':'fo','うぃ':'wi','うぇ':'we','うぉ':'wo',
                'しぇ':'sye','じぇ':'zye','ちぇ':'tye','ゔぁ':'va','ゔぃ':'vi','ゔぇ':'ve','ゔぉ':'vo'})

def romanize(reading):
    text = hiragana(reading).lower()
    out = []; i = 0
    while i < len(text):
        c = text[i]
        if c == 'っ':
            i += 1; continue
        if c == 'ー':
            # Sustained vowels and small tsu do not add filename characters.
            i += 1; continue
        pair = text[i:i+2]
        if pair in DIGRAPH:
            token = DIGRAPH[pair]; i += 2
        elif c in BASE:
            token = BASE[c]; i += 1
        elif c.isascii() and c.isdigit():
            token = c; i += 1
        elif c.isascii() and c.isalpha():
            raise ValueError('英単語・数字の読みを辞書に登録するか、かなで入力してください: '+reading)
        elif c.isspace() or unicodedata.category(c).startswith(('P','S')):
            i += 1; continue
        else:
            raise ValueError('読みをひらがな・カタカナで修正してください: '+c)
        out.append(token)
    if not out:
        raise ValueError('読みが空です。')
    return 'x'+''.join(out)+'x.wav'


def validate_clip(clip, duration):
    if not (math.isfinite(clip.start) and math.isfinite(clip.end) and 0 <= clip.start < clip.end <= duration+.000001):
        raise ValueError('開始・終了を音声の範囲内に設定してください。')


def export_clips(audio, rate, clips, folder, subtype='PCM_16', numbered=True):
    """Exclusive creation protects both existing exports and original files."""
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    created = []
    for i, clip in enumerate(clips, 1):
        if clip.excluded:
            continue
        validate_clip(clip, len(audio)/rate)
        name = clip.name.strip() or f'xclip{i:03d}x.wav'
        if not re.fullmatch(r'[A-Za-z0-9_-]+\.wav', name, re.I):
            raise ValueError(f'{i}行目: ファイル名には半角英数字・_・-を使用してください。')
    for i, clip in enumerate(clips, 1):
        if clip.excluded:
            continue
        name = clip.name.strip() or f'xclip{i:03d}x.wav'
        stem = (f'{i:02d}.' if numbered else '') + Path(name).stem
        j = 1
        while True:
            target = folder / (stem + (f'_{j}' if j > 1 else '') + '.wav')
            try:
                handle = target.open('xb'); break
            except FileExistsError:
                j += 1
        try:
            with handle:
                sf.write(handle, audio[round(clip.start*rate):round(clip.end*rate)], rate,
                         format='WAV', subtype=subtype if subtype in ('PCM_16','PCM_24','PCM_32','FLOAT','DOUBLE','PCM_U8') else 'PCM_24')
        except Exception:
            target.unlink(missing_ok=True)
            raise
        created.append(str(target))
    return created
