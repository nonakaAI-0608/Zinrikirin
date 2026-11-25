"""Put the known lyrics into the intervals with Qwen3-ForcedAligner.

The text of each interval is cut out of the lyrics themselves, so there is no
mishearing; only the place of each cut has to be found.

* Timed LRC: every line is aligned inside its own short window, starting at
  its LRC time. Each word then goes to the interval that contains it.
* Plain lyrics: the aligner alone drifts on long sung passages (tested on a
  separated vocal: half the lines off by more than 0.5 s), so Qwen3-ASR first
  recognises every interval, the recognised kana are matched to the lyric kana
  to give each interval its rough share, and the aligner then moves every
  boundary between two neighbouring intervals using a short window.
"""
import re
import unicodedata

from lyrics import TIME, OFFSET, clean_text

ALIGNER_ID = 'Qwen/Qwen3-ForcedAligner-0.6B-hf'
NEAREST_LIMIT = 1.0      # a word this far outside every interval is dropped
LINE_LEAD = .4           # window starts this much before the LRC time
LINE_MAX = 15.0          # longest window for one line (instrumental gap after it)
PAIR_MAX = 30.0          # longest window for moving a boundary between two intervals
OPENERS = '「『（(【〈《［[“‘'


def lyric_lines(raw, shift=0.0):
    """[(time or None, text)] in singing order. LRC lines with several time tags
    (repeated choruses) appear once per tag, because they are sung each time."""
    offset = int(OFFSET.findall(raw)[-1]) / 1000 if OFFSET.search(raw) else 0.
    timed, plain = [], []
    for raw_line in raw.splitlines():
        tags = list(TIME.finditer(raw_line))
        text = clean_text(TIME.sub('', raw_line))
        if re.fullmatch(r'\[[^\]]*\]', text):
            text = ''
        if tags:
            for m in tags:
                hour, minute, sec, fraction = m.groups()
                t = int(hour or 0) * 3600 + int(minute) * 60 + int(sec) + float('0.' + (fraction or '0'))
                # LRC offset positive means earlier; UI shift positive means later.
                timed.append((t - offset + shift, text))
        elif text:
            plain.append((None, text))
    if timed:
        # Empty timed lines stay: they end the previous line (instrumental gap).
        return sorted(timed, key=lambda x: x[0])
    return [line for line in plain if _has_kept(line[1])]


def _kept(ch):
    """Characters the aligner keeps: letters, numbers and apostrophe."""
    return ch == "'" or unicodedata.category(ch)[0] in 'LN'


def _has_kept(text):
    return any(_kept(c) for c in text)


def build_source(lines):
    """Join lines into one text; returns (text, [(start, end) position of each line])."""
    parts, spans, pos = [], [], 0
    for _t, text in lines:
        spans.append((pos, pos + len(text)))
        parts.append(text)
        pos += len(text) + 1
    return '\n'.join(parts), spans


def match_tokens(text, tokens, base=0):
    """Positions of aligner tokens in text: [(start, end) or None] per token."""
    out, cursor = [], 0
    for token in tokens:
        span = _match(text, token, cursor) if token else None
        if span:
            cursor = span[1]
            out.append((span[0] + base, span[1] + base))
        else:
            out.append(None)
    return out


def _match(text, token, cursor):
    token = unicodedata.normalize('NFKC', token)
    start = cursor
    while start < len(text):
        while start < len(text) and not _kept(text[start]):
            start += 1
        j, k = start, 0
        while j < len(text) and k < len(token):
            a = unicodedata.normalize('NFKC', text[j])
            if a and token.startswith(a, k):
                j += 1
                k += len(a)
            elif not _kept(text[j]) and k:
                j += 1
            else:
                break
        if k >= len(token):
            return start, j
        start += 1
        if start - cursor > 40:  # tokens are found in order; do not jump far ahead
            return None
    return None


def assign_words(words, clips):
    """words: [(pos_start, pos_end, t0, t1)]. Each word goes to the interval that
    contains its middle, else the nearest one within NEAREST_LIMIT seconds.
    Returns {clip index: [word, ...]}."""
    out = {}
    for w in words:
        mid = (w[2] + w[3]) / 2
        inside = [i for i, c in enumerate(clips) if c.start <= mid < c.end]
        if inside:
            i = max(inside, key=lambda j: clips[j].start)
        else:
            near = [(min(abs(mid - c.start), abs(mid - c.end)), i) for i, c in enumerate(clips)]
            if not near:
                continue
            dist, i = min(near)
            if dist > NEAREST_LIMIT:
                continue
        out.setdefault(i, []).append(w)
    return out


def clip_text(source, words):
    """Original lyric text covered by the words, with line breaks as spaces,
    keeping punctuation that follows the last word (？！…）) and an opening bracket."""
    a = min(w[0] for w in words)
    b = max(w[1] for w in words)
    while a > 0 and source[a - 1] in OPENERS:
        a -= 1
    while b < len(source) and not _kept(source[b]) and not source[b].isspace():
        b += 1
    text = re.sub(r'\s*\n\s*', ' ', source[a:b]).strip()
    return re.sub(r' {2,}', ' ', text)


# --- plain lyrics: recognised kana -> rough share of the lyrics per interval ---

def kana_key(reading):
    """Comparable form of a kana reading: hiragana, no long-vowel marks, small vowels or symbols."""
    text = unicodedata.normalize('NFKC', reading).casefold()
    text = ''.join(chr(ord(c) - 0x60) if 'ァ' <= c <= 'ヶ' else c for c in text)
    return ''.join(c for c in text if _kept(c) and c not in 'ーぁぃぅぇぉ')


def edit_path(a, b):
    """For each position of a, the aligned position in b (global edit alignment)."""
    n, m = len(a), len(b)
    prev = list(range(m + 1))
    back = []
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        row = bytearray(m + 1)
        row[0] = 1
        ai = a[i - 1]
        for j in range(1, m + 1):
            d = prev[j - 1] + (ai != b[j - 1])
            u = prev[j] + 1
            left = cur[j - 1] + 1
            if d <= u and d <= left:
                cur[j] = d
            elif u <= left:
                cur[j] = u
                row[j] = 1
            else:
                cur[j] = left
                row[j] = 2
        back.append(row)
        prev = cur
    pos = [0] * n
    i, j = n, m
    while i > 0:
        move = back[i - 1][j] if j > 0 else 1
        if move == 0:
            i -= 1
            j -= 1
            pos[i] = j
        elif move == 1:
            i -= 1
            pos[i] = j
        else:
            j -= 1
    return pos


def partition_words(asr_keys, word_keys):
    """asr_keys: kana key per interval; word_keys: kana key per lyric word.
    Returns boundaries: len(intervals)+1 non-decreasing word indexes."""
    a = ''.join(asr_keys)
    b, owner, first_char = '', [], []
    for w, key in enumerate(word_keys):
        first_char.append(len(b))
        b += key
        owner += [w] * len(key)
    if not a or not b:
        return [0] * len(asr_keys) + [len(word_keys) if b else 0]
    pos = edit_path(a, b)

    def word_boundary(char_index):
        if char_index >= len(b):
            return len(word_keys)
        w = owner[char_index]
        return w if char_index - first_char[w] <= len(word_keys[w]) / 2 else w + 1

    bounds, total = [], 0
    for key in asr_keys:
        bounds.append(word_boundary(pos[total]) if key else None)  # empty: decided by neighbours
        total += len(key)
    bounds.append(word_boundary(pos[-1] + 1))
    for k in range(len(bounds) - 2, -1, -1):
        if bounds[k] is None:
            bounds[k] = bounds[k + 1]
    out, low = [], 0
    for v in bounds:
        low = max(low, v)
        out.append(low)
    return out


def best_split(mids, cut, prefer):
    """Index s so words [:s] go before time cut and [s:] after, with fewest words on
    the wrong side; ties keep the split nearest to prefer. None = word not placed."""
    best = None
    for s in range(len(mids) + 1):
        cost = sum(m is not None and m > cut for m in mids[:s]) + sum(m is not None and m < cut for m in mids[s:])
        key = (cost, abs(s - prefer))
        if best is None or key < best[0]:
            best = (key, s)
    return best[1]


# --- the two procedures ---

def align_timed(align, mono, lines, duration, rate=16000):
    """Align every LRC line in its own window. Returns (source, words) where
    words = [(pos_start, pos_end, t0, t1)] in seconds of the whole audio."""
    source, spans = build_source(lines)
    words = []
    for i, ((t, text), (a, b)) in enumerate(zip(lines, spans)):
        if not _has_kept(text):
            continue
        nxt = next((lines[j][0] for j in range(i + 1, len(lines)) if lines[j][0] > t), duration)
        start = max(0., t - LINE_LEAD)
        end = min(duration, nxt + LINE_LEAD, t + LINE_MAX)
        if end - start < .2:
            continue
        tokens, times = align(mono[int(start * rate):int(end * rate)], text)
        for p, (t0, t1) in zip(match_tokens(text, tokens, base=a), times):
            if p:
                words.append((p[0], p[1], t0 + start, t1 + start))
    return source, words


def align_plain(align, split_words, mono, lines, clips, transcripts, reading, rate=16000, notify=lambda i, n: None):
    """Plain lyrics. transcripts: recognised text per interval (all intervals, in time
    order). Returns (source, {clip index: [word, ...]})."""
    source, _spans = build_source(lines)
    words = [p for p in match_tokens(source, split_words(source)) if p]
    word_keys = [kana_key(reading(source[a:b])) for a, b in words]
    asr_keys = [kana_key(reading(t)) if t else '' for t in transcripts]
    bounds = partition_words(asr_keys, word_keys)
    for k in range(len(clips) - 1):
        notify(k + 1, len(clips) - 1)
        lo, hi = bounds[k], bounds[k + 2]
        left, right = clips[k], clips[k + 1]
        if hi <= lo or right.end - left.start > PAIR_MAX:
            continue
        seg_a, seg_b = words[lo][0], words[hi - 1][1]
        text = source[seg_a:seg_b]
        tokens, times = align(mono[int(left.start * rate):int(right.end * rate)], text)
        placed = {}
        for p, (t0, t1) in zip(match_tokens(text, tokens, base=seg_a), times):
            if p:
                placed.setdefault(p[0], (t0 + t1) / 2 + left.start)
        mids = [placed.get(words[w][0]) for w in range(lo, hi)]
        cut = (left.end + right.start) / 2
        bounds[k + 1] = lo + best_split(mids, cut, bounds[k + 1] - lo)
    groups = {}
    for k in range(len(clips)):
        if bounds[k + 1] > bounds[k]:
            groups[k] = [(a, b, 0., 0.) for a, b in words[bounds[k]:bounds[k + 1]]]
    return source, groups


CHUNK_MAX = 12  # a run of lyric text this short (no space or punctuation) is never split


def keep_chunks(source, groups):
    """Move every short phrase (text between spaces/punctuation) wholly into the
    interval that holds most of its characters, so cuts fall between phrases:
    「今日｜のデザート」 becomes 「｜今日のデザート」."""
    owner = sorted((w[0], w[1], k, w) for k, ws in groups.items() for w in ws)
    out = {}
    i = 0
    while i < len(owner):
        j = i + 1
        # Words belong to one chunk while only letters lie between them.
        while j < len(owner) and owner[j][0] >= owner[j - 1][1] and \
                all(_kept(c) for c in source[owner[j - 1][1]:owner[j][0]]):
            j += 1
        chunk = owner[i:j]
        size = sum(b - a for a, b, _k, _w in chunk)
        if size <= CHUNK_MAX and len({k for _a, _b, k, _w in chunk}) > 1:
            weight = {}
            for a, b, k, _w in chunk:
                weight[k] = weight.get(k, 0) + (b - a)
            # Ties: a phrase that starts a lyric line opens the later interval, otherwise
            # it closes the earlier one.
            before = source[:chunk[0][0]].rstrip(' 　（(「『')
            line_start = not before.strip() or before.endswith('\n')
            target = max(weight, key=lambda k: (weight[k], k if line_start else -k))
            chunk = [(a, b, target, w) for a, b, _k, w in chunk]
        for _a, _b, k, w in chunk:
            out.setdefault(k, []).append(w)
        i = j
    return out


def texts_for_clips(source, groups):
    """{clip index: text}; intervals without lyric words are left out."""
    return {k: clip_text(source, ws) for k, ws in keep_chunks(source, groups).items()}
