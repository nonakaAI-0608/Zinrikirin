"""Full-song lyrics retrieval. Hints only; never replaces recognized text."""
from dataclasses import dataclass
from pathlib import Path
from difflib import SequenceMatcher
import re, unicodedata, math

TIME = re.compile(r'\[(?:(\d+):)?(\d+):([0-5]?\d)(?:[.:](\d{1,3}))?\]')
OFFSET = re.compile(r'\[offset\s*:\s*([+-]?\d+)\]', re.I)
META = re.compile(r'\[(?:ar|al|ti|au|by|re|tool|ve|length|la|offset|#)\s*:[^\]]*\]', re.I)
WORD_TIME = re.compile(r'<\d+(?::\d+){1,2}(?:\.\d+)?>')

@dataclass
class LyricLine:
    text: str
    time: float | None = None


def read_lyrics_file(path):
    raw=Path(path).read_bytes()
    if raw.startswith((b'\xff\xfe',b'\xfe\xff')):
        return raw.decode('utf-16')
    for encoding in ('utf-8-sig','cp932'):
        try:return raw.decode(encoding)
        except UnicodeDecodeError:pass
    raise ValueError('歌詞の文字コードを読めません。UTF-8形式で保存し直してください。')


def clean_text(text):
    return WORD_TIME.sub('',META.sub('',text)).strip()


def normalize(text):
    return ''.join(c for c in unicodedata.normalize('NFKC',text).casefold()
                   if c.isalnum())


class LyricsDocument:
    def __init__(self,raw,shift=0.0,normalizer=None):
        if not math.isfinite(shift):raise ValueError('LRC時刻調整には有限の秒数を入力してください。')
        self.raw=raw;self.shift=shift;self.normalizer=normalizer
        self.offset_ms=int(OFFSET.findall(raw)[-1]) if OFFSET.search(raw) else 0
        timed=[];plain=[]
        for raw_line in raw.splitlines():
            matches=list(TIME.finditer(raw_line))
            text=clean_text(TIME.sub('',raw_line))
            if matches:
                for m in matches:
                    hour,minute,sec,fraction=m.groups()
                    t=int(hour or 0)*3600+int(minute)*60+int(sec)+float('0.'+(fraction or '0'))
                    # LRC offset positive means earlier; UI shift positive means later.
                    timed.append(LyricLine(text,t-self.offset_ms/1000+shift))
            elif text and not re.fullmatch(r'\[[^\]]+\]',text):
                # Untimed lyrics need no manual segmentation; keep sensible search units.
                for part in re.split(r'[。！？!?]+',text):
                    for pos in range(0,len(part),80):
                        if part[pos:pos+80].strip():plain.append(LyricLine(part[pos:pos+80].strip()))
        self.timed=bool(timed)
        if timed:
            grouped={}
            for line in sorted(timed,key=lambda x:x.time):
                texts=grouped.setdefault(line.time,[])
                if line.text and line.text not in texts:texts.append(line.text)
            self.lines=[LyricLine(' / '.join(texts),time) for time,texts in grouped.items()]
        else:self.lines=plain
        self.normalized=None

    def prepare_matching(self):
        if self.normalized is not None:return
        def convert(text):
            if self.normalizer:
                try:text=self.normalizer(text)
                except Exception:pass
            return normalize(text)
        self.convert=convert
        self.normalized=[convert(line.text) for line in self.lines]

    def reference(self,start,end,duration,recognized=''):
        """Returns (candidate line indexes in priority order, selection description)."""
        if not self.lines:return [],'歌詞なし'
        center=None
        if self.timed:
            overlaps=[]
            for i,line in enumerate(self.lines):
                limit=self.lines[i+1].time if i+1<len(self.lines) else max(duration,end)
                overlap=max(0,min(end,limit)-max(start,line.time))
                if overlap>0:overlaps.append((overlap,i))
            if not overlaps:return [],'LRC時刻の範囲外'
            # Empty timed lines mark instrumental gaps; do not borrow unrelated lyrics.
            _,center=max(overlaps,key=lambda pair:(pair[0],-pair[1]))
            if not self.lines[center].text:return [],'LRC空行・間奏'
            priority=[center]+[i for _,i in sorted(overlaps,reverse=True) if i!=center and self.lines[i].text]
            mode='LRC時刻'
        else:
            fraction=max(0,min(1,((start+end)/2)/max(duration,.001)))
            weights=[max(1,len(line.text)) for line in self.lines]
            target=fraction*sum(weights);total=0;approx=len(weights)-1
            for i,weight in enumerate(weights):
                total+=weight
                if total>=target:approx=i;break
            center=approx;mode='曲内位置の概算'
            if recognized:
                self.prepare_matching();query=self.convert(recognized)
                if len(query)>=3:
                    scores=[]
                    for i,text in enumerate(self.normalized):
                        choices=[text]
                        if i+1<len(self.normalized):choices.append(text+self.normalized[i+1])
                        score=0.
                        for candidate in choices:
                            if not candidate:continue
                            match=SequenceMatcher(None,query,candidate,autojunk=False)
                            partial=match.find_longest_match().size/len(query)
                            score=max(score,match.ratio(),partial if query in candidate else partial*.85)
                        scores.append(score)
                    best=max(scores,default=0)
                    if best>=.45:
                        contenders=[i for i,score in enumerate(scores) if score>=best-.025]
                        center=min(contenders,key=lambda i:abs(i-approx));mode='認識文との照合'
            priority=[center]
        for distance in (1,2):
            for i in (center-distance,center+distance):
                if 0<=i<len(self.lines) and self.lines[i].text and i not in priority:priority.append(i)
        return priority,mode

    def prompt(self,priority,tokenizer,limit=200):
        """Budget the actual prompt tokens; full source text stays intact."""
        def encode(text):return tokenizer.encode(text,add_special_tokens=False).ids
        chosen=[];last=''
        for i in priority:
            proposed=sorted(chosen+[i])
            text='\n'.join(self.lines[j].text for j in proposed)
            if len(encode(text))<=limit:
                chosen=proposed;last=text
            elif not chosen:
                tokens=encode(self.lines[i].text)[:limit]
                last=tokenizer.decode(tokens)
                # Avoid any round-trip normalization growing beyond the token budget.
                while tokens and len(encode(last))>limit:
                    tokens=tokens[:-1];last=tokenizer.decode(tokens)
                return last or None
        return last or None
