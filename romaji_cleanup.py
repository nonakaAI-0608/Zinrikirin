"""Conservative removal of appended phonetic duplicates, not arbitrary English."""
import re
import unicodedata
from core import BASE, DIGRAPH, get_reading, hiragana

LATIN = re.compile(r"[A-Za-z]+(?:[ \t]+[A-Za-z]+)*")
JAPANESE_END = re.compile(r'[一-鿿々ぁ-ゖァ-ヶー \t]+$')


def canonical(text):
    text = unicodedata.normalize('NFKC', text).lower()
    for old, new in [('shi','si'),('chi','ti'),('tsu','tu'),('sha','sya'),('shu','syu'),('sho','syo'),
                     ('cha','tya'),('chu','tyu'),('cho','tyo'),('ja','zya'),('ju','zyu'),('jo','zyo'),('ji','zi'),('fu','hu')]:
        text = text.replace(old, new)
    return re.sub('[^a-z]', '', text)


def reading_suffixes(reading):
    """Use normal romaji (including gemination), never filename romanization."""
    text = hiragana(reading); tokens=[]; i=0; geminate=False
    while i<len(text):
        c=text[i]
        if c.isspace():i+=1;continue
        if c=='っ':geminate=True;i+=1;continue
        if c=='ー':
            if tokens and tokens[-1][-1] in 'aeiou':tokens.append(tokens[-1][-1])
            i+=1;continue
        pair=text[i:i+2]
        if pair in DIGRAPH:token=DIGRAPH[pair];i+=2
        elif c in BASE:
            token=BASE[c];i+=1
            if token.startswith('l'):token=token[1:]
        else:return set()
        if geminate:
            if token[0] not in 'aeioun':token=token[0]+token
            geminate=False
        tokens.append(token)
    return {canonical(''.join(tokens[i:])) for i in range(len(tokens))}


def remove_romaji_duplicates(text, overrides=None, hint=''):
    """Return (clean text, removed spans). Preserve uncertain cases and hint English.

    Only ASCII word suffixes immediately following a Japanese phrase are tested.
    A complete phonetic match of at least four letters is required; no fuzzy match.
    """
    protected=[canonical(m.group()) for m in LATIN.finditer(hint)]
    edits=[]
    for run in LATIN.finditer(text):
        before=text[:run.start()].rstrip(' \t、,：:')
        jp=JAPANESE_END.search(before)
        if not jp or not jp.group().strip():continue
        source=jp.group().strip()
        suffixes=reading_suffixes(get_reading(source,overrides))
        words=list(re.finditer('[A-Za-z]+',run.group()))
        for word in words:
            prefix=run.group()[:word.start()].strip()
            if prefix and not (canonical(prefix) in protected or
                               all(re.fullmatch(r'(?:wow)+|oh|yeah|hey|ooh|ah',w,re.I) for w in prefix.split())):
                continue
            candidate=run.group()[word.start():]
            key=canonical(candidate)
            if len(key)<4 or key not in suffixes:continue
            # If a spelling appears in the supplied lyrics, keep it even if phonetic.
            if any(key in phrase for phrase in protected):continue
            start=run.start()+word.start()
            while start>0 and text[start-1] in ' \t':start-=1
            edits.append((start,run.end(),candidate));break
    result=text
    for start,end,_ in reversed(edits):result=result[:start]+result[end:]
    return result,[removed for _,_,removed in edits]
