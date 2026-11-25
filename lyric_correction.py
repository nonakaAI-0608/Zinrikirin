"""Conservative post-recognition lyric matching, without prompting ASR."""
import re
import unicodedata
from difflib import SequenceMatcher
from lyrics import normalize


def mapped_text(text):
    normalized=[];positions=[]
    for index,char in enumerate(text):
        for c in unicodedata.normalize('NFKC',char).casefold():
            if c.isalnum():normalized.append(c);positions.append(index)
    return ''.join(normalized),positions


def correct_from_lyrics(document,recognized,start,end,duration):
    query=normalize(recognized)
    if not document.lines or len(query)<6 or len(query)>180:return recognized,False
    if document.timed:
        indices,_=document.reference(start,end,duration,recognized)
        if not indices:return recognized,False
    else:indices=list(range(len(document.lines)))
    candidates={}
    for i in indices:
        texts=[document.lines[i].text]
        if i+1<len(document.lines):texts.append(document.lines[i].text+' '+document.lines[i+1].text)
        for text in texts:
            norm,positions=mapped_text(text)
            if not norm:continue
            match=SequenceMatcher(None,query,norm,autojunk=False).find_longest_match()
            if match.size<max(3,len(query)//3):continue
            anchor=match.b-match.a
            slack=min(5,max(1,len(query)//10))
            for left in range(max(0,anchor-2),min(len(norm),anchor+3)):
                for count in range(max(6,len(query)-slack),len(query)+slack+1):
                    right=left+count
                    if right>len(norm):continue
                    fragment=norm[left:right]
                    score=SequenceMatcher(None,query,fragment,autojunk=False).ratio()
                    # Both edges must match: do not expand a short clip to a whole lyric line.
                    if fragment[:2]!=query[:2] or fragment[-2:]!=query[-2:]:continue
                    if score<.90:continue
                    original=text[positions[left]:positions[right-1]+1]
                    candidates[original]=max(score,candidates.get(original,0))
    if not candidates:return recognized,False
    ranked=sorted(candidates.items(),key=lambda x:x[1],reverse=True)
    best,score=ranked[0]
    # Distinct competing readings cause abstention, rather than guessing from song position.
    alternatives=[s for text,s in ranked[1:] if normalize(text)!=normalize(best)]
    if alternatives and score-max(alternatives)<.035:return recognized,False
    if best==recognized:return recognized,False
    return best,True
