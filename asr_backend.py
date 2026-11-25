"""Lazy local ASR adapters. The recognisers never receive the full lyric text;
the forced aligner is the opposite: it only places the given lyrics on the audio."""
from pathlib import Path
import gc
import numpy as np

QWEN_LABEL='Qwen3-ASR-1.7B'
ALIGN_LABEL='Qwen3-ForcedAligner（歌詞を合わせる）'
MODELS=[QWEN_LABEL,ALIGN_LABEL,'Whisper tiny','Whisper base','Whisper small','Whisper medium','Whisper large-v3']
QWEN_ID='Qwen/Qwen3-ASR-1.7B-hf'

class Recognizer:
    def __init__(self,root):
        self.root=Path(root);self.key=None;self.model=None;self.processor=None

    def unload(self):
        self.model=None;self.processor=None;self.key=None
        gc.collect()
        import sys
        torch=sys.modules.get('torch')
        if torch is not None:
            try:
                if torch.cuda.is_available():torch.cuda.empty_cache()
            except (AttributeError,RuntimeError):pass

    def load(self,name,device='自動',notify=lambda text:None):
        key=(name,device)
        if self.key==key:return
        self.unload()
        if name in (QWEN_LABEL,ALIGN_LABEL):
            try:
                import torch
                from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration, Qwen3ASRForTokenClassification
                if name==ALIGN_LABEL:import nagisa  # Japanese word splitting for the aligner
            except (ImportError,AttributeError) as e:
                raise RuntimeError('Qwenの準備が必要です。アプリを閉じ、setup_qwen_gpu.bat（NVIDIA GPU）またはsetup_qwen_cpu.batを実行してください。\n'+str(e)) from e
            cuda=torch.cuda.is_available() and device!='CPU'
            if device=='GPU' and not cuda:raise RuntimeError('CUDA対応GPUが利用できません。setup_qwen_gpu.batとNVIDIAドライバーを確認するか、CPUを選択してください。')
            self.device='cuda:0' if cuda else 'cpu'
            # RTX 2070 SUPER does not support native BF16/FlashAttention 2.
            self.dtype=torch.float16 if cuda else torch.float32
            notify(('ForcedAligner' if name==ALIGN_LABEL else 'Qwen')+'を準備中：'+('GPU・FP16' if cuda else 'CPU・FP32（処理に時間がかかります）')+'／初回はモデルを取得します。')
            cache=str(self.root/'models')
            if name==ALIGN_LABEL:
                from lyric_aligner import ALIGNER_ID
                self.processor=AutoProcessor.from_pretrained(ALIGNER_ID,cache_dir=cache)
                self.model=Qwen3ASRForTokenClassification.from_pretrained(
                    ALIGNER_ID,cache_dir=cache,dtype=self.dtype,attn_implementation='sdpa',
                    low_cpu_mem_usage=True).to(self.device).eval()
                self.key=key;return
            self.processor=AutoProcessor.from_pretrained(QWEN_ID,cache_dir=cache)
            self.model=Qwen3ASRForConditionalGeneration.from_pretrained(
                QWEN_ID,cache_dir=cache,dtype=self.dtype,attn_implementation='sdpa',
                low_cpu_mem_usage=True).to(self.device).eval()
        else:
            from faster_whisper import WhisperModel
            name=name.removeprefix('Whisper ')
            if name not in ('tiny','base','small','medium','large-v3'):raise ValueError('未知のモデルです。')
            notify('Whisperを準備中：CPU・int8／初回はモデルを取得します。')
            self.model=WhisperModel(name,device='cpu',compute_type='int8',download_root=str(self.root/'models'))
        self.key=key

    def align(self,mono,text):
        """16 kHz mono + lyric text -> (tokens, [(start, end) seconds])."""
        import torch
        inputs,word_lists=self.processor.prepare_forced_aligner_inputs(
            audio=np.asarray(mono,dtype=np.float32),transcript=text,language='Japanese')
        if not word_lists[0]:return [],[]
        inputs=inputs.to(self.device,self.dtype)
        with torch.inference_mode():
            logits=self.model(**inputs).logits
        items=self.processor.decode_forced_alignment(
            logits=logits,input_ids=inputs['input_ids'],word_lists=word_lists,
            timestamp_token_id=self.model.config.timestamp_token_id)[0]
        del inputs,logits
        return [i['text'] for i in items],[(float(i['start_time']),float(i['end_time'])) for i in items]

    def transcribe(self,mono):
        mono=np.asarray(mono,dtype=np.float32)
        if not len(mono) or np.max(np.abs(mono))<1e-6:return ''
        if self.key[0]==ALIGN_LABEL:raise RuntimeError('ForcedAlignerは文字起こしをしません。')
        if self.key[0]!=QWEN_LABEL:
            segments,_=self.model.transcribe(mono,language='ja',beam_size=5,vad_filter=False,
                condition_on_previous_text=False,initial_prompt=None,temperature=0.)
            return ''.join(s.text for s in segments).strip()
        import torch
        # Split only for inference memory; exported clip boundaries do not change.
        # Long clips use low-energy cut points near 20 seconds, with no overlap.
        parts=[];start=0
        while start<len(mono):
            end=min(len(mono),start+20*16000)
            if end<len(mono):
                low=max(start+16000,end-2*16000)
                frames=mono[low:end].reshape(-1,160)
                end=low+(int(np.argmin(np.mean(frames**2,axis=1)))+1)*160
            chunk=mono[start:end];start=end
            # The HF processor only puts the language into the system prompt, which the model
            # treats as context; the reference implementation forces it by starting the answer
            # with "language Japanese<asr_text>". Without this, sung parts come back in English/Chinese.
            inputs=self.processor.apply_transcription_request(audio=chunk,return_tensors='pt')
            prefix=self.processor.tokenizer('language Japanese<asr_text>',add_special_tokens=False,return_tensors='pt')['input_ids']
            inputs['input_ids']=torch.cat([inputs['input_ids'],prefix],dim=1)
            inputs['attention_mask']=torch.cat([inputs['attention_mask'],torch.ones_like(prefix)],dim=1)
            inputs=inputs.to(self.device,self.dtype)
            with torch.inference_mode():
                output=self.model.generate(**inputs,max_new_tokens=512,do_sample=False)
            generated=output[:,inputs['input_ids'].shape[1]:]
            if generated.shape[1]>=512:
                raise RuntimeError('認識文が上限に達しました。この区間を短く分割して再認識してください。')
            decoded=self.processor.decode(generated,return_format='transcription_only')
            text=decoded[0] if isinstance(decoded,list) else decoded
            parts.append(text.strip())
            del inputs,output,generated
        return ''.join(parts).strip()
