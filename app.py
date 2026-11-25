import sys as _sys
if _sys.stdout is None or _sys.stderr is None:
    # Started with pythonw (no console): send print/progress output to a log file
    # instead of crashing on writes to a missing stdout/stderr.
    try:
        _console_log = open(__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), 'console.log'), 'w', encoding='utf-8', buffering=1)
    except OSError:
        _console_log = open(__import__('os').devnull, 'w')
    _sys.stdout = _sys.stdout or _console_log
    _sys.stderr = _sys.stderr or _console_log
import os
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
import json, queue, threading, traceback, math
from pathlib import Path
from dataclasses import asdict
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinterdnd2 import TkinterDnD, DND_FILES, COPY, REFUSE_DROP
import numpy as np
import soundfile as sf
from core import Clip, Settings, split_audio, get_reading, romanize, validate_clip, export_clips, resize_bounds, resize_linked

from romaji_cleanup import remove_romaji_duplicates
from asr_backend import Recognizer, MODELS, QWEN_LABEL, ALIGN_LABEL
from breaths import BreathSettings, detect_breaths, split_at_candidates
from lyric_correction import correct_from_lyrics
from history import History
from lyrics import LyricsDocument, read_lyrics_file

class App(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()
        self.title('ボーカル自動切り分け v0.2.0'); self.geometry('1200x880'); self.minsize(1050,750)
        self.audio=None; self.rate=0; self.path=None; self.subtype='PCM_16'; self.clips=[]
        self.events=queue.Queue(); self.busy=False; self.cursor=0.; self.recognizer=Recognizer(Path(__file__).parent)
        self.history=History(); self.zoom=1.; self.view_start=0.
        self._wave_drag=None; self.measurement=None
        self._levels=None; self._levels_key=None; self.hover_time=None
        self.dictionary_path=Path(__file__).with_name('reading_dictionary.json')
        self.reading_dictionary={'私':'わたし','明日':'あした'}
        self.dictionary_warning=''
        if self.dictionary_path.exists():
            try:
                loaded=json.loads(self.dictionary_path.read_text(encoding='utf-8'))
                if not isinstance(loaded,dict) or not all(isinstance(k,str) and isinstance(v,str) for k,v in loaded.items()):
                    raise ValueError('辞書の形式が不正です。')
                self.reading_dictionary=loaded
            except Exception as e:self.dictionary_warning=str(e)
        self.params={k:tk.StringVar(value=str(v)) for k,v in asdict(Settings()).items()}
        self.status=tk.StringVar(value='WAVを開くか、このウィンドウへWAVファイルをドロップしてください。')
        self.model=tk.StringVar(value=QWEN_LABEL); self.device=tk.StringVar(value='自動'); self.numbered=tk.BooleanVar(value=True)
        self.edit={k:tk.StringVar() for k in ('start','end','text','reading','name')}
        self._loading_editor=False; self._editor_id=None; self._inline=None
        self._space_down=False; self._space_release_job=None
        for key in ('text','reading','name'):
            self.edit[key].trace_add('write',lambda *args,k=key:self.auto_edit(k))
        self.bind_class('PlaybackShortcut','<space>',self.space_play)
        self.bind_class('PlaybackShortcut','<Shift-space>',self.space_play)
        self.bind_class('PlaybackShortcut','<KeyPress>',self.fullwidth_space_play)
        self.bind_class('PlaybackShortcut','<KeyRelease>',self.release_space)
        for sequence,callback in [('<Control-z>',self.undo),('<Control-Z>',self.undo),
                                  ('<Control-y>',self.redo),('<Control-Y>',self.redo)]:
            self.bind_class('PlaybackShortcut',sequence,callback)
        self.bind_class('PlaybackShortcut','<F2>',self.edit_lyrics_key)
        self.bind_class('PlaybackShortcut','<KeyPress-s>',self.split_key)
        self.bind_class('PlaybackShortcut','<KeyPress-S>',self.split_key)
        top=ttk.Frame(self,padding=10);top.pack(fill='x')
        for label,command in [('WAVを開く',self.open_audio),('プロジェクトを開く',self.load_project),('プロジェクト保存',self.save_project),('戻る Ctrl+Z',self.undo),('進む Ctrl+Y',self.redo)]:
            ttk.Button(top,text=label,command=command).pack(side='left',padx=3)
        self.filelabel=ttk.Label(top,text='未選択');self.filelabel.pack(side='left',padx=12)
        box=ttk.LabelFrame(self,text='自動分割（無音／ブレス候補）',padding=8);box.pack(fill='x',padx=10)
        for key,label in [('threshold','無音しきい値 dBFS'),('gap','最小の途切れ 秒'),('before','前の余白 秒'),('after','後の余白 秒')]:
            ttk.Label(box,text=label).pack(side='left',padx=4)
            ttk.Entry(box,textvariable=self.params[key],width=7).pack(side='left')
        ttk.Button(box,text='自動分割',command=self.segment).pack(side='left',padx=10)
        breath_row=ttk.Frame(self,padding=(10,4));breath_row.pack(fill='x')
        self.use_breath=tk.BooleanVar(value=True)
        self.breath_params={k:tk.StringVar(value=str(v)) for k,v in asdict(BreathSettings()).items()}
        ttk.Checkbutton(breath_row,text='ブレス候補も使う（適用前に確認）',variable=self.use_breath).pack(side='left')
        self.breath_position=tk.StringVar(value='ブレスの開始')
        ttk.Label(breath_row,text='分割位置').pack(side='left',padx=(10,3))
        ttk.Combobox(breath_row,textvariable=self.breath_position,values=['ブレスの開始','ブレスの中央'],state='readonly',width=11).pack(side='left')
        for key,label in [('sensitivity','感度 0～100'),('minimum','最短ブレス 秒'),('spacing','最短区間 秒')]:
            ttk.Label(breath_row,text=label).pack(side='left',padx=(10,3))
            ttk.Entry(breath_row,textvariable=self.breath_params[key],width=6).pack(side='left')
        row=ttk.Frame(self,padding=8);row.pack(fill='x')
        ttk.Label(row,text='認識モデル').pack(side='left')
        ttk.Combobox(row,textvariable=self.model,values=MODELS,state='readonly',width=34).pack(side='left',padx=5)
        ttk.Button(row,text='全区間の歌詞・名前をAI生成',command=self.recognize).pack(side='left',padx=5)
        ttk.Button(row,text='選択行のみ再生成',command=lambda:self.recognize(selected_only=True)).pack(side='left',padx=5)
        ttk.Button(row,text='読み辞書・一括変換',command=self.open_dictionary).pack(side='left',padx=5)
        ttk.Label(row,text='Qwen実行').pack(side='left',padx=(8,2))
        ttk.Combobox(row,textvariable=self.device,values=['自動','GPU','CPU'],state='readonly',width=5).pack(side='left')
        hint_panel=ttk.Frame(self,padding=(10,0));hint_panel.pack(fill='x')
        self._lyrics_expanded=False
        self.lyrics_shift=tk.StringVar(value="0")
        self.hint_button=ttk.Button(hint_panel,command=self.toggle_lyrics_hint)
        self.hint_button.pack(anchor='w')
        self.hint_body=ttk.Frame(hint_panel,padding=(0,4))
        hint_tools=ttk.Frame(self.hint_body);hint_tools.pack(fill='x')
        ttk.Button(hint_tools,text='LRC／歌詞TXTを開く',command=self.open_lyrics_file).pack(side='left')
        ttk.Label(hint_tools,text='LRC時刻調整（秒）').pack(side='left',padx=(12,3))
        ttk.Entry(hint_tools,textvariable=self.lyrics_shift,width=8).pack(side='left')
        ttk.Label(hint_tools,text='＋で遅らせる／－で早める').pack(side='left',padx=5)
        ttk.Button(hint_tools,text='選択行の参照候補',command=self.preview_lyrics).pack(side='right')
        ttk.Label(self.hint_body,text='フル歌詞またはLRCを入力できます。歌詞は認識後の照合に使い、認識モデルへは渡しません。').pack(anchor='w')
        text_frame=ttk.Frame(self.hint_body);text_frame.pack(fill='x')
        self.lyrics_text=tk.Text(text_frame,height=6,wrap='word',undo=True,maxundo=100)
        self.lyrics_text.pack(side='left',fill='x',expand=True)
        hint_scroll=ttk.Scrollbar(text_frame,orient='vertical',command=self.lyrics_text.yview)
        hint_scroll.pack(side='right',fill='y');self.lyrics_text.configure(yscrollcommand=hint_scroll.set)
        self.lyrics_text.bind('<<Modified>>',self.lyrics_hint_changed)
        hint_footer=ttk.Frame(self.hint_body);hint_footer.pack(fill='x')
        ttk.Label(hint_footer,text='高い一致が得られた部分だけ補正します。短い区間・曖昧な候補は認識結果を保持します。').pack(side='left')
        ttk.Button(hint_footer,text='クリア',command=self.clear_lyrics_hint).pack(side='right')
        self.update_lyrics_header()
        cleanup_row=ttk.Frame(self);cleanup_row.pack(fill='x',padx=10)
        self.post_correct=tk.BooleanVar(value=True)
        ttk.Checkbutton(cleanup_row,text='認識後に歌詞と照合・補正',variable=self.post_correct).pack(side='left')
        ttk.Button(cleanup_row,text='選択行の認識原文',command=self.show_raw_text).pack(side='left',padx=5)
        self.cleanup_romaji=tk.BooleanVar(value=True)
        ttk.Checkbutton(cleanup_row,text='AI認識後に重複ローマ字を除去',variable=self.cleanup_romaji).pack(side='left')
        ttk.Button(cleanup_row,text='選択行の重複ローマ字を除去',command=self.clean_selected_romaji).pack(side='left',padx=8)
        self.canvas=tk.Canvas(self,height=135,bg='#152131',highlightthickness=0);self.canvas.pack(fill='x',padx=10,pady=5)
        self.canvas.bind('<Configure>',lambda e:self.draw());self.canvas.bind('<Button-1>',self.wave_press)
        self.canvas.bind('<B1-Motion>',self.wave_drag)
        self.canvas.bind('<ButtonRelease-1>',self.wave_release)
        self.canvas.bind('<MouseWheel>',self.wave_wheel)
        self.canvas.bind('<Button-4>',self.wave_wheel);self.canvas.bind('<Button-5>',self.wave_wheel)
        self.wave_scroll=ttk.Scrollbar(self,orient='horizontal',command=self.scroll_wave)
        self.wave_scroll.pack(fill='x',padx=10)
        self.measure_label=ttk.Label(self,text='Shift＋ドラッグ：時間を測る／選択区間の端をドラッグ：長さを変更／Ctrl＋クリック：複数選択／F2：歌詞編集／S：分割／Shift+S：結合')
        self.measure_label.pack(anchor='w',padx=12)
        level_row=ttk.Frame(self);level_row.pack(fill='x',padx=12)
        self.show_level=tk.BooleanVar(value=True)
        ttk.Checkbutton(level_row,text='音量(dBFS)を表示',variable=self.show_level,command=self.draw).pack(side='left')
        self.link_edges=tk.BooleanVar(value=True)
        ttk.Checkbutton(level_row,text='端のドラッグで隣の区間も連動',variable=self.link_edges).pack(side='left',padx=(8,0))
        self.level_label=ttk.Label(level_row,text='波形にマウスを乗せると、その位置の音量（10ms RMS／無音しきい値と同じ計算）を表示します。')
        self.level_label.pack(side='left',padx=8)
        self.canvas.bind('<Motion>',self.wave_hover);self.canvas.bind('<Leave>',self.wave_leave)
        self.params['threshold'].trace_add('write',lambda *args:self.after_idle(self.draw))
        ttk.Button(self,text='波形を全体表示',command=self.reset_zoom).pack(anchor='e',padx=10)
        ttk.Label(self,text='Ctrl＋ホイール：拡大縮小／スペース：再生・停止／Shift＋スペース：バーから再生・停止／WAVドロップ対応').pack(anchor='w',padx=12)
        cols=('start','end','text','reading','name')
        frame=ttk.Frame(self);frame.pack(fill='both',expand=True,padx=10,pady=6)
        self.tree=ttk.Treeview(frame,columns=cols,show='tree headings',selectmode='extended',height=9)
        for k,label,width in zip(cols,['開始 秒','終了 秒','認識した歌詞','読み（編集可）','保存名（編集可）'],[85,85,220,220,300]):
            self.tree.heading(k,text=label);self.tree.column(k,width=width,stretch=k not in ('start','end'))
        self.tree.heading('#0',text='No.');self.tree.column('#0',width=48,stretch=False)
        self.tree.tag_configure('excluded',foreground='#8b8b8b',background='#e6e6e6')
        self.tree.pack(side='left',fill='both',expand=True)
        scroll=ttk.Scrollbar(frame,orient='vertical',command=self.tree.yview);scroll.pack(side='right',fill='y');self.tree.configure(yscrollcommand=scroll.set)
        self.tree.bind('<<TreeviewSelect>>',self.select)
        self.tree.bind('<ButtonRelease-1>',self.row_clicked)
        self.tree.bind('<Double-1>',self.inline_edit)
        editor=ttk.Frame(self,padding=8);editor.pack(fill='x')
        for i,(k,label) in enumerate(zip(cols,['開始','終了','歌詞','読み','保存名'])):
            ttk.Label(editor,text=label).grid(row=0,column=i,sticky='w')
            field=ttk.Entry(editor,textvariable=self.edit[k],width=10 if i<2 else 25)
            field.grid(row=1,column=i,sticky='ew',padx=3)
            field.bind('<Return>',lambda e,key=k:self.finish_edit(e,apply_times=key in ('start','end')))
            field.bind('<KP_Enter>',lambda e,key=k:self.finish_edit(e,apply_times=key in ('start','end')))
            editor.columnconfigure(i,weight=1 if i>=2 else 0)
        actions=ttk.Frame(self,padding=6);actions.pack(fill='x')
        for label,command in [('歌詞→読み',self.reading_from_text),('読み→保存名',self.name_from_reading),('編集を反映',self.apply_edit),('選択区間を再生',self.play),('停止',self.stop),('ここで分割 (S)',self.split_here),('選択行を結合 (Shift+S)',self.merge),('除外／解除',self.delete)]:
            ttk.Button(actions,text=label,command=command).pack(side='left',padx=2)
        bottom=ttk.Frame(self,padding=8);bottom.pack(fill='x')
        ttk.Checkbutton(bottom,text='01. の連番を付ける',variable=self.numbered).pack(side='left')
        ttk.Button(bottom,text='除外行を除いてWAV保存',command=self.export).pack(side='right')
        ttk.Label(self,textvariable=self.status,wraplength=1100,padding=10).pack(fill='x')
        # Run the playback shortcut before native button/tree class key bindings.
        def add_shortcut(widget):
            widget.bindtags(('PlaybackShortcut',)+widget.bindtags())
            for child in widget.winfo_children():add_shortcut(child)
        add_shortcut(self)
        for target in (self,self.canvas,self.tree,self.filelabel):
            target.drop_target_register(DND_FILES)
            target.dnd_bind('<<Drop>>',self.drop_audio)
        self.after(100,self.poll);self.protocol('WM_DELETE_WINDOW',self.close)

    def clean_selected_romaji(self):
        if not self.ready():return
        ids=self.selected()
        if not ids:self.status.set('処理する行を選択してください。');return
        try:
            updates=[]
            for i in ids:
                c=Clip(**asdict(self.clips[i]))
                text,removed=remove_romaji_duplicates(c.text,self.reading_dictionary,self.lyrics_hint())
                if not removed:continue
                c.text=text;c.reading=get_reading(text,self.reading_dictionary)
                try:c.name=romanize(c.reading)
                except ValueError:c.name=''
                updates.append((i,c))
            if updates:
                self.apply_recognition(updates)
                self.status.set(f'{len(updates)}行の重複を除去し、読み・保存名を更新しました。Ctrl+Zで戻せます。')
            else:self.status.set('読みが一致する重複ローマ字は見つかりませんでした。')
        except Exception as e:self.error(e)

    def split_key(self,event):
        if event.widget.winfo_class() in ('Entry','TEntry','Text','TCombobox','Spinbox','TSpinbox'):return
        # Control (0x4) or Alt (0x20000 on Windows): leave the key alone. NumLock/CapsLock are ignored.
        if event.state & 0x4 or event.state & 0x20000:return
        if event.state & 0x1:self.merge()
        else:self.split_here()
        return 'break'

    def edit_lyrics_key(self,event):
        if self.busy:return 'break'
        ids=self.selected()
        if len(ids)==1:return self.begin_cell_edit(str(ids[0]),'text')
        self.status.set('歌詞を編集する行を1つ選んでください。')
        return 'break'

    def open_dictionary(self):
        if not self.ready():return
        win=tk.Toplevel(self);win.title('読み辞書・一括変換');win.geometry('600x460')
        win.transient(self);win.grab_set()
        ttk.Label(win,text='生成前に登録すると、歌詞から読み・保存名を作る際に優先します。\n英単語も登録可能（大文字・小文字は区別しません）。').pack(padx=10,pady=10)
        listing=ttk.Treeview(win,columns=('word','reading'),show='headings',height=10)
        listing.heading('word',text='単語・フレーズ');listing.heading('reading',text='読み（かな）')
        listing.pack(fill='both',expand=True,padx=10)
        word=tk.StringVar();reading=tk.StringVar()
        row=ttk.Frame(win);row.pack(fill='x',padx=10,pady=8)
        ttk.Entry(row,textvariable=word,width=24).pack(side='left',padx=4)
        ttk.Entry(row,textvariable=reading,width=24).pack(side='left',padx=4)
        def refresh():
            listing.delete(*listing.get_children())
            for k,v in self.reading_dictionary.items():listing.insert('','end',values=(k,v))
        def choose(e):
            ids=listing.selection()
            if ids:
                k,v=listing.item(ids[0],'values');word.set(k);reading.set(v)
        listing.bind('<<TreeviewSelect>>',choose)
        def persist():
            try:
                temporary=self.dictionary_path.with_suffix('.tmp')
                temporary.write_text(json.dumps(self.reading_dictionary,ensure_ascii=False,indent=2),encoding='utf-8')
                temporary.replace(self.dictionary_path)
            except Exception as e:self.error('辞書を保存できませんでした。プロジェクト保存で保管できます。\n'+str(e))
            refresh()
        def register():
            import re,unicodedata
            k=unicodedata.normalize('NFKC',word.get()).strip()
            v=unicodedata.normalize('NFKC',reading.get()).strip()
            if not k or not re.fullmatch('[ぁ-ゖァ-ヶー]+',v):
                messagebox.showerror('読み辞書','単語と、かなの読みを入力してください。',parent=win);return
            for old in list(self.reading_dictionary):
                if old.casefold()==k.casefold():del self.reading_dictionary[old]
            self.reading_dictionary[k]=v;persist()
        def remove():
            for item in listing.selection():self.reading_dictionary.pop(listing.item(item,'values')[0],None)
            persist()
        ttk.Button(row,text='登録・更新',command=register).pack(side='left')
        buttons=ttk.Frame(win);buttons.pack(fill='x',padx=10,pady=5)
        ttk.Button(buttons,text='登録を削除',command=remove).pack(side='left')
        ttk.Button(buttons,text='選択行に一括適用',command=lambda:self.apply_dictionary(True)).pack(side='left',padx=5)
        ttk.Button(buttons,text='全行に一括適用',command=lambda:self.apply_dictionary(False)).pack(side='left')
        ttk.Label(win,text='一括適用は歌詞を保持し、読み・保存名を再生成します（Ctrl+Zで戻せます）。\n未対応の英単語は読み欄に残し、保存名を空欄にします。辞書に登録してください。').pack(padx=10,pady=8)
        refresh()
        if self.dictionary_warning:
            messagebox.showwarning('辞書の読み込み',self.dictionary_warning,parent=win);self.dictionary_warning=''

    def apply_dictionary(self,selected_only):
        if not self.ready():return
        ids=self.selected() if selected_only else list(range(len(self.clips)))
        if not ids:self.status.set('対象の行がありません。');return
        if not messagebox.askyesno('読み・保存名の一括変換',f'{len(ids)}行の読み・保存名を再生成します。手修正した内容も置き換えますか？'):return
        try:
            updates=[];unknown=0
            for i in ids:
                clip=Clip(**asdict(self.clips[i]));clip.reading=get_reading(clip.text,self.reading_dictionary)
                try:clip.name=romanize(clip.reading)
                except ValueError:clip.name='';unknown+=1
                updates.append((i,clip))
            self.apply_recognition(updates)
            self.status.set(f'{len(ids)}行の読み・保存名を更新しました。未設定の保存名：{unknown}行。')
        except Exception as e:self.error(e)

    def wave_time(self,x):
        duration,span=self.wave_extent()
        return max(0.,min(duration,self.view_start+x/max(1,self.canvas.winfo_width())*span))

    def wave_press(self,event):
        if self.audio is None or self.busy:return
        self.canvas.focus_set();self._wave_drag=None
        ids=self.selected()
        if event.state & 0x4:
            self.measurement=None;self.select_on_wave(event,extend=bool(event.state & 0x1));return
        if not event.state & 1 and len(ids)==1 and 10<=event.y<=109:
            i=ids[0];clip=self.clips[i];_,span=self.wave_extent()
            scale=max(1,self.canvas.winfo_width())/span
            edge=min(('start','end'),key=lambda k:abs((getattr(clip,k)-self.view_start)*scale-event.x))
            if abs((getattr(clip,edge)-self.view_start)*scale-event.x)<=7:
                self.stop()
                self._wave_drag={'mode':'edge','index':i,'edge':edge,'before':self.snapshot(),
                                 'orig':[Clip(**asdict(c)) for c in self.clips]}
                return
        if event.state & 1:
            self._wave_drag={'mode':'measure','start':self.wave_time(event.x)}
            self.measurement=(self.wave_time(event.x),)*2
            self.wave_drag(event)
        else:
            self.measurement=None;self.measure_label.configure(text='Shift＋ドラッグ：時間を測る／選択区間の端をドラッグ：長さを変更／Ctrl＋クリック：複数選択／F2：歌詞編集／S：分割／Shift+S：結合');self.click_wave(event)

    def wave_drag(self,event):
        drag=self._wave_drag
        if not drag:return
        value=self.wave_time(event.x)
        if drag['mode']=='measure':
            self.measurement=(drag['start'],value);a,b=sorted(self.measurement)
            self.measure_label.configure(text=f'選択範囲 {a:.4f} ～ {b:.4f} 秒 ｜ 長さ {b-a:.4f} 秒（{(b-a)*1000:.1f} ms）'+self.level_summary(a,b))
        else:
            i=drag['index'];clip=self.clips[i];duration=len(self.audio)/self.rate
            if self.link_edges.get():
                changes=resize_linked(drag['orig'],i,drag['edge'],value,duration,self.rate)
                # Rows not in this update go back to their drag-start position (e.g. after dragging back).
                for j in (i-1,i+1):
                    if 0<=j<len(self.clips) and j not in changes:
                        o=drag['orig'][j];changes[j]=(o.start,o.end)
            else:
                try:changes={i:resize_bounds(self.clips,i,drag['edge'],value,duration,self.rate)}
                except ValueError as e:self.status.set(str(e));return
            for j,(a,b) in changes.items():
                c=self.clips[j];c.start,c.end=a,b
                self.tree.item(str(j),values=(f'{c.start:.4f}',f'{c.end:.4f}',c.text,c.reading,c.name))
            self.cursor=getattr(clip,drag['edge'])
            self._loading_editor=True
            try:
                self.edit['start'].set(f'{clip.start:.8f}');self.edit['end'].set(f'{clip.end:.8f}')
            finally:self._loading_editor=False
            self.measure_label.configure(text=f'区間 {i+1}：{clip.start:.4f} ～ {clip.end:.4f} 秒 ｜ 長さ {clip.end-clip.start:.4f} 秒')
        self.draw()

    def wave_release(self,event):
        drag=self._wave_drag
        if not drag:return
        self.wave_drag(event);self._wave_drag=None
        if drag['mode']=='edge' and drag['before']['clips'] != [asdict(c) for c in self.clips]:
            self.history.record(drag['before'])
            self.status.set('区間の端を変更しました。Ctrl+Zで戻せます。')

    def lyrics_hint(self):
        return self.lyrics_text.get('1.0','end-1c').strip()

    def update_lyrics_header(self):
        raw=self.lyrics_hint();doc=LyricsDocument(raw)
        state=(f'LRC・{len(doc.lines)}時刻' if doc.timed else f'全文・{len(raw)}文字') if raw else '未入力'
        self.hint_button.configure(text=('▼' if self._lyrics_expanded else '▶')+' フル歌詞・LRC（ヒント：'+state+'）')

    def toggle_lyrics_hint(self):
        self.set_lyrics_expanded(not self._lyrics_expanded)

    def set_lyrics_expanded(self,expanded):
        self._lyrics_expanded=bool(expanded)
        if self._lyrics_expanded:self.hint_body.pack(fill='x')
        else:self.hint_body.pack_forget()
        self.update_lyrics_header()

    def lyrics_hint_changed(self,event=None):
        if self.lyrics_text.edit_modified():
            self.lyrics_text.edit_modified(False)
            self.update_lyrics_header()

    def clear_lyrics_hint(self):
        self.lyrics_text.delete('1.0','end')
        self.lyrics_shift.set('0')
        self.update_lyrics_header()

    def restore_lyrics_hint(self,data):
        self.clear_lyrics_hint()
        self.lyrics_text.insert('1.0',str(data.get('lyrics_hint','')))
        self.lyrics_text.edit_reset()
        self.lyrics_text.edit_modified(False)
        self.lyrics_shift.set(str(data.get('lyrics_shift',0)))
        self.set_lyrics_expanded(data.get('lyrics_hint_expanded',False))

    def lyrics_document(self):
        raw=self.lyrics_hint()
        try:shift=float(self.lyrics_shift.get()) if LyricsDocument(raw).timed else 0.
        except ValueError:raise ValueError('LRC時刻調整には秒数を入力してください。例：1.5 または -0.5')
        return LyricsDocument(raw,shift=shift,normalizer=get_reading)

    def open_lyrics_file(self):
        if not self.ready():return
        path=filedialog.askopenfilename(title='LRCまたはフル歌詞を開く',filetypes=[('LRC・歌詞テキスト','*.lrc *.txt'),('すべて','*.*')])
        if not path:return
        try:
            text=read_lyrics_file(path);doc=LyricsDocument(text)
            if Path(path).suffix.lower()=='.lrc' and not doc.timed:
                raise ValueError('LRCの時刻タグを読み取れません。[00:12.50]形式か確認してください。')
            if not any(line.text for line in doc.lines):raise ValueError('歌詞の本文がありません。')
            self.restore_lyrics_hint({'lyrics_hint':text,'lyrics_hint_expanded':True})
            self.status.set('歌詞を読み込みました。'+('LRCの時刻に合わせてヒントを選びます。' if doc.timed else '全文から区間ごとに参照候補を選びます。'))
        except Exception as e:self.error(e)

    def preview_lyrics(self):
        ids=self.selected()
        if self.audio is None or len(ids)!=1:
            self.status.set('参照候補を確認する行を1つ選んでください。');return
        try:
            doc=self.lyrics_document();clip=self.clips[ids[0]]
            priority,mode=doc.reference(clip.start,clip.end,len(self.audio)/self.rate,clip.text)
            text='\n'.join(doc.lines[i].text for i in sorted(priority)) or '参照候補なし'
            messagebox.showinfo('歌詞の参照候補',f'選択方法：{mode}\n\n{text[:2500]}\n\nこれは参考候補です。認識モデルへは渡しません。実際の補正は認識後に高い一致が得られた場合だけ行います。')
        except Exception as e:self.error(e)

    def ready(self):
        if self.busy:
            messagebox.showinfo('処理中','処理が終わるまでお待ちください。');return False
        return True

    def error(self,e):
        messagebox.showerror('エラー',str(e))

    def open_audio(self):
        if not self.ready():return
        p=filedialog.askopenfilename(filetypes=[('WAV音声','*.wav')])
        if p:self.load_audio_file(p)

    def load_audio_file(self,path):
        try:
            self.read_audio(path);self.clips=[];self.refresh()
            self.status.set('音声を読み込みました。「自動分割」を押してください。')
            return True
        except Exception as e:
            self.error(e)
            return False

    def drop_audio(self,event):
        if not self.ready():return REFUSE_DROP
        try:
            # TkDnD uses a Tcl list, including braces around paths with spaces.
            paths=self.tk.splitlist(event.data)
            if len(paths)!=1:
                self.status.set('WAVファイルを1つずつドロップしてください。');return REFUSE_DROP
            path=Path(paths[0])
            if path.suffix.lower()!='.wav' or not path.is_file():
                self.status.set('読み込めるWAVファイルをドロップしてください。');return REFUSE_DROP
            return COPY if self.load_audio_file(str(path)) else REFUSE_DROP
        except Exception as e:
            self.error(e)
            return REFUSE_DROP

    def read_audio(self,p):
        a,r=sf.read(p,dtype='float32',always_2d=True)
        if not len(a) or not np.isfinite(a).all():raise ValueError('音声が空か、無効な値を含んでいます。')
        self.stop();self._levels_key=None;self.audio=a;self.rate=r;self.path=str(Path(p).resolve());self.subtype=sf.info(p).subtype;self.cursor=0
        self.history.clear();self.zoom=1.;self.view_start=0.
        self.measurement=None;self._wave_drag=None
        self.measure_label.configure(text='Shift＋ドラッグ：時間を測る／選択区間の端をドラッグ：長さを変更／Ctrl＋クリック：複数選択／F2：歌詞編集／S：分割／Shift+S：結合')
        self.restore_lyrics_hint({})
        self.filelabel.configure(text=f'{Path(p).name} / {len(a)/r:.2f}秒 / {r}Hz / {a.shape[1]}ch')

    def settings(self):
        s=Settings(**{k:float(v.get()) for k,v in self.params.items()})
        if not all(math.isfinite(v) for v in asdict(s).values()) or not (-100<=s.threshold<=0 and .01<=s.gap<=10 and 0<=s.before<=5 and 0<=s.after<=5):
            raise ValueError('しきい値は-100～0、途切れは0.01～10秒、余白は0～5秒にしてください。')
        return s

    def segment(self):
        if not self.ready() or self.audio is None:return
        try:
            settings=self.settings()
            breath=BreathSettings(**{k:float(v.get()) for k,v in self.breath_params.items()})
            if self.clips and not messagebox.askyesno('再分割','適用すると現在の一覧を置き換えます。分割候補を計算しますか？'):return
            use_breath=self.use_breath.get();audio=self.audio;rate=self.rate
            position='middle' if self.breath_position.get()=='ブレスの中央' else 'start'
            self.busy=True;self.status.set('分割位置を計算中…ブレス候補は音響特徴からの推定です。')
            def worker():
                try:
                    clips=split_audio(audio,rate,settings)
                    candidates=detect_breaths(audio,rate,breath,position) if use_breath else []
                    # Candidates falling in silence gaps or too close to an edge cannot split a clip.
                    candidates=[c for c in candidates if any(x.start+breath.spacing<=c.split<=x.end-breath.spacing for x in clips)]
                    self.events.put(('segmented',(clips,candidates,breath.spacing,use_breath)))
                except Exception:self.events.put(('error',traceback.format_exc()))
            threading.Thread(target=worker,daemon=True).start()
        except Exception as e:self.error(e)

    def show_breath_preview(self,clips,candidates,spacing):
        win=tk.Toplevel(self);win.title('ブレス候補の確認');win.geometry('660x470')
        win.transient(self);win.grab_set()
        ttk.Label(win,text=f'無音分割：{len(clips)}区間 ／ 追加ブレス候補：{len(candidates)}件\n採用する候補を選択してください（Ctrl／Shiftで複数選択）。').pack(padx=10,pady=10)
        win.geometry('840x490')
        tree=ttk.Treeview(win,columns=('start','end','split','level','around','drop'),show='headings',selectmode='extended')
        for key,title,width in [('start','候補開始 秒',105),('end','候補終了 秒',105),('split','分割位置 秒',105),
                                ('level','ブレス音量 dBFS',135),('around','前後の歌声 dBFS',135),('drop','落差 dB',90)]:
            tree.heading(key,text=title);tree.column(key,width=width)
        tree.pack(fill='both',expand=True,padx=10)
        def stats(c):
            part=self.level_range(c.start,c.end)
            if part is None:return (float('nan'),)*3
            near=np.r_[self.level_range(max(0.,c.start-.5),c.start),self.level_range(c.end,c.end+.5)]
            level=self.mean_db(part);around=float(near.max())
            return level,around,around-level
        for i,c in enumerate(candidates):
            level,around,drop=stats(c)
            tree.insert('','end',iid=str(i),values=(f'{c.start:.3f}',f'{c.end:.3f}',f'{c.split:.3f}',f'{level:.1f}',f'{around:.1f}',f'{drop:.1f}'))
        tree.selection_set(tree.get_children());tree.focus_set()
        ttk.Label(win,text='選んだ分割位置（開始なら息の直前）で分割します。息の音は削除しません。\n子音・残響も候補になるため、前後を試聴して確認してください。\n'
                  '音量は10ms RMS。「前後の歌声」は候補の前後0.5秒の最大、「落差」はその差です（落差が小さい候補は子音などの誤検出の可能性）。').pack(padx=10,pady=8)
        buttons=ttk.Frame(win);buttons.pack(pady=10)
        def play_candidate():
            ids=tree.selection()
            if not ids:return
            c=candidates[int(ids[0])]
            try:
                import sounddevice as sd
                sd.stop();sd.play(self.audio[max(0,round((c.start-.5)*self.rate)):min(len(self.audio),round((c.end+.5)*self.rate))],self.rate)
            except Exception as e:self.error(e)
        def apply():
            chosen=[candidates[int(i)] for i in tree.selection()]
            result=split_at_candidates(clips,chosen,spacing)
            self.stop();self.checkpoint();self.clips=result;self.refresh()
            self.status.set(f'{len(result)}区間に分割しました。ブレスによる追加分割：{len(result)-len(clips)}箇所。Ctrl+Zで戻せます。')
            win.destroy()
        def cancel():self.stop();win.destroy()
        ttk.Button(buttons,text='全候補を選択',command=lambda:tree.selection_set(tree.get_children())).pack(side='left',padx=3)
        ttk.Button(buttons,text='選択解除',command=lambda:tree.selection_remove(*tree.selection())).pack(side='left',padx=3)
        ttk.Button(buttons,text='候補の前後を再生',command=play_candidate).pack(side='left',padx=3)
        ttk.Button(buttons,text='適用',command=apply).pack(side='left',padx=3)
        ttk.Button(buttons,text='キャンセル',command=cancel).pack(side='left',padx=3)
        win.protocol('WM_DELETE_WINDOW',cancel)

    def show_raw_text(self):
        ids=self.selected()
        if len(ids)!=1:self.status.set('認識原文を確認する行を1つ選んでください。');return
        c=self.clips[ids[0]]
        messagebox.showinfo('認識原文',f'認識原文（補正前）：\n{c.raw_text or "原文の記録なし（旧プロジェクト・手入力など）"}\n\n現在の歌詞：\n{c.text}')

    def close_inline(self):
        if self._inline is not None:
            widget=self._inline; self._inline=None
            widget.destroy()

    def refresh(self):
        self.close_inline()
        self._editor_id=None
        self._loading_editor=True
        try:
            self.tree.delete(*self.tree.get_children())
            for i,c in enumerate(self.clips):
                self.tree.insert('', 'end',iid=str(i),text=str(i+1),tags=('excluded',) if c.excluded else (),values=(f'{c.start:.4f}',f'{c.end:.4f}',c.text,c.reading,c.name))
            for v in self.edit.values():v.set('')
        finally:self._loading_editor=False
        self.draw()

    def selected(self):return sorted(int(i) for i in self.tree.selection())

    def select(self,event=None,move_cursor=True):
        ids=self.selected()
        target=ids[0] if len(ids)==1 else None
        if target!=self._editor_id:
            self.history.boundary()
            self.close_inline()
            self._editor_id=target
            self._loading_editor=True
            try:
                for k,v in self.edit.items():
                    v.set(str(getattr(self.clips[target],k)) if target is not None else '')
            finally:self._loading_editor=False
            if target is not None and move_cursor:self.cursor_to_row(target)
        self.draw()

    def cursor_to_row(self,index):
        self.cursor=self.clips[index].start
        if self.audio is not None:
            _,span=self.wave_extent()
            if not self.view_start<=self.cursor<self.view_start+span:
                self.view_start=self.cursor-span*.1
                self.wave_extent()

    def row_clicked(self,event):
        row=self.tree.identify_row(event.y)
        if row and self.selected()==[int(row)]:
            self.cursor_to_row(int(row));self.draw()

    def auto_edit(self,key):
        # Loading another row must never regenerate its manually corrected filename.
        if self._loading_editor or self.busy or self._editor_id is None:return
        i=self._editor_id
        if i>=len(self.clips):return
        if self.edit[key].get()==getattr(self.clips[i],key):return
        self.checkpoint(token=(i,key))
        self._loading_editor=True
        try:
            if key=='text':
                try:self.edit['reading'].set(get_reading(self.edit['text'].get(),self.reading_dictionary))
                except Exception as e:
                    self.edit['reading'].set('')
                    self.status.set('読みの変換に失敗しました。読み欄で修正してください: '+str(e))
            if key in ('text','reading'):
                reading=self.edit['reading'].get().strip()
                try:self.edit['name'].set(romanize(reading) if reading else '')
                except ValueError:
                    self.edit['name'].set('')
                    self.status.set('読みをかなで入力すると保存名を自動生成します。')
            c=self.clips[i]
            for field in ('text','reading','name'):
                setattr(c,field,self.edit[field].get())
            self.tree.item(str(i),values=(f'{c.start:.4f}',f'{c.end:.4f}',c.text,c.reading,c.name))
        finally:self._loading_editor=False

    def inline_edit(self,event):
        if self.busy:return
        row=self.tree.identify_row(event.y)
        col=self.tree.identify_column(event.x)
        key={'#3':'text','#4':'reading','#5':'name'}.get(col)
        if not row or key is None:return
        return self.begin_cell_edit(row,key)

    def begin_cell_edit(self,row,key):
        self.tree.selection_set(row);self.tree.see(row);self.select();self.close_inline()
        self.tree.update_idletasks()
        bounds=self.tree.bbox(row,key)
        if not bounds:return
        x,y,w,h=bounds
        entry=ttk.Entry(self.tree,textvariable=self.edit[key])
        entry.bindtags(('PlaybackShortcut',)+entry.bindtags())
        self._inline=entry
        entry.place(x=x,y=y,width=w,height=h)
        entry.focus_set();entry.select_range(0,'end')
        entry.bind('<Return>',self.finish_edit)
        entry.bind('<KP_Enter>',self.finish_edit)
        entry.bind('<Escape>',self.finish_edit)
        entry.bind('<FocusOut>',lambda e:self.close_inline())
        return 'break'

    def finish_edit(self,event=None,apply_times=False):
        self.history.boundary()
        if apply_times and not self.apply_edit():
            return 'break'
        self.close_inline()
        ids=self.selected()
        if ids:
            self.tree.focus(str(ids[0]))
            self.tree.see(str(ids[0]))
        self.tree.focus_set()
        return 'break'

    def fullwidth_space_play(self,event):
        # IMEs/Tk builds may report the Unicode character or a Unicode keysym.
        if getattr(event,'char','')=='\u3000' or getattr(event,'keysym','') in ('U3000','ideographicspace'):
            return self.space_play(event)
        return None

    def space_play(self,event):
        # Preserve spaces/IME conversion and native key actions in form controls.
        if event.widget.winfo_class() in ('Entry','TEntry','Text','TCombobox','Spinbox','TSpinbox'):
            return None
        if self._space_release_job is not None:
            self.after_cancel(self._space_release_job);self._space_release_job=None
        if self._space_down:return 'break'
        self._space_down=True
        if self.is_playing():
            self.stop()
        else:
            self.play(from_cursor=bool(getattr(event,'state',0)&0x1))
        return 'break'

    def release_space(self,event):
        if (getattr(event,'keysym','') in ('space','U3000','ideographicspace')
                or getattr(event,'char','') in (' ', '\u3000')
                or getattr(event,'keycode',None)==32):
            if self._space_release_job is not None:self.after_cancel(self._space_release_job)
            # Some Tk platforms emit release/press pairs during key repeat.
            self._space_release_job=self.after(20,self.reset_space_key)

    def reset_space_key(self):
        self._space_down=False;self._space_release_job=None

    def is_playing(self):
        try:
            import sounddevice as sd
            return bool(sd.get_stream().active)
        except (ImportError,OSError,RuntimeError):
            return False

    def apply_edit(self):
        if not self.ready():return False
        ids=self.selected()
        if len(ids)!=1:return False
        try:
            c=Clip(float(self.edit['start'].get()),float(self.edit['end'].get()),*[self.edit[k].get().strip() for k in ('text','reading','name')])
            c.excluded=self.clips[ids[0]].excluded
            c.raw_text=self.clips[ids[0]].raw_text
            validate_clip(c,len(self.audio)/self.rate)
            old=self.clips[ids[0]]
            if (c.start,c.end)!=(old.start,old.end):
                i=ids[0]
                if any(c.start<x.end for x in self.clips[:i]) or any(c.end>x.start for x in self.clips[i+1:]):
                    raise ValueError('隣接区間と重ならない開始・終了にしてください。')
            if c!=self.clips[ids[0]]:self.checkpoint()
            self.clips[ids[0]]=c;self.refresh();self.tree.selection_set(str(ids[0]));self.select(move_cursor=False)
            return True
        except Exception as e:
            self.error(e)
            return False

    def reading_from_text(self):
        if not self.ready():return
        try:self.edit['reading'].set(get_reading(self.edit['text'].get(),self.reading_dictionary))
        except Exception as e:self.error(e)

    def name_from_reading(self):
        if not self.ready():return
        try:self.edit['name'].set(romanize(self.edit['reading'].get()))
        except Exception as e:self.error(e)

    def wave_extent(self):
        duration=len(self.audio)/self.rate
        span=duration/self.zoom
        self.view_start=max(0.,min(self.view_start,duration-span))
        return duration,span

    def reset_zoom(self):
        self.zoom=1.;self.view_start=0.;self.draw()

    def wave_wheel(self,event):
        if self.audio is None:return 'break'
        num=getattr(event,'num',None)
        delta=1 if num==4 else -1 if num==5 else getattr(event,'delta',0)/120
        if not delta:return 'break'
        duration,span=self.wave_extent()
        if getattr(event,'state',0)&0x4:
            fraction=max(0.,min(1.,event.x/max(1,self.canvas.winfo_width())))
            anchor=self.view_start+fraction*span
            self.zoom=max(1.,min(256.,self.zoom*1.25**max(-8,min(8,delta))))
            self.view_start=anchor-fraction*duration/self.zoom
        else:
            self.view_start-=delta*span*.12
        self.draw()
        return 'break'

    def scroll_wave(self,*args):
        if self.audio is None:return
        duration,span=self.wave_extent()
        if args[0]=='moveto':self.view_start=float(args[1])*duration
        elif args[0]=='scroll':self.view_start+=int(args[1])*span*(.9 if args[2]=='pages' else .1)
        self.draw()

    def levels(self):
        """10ms RMS in dBFS over all channels, identical to the silence splitter."""
        if self.audio is None:return None
        key=(id(self.audio),len(self.audio),self.rate)
        if self._levels_key!=key:
            a=self.audio if self.audio.ndim==2 else self.audio[:,None]
            hop=max(1,round(self.rate*.01));n=len(a)//hop
            out=np.empty(n+(1 if len(a)%hop else 0))
            for i in range(0,n,20000):  # bounded memory for long songs
                j=min(n,i+20000)
                out[i:j]=np.mean(a[i*hop:j*hop].reshape(j-i,hop,a.shape[1]).astype(np.float64)**2,axis=(1,2))
            if len(a)%hop:out[-1]=np.mean(a[n*hop:].astype(np.float64)**2)
            self._levels=20*np.log10(np.sqrt(np.maximum(out,1e-24)))
            self._levels_key=key
        return self._levels

    def level_at(self,t):
        lv=self.levels()
        if lv is None or not len(lv):return None
        return float(lv[max(0,min(len(lv)-1,int(t/.01)))])

    def level_range(self,a,b):
        lv=self.levels()
        if lv is None or not len(lv):return None
        i=max(0,min(len(lv)-1,int(a/.01)));j=max(i+1,min(len(lv),int(math.ceil(b/.01))))
        return lv[i:j]

    @staticmethod
    def mean_db(part):
        # Mean power (not the mean of dB values), as a level meter would show.
        return 10*math.log10(max(float(np.mean(10**(np.asarray(part)/10))),1e-24))

    def level_summary(self,a,b):
        part=self.level_range(a,b)
        if part is None:return ''
        return f' ｜ 音量 平均 {self.mean_db(part):.1f} / 最大 {float(part.max()):.1f} / 最小 {float(part.min()):.1f} dBFS'

    def wave_hover(self,event):
        if self.audio is None or self._wave_drag:return
        t=self.wave_time(event.x);self.hover_time=t
        db=self.level_at(t)
        try:threshold=float(self.params['threshold'].get())
        except ValueError:threshold=None
        state='' if threshold is None else ('（しきい値未満＝無音扱い）' if db<threshold else '（しきい値以上＝音あり）')
        self.level_label.configure(text=f'マウス位置 {t:.3f} 秒：{db:.1f} dBFS {state}')
        if self.show_level.get():self.draw()

    def wave_leave(self,event):
        self.hover_time=None
        if self.show_level.get() and self.audio is not None:self.draw()

    def draw(self):
        c=self.canvas;c.delete('all')
        if self.audio is None:return
        w=max(c.winfo_width(),10);h=135
        duration,span=self.wave_extent();start=self.view_start;end=start+span
        self.wave_scroll.set(start/duration,end/duration)
        selected=set(self.selected())
        for i,clip in enumerate(self.clips):
            if clip.end<start or clip.start>end:continue
            x0=(clip.start-start)/span*w;x1=(clip.end-start)/span*w
            color='#3a3a3a' if clip.excluded else '#366f74' if i in selected else '#253f51'
            c.create_rectangle(x0,10,x1,h-26,fill=color,outline='#ffce70' if i in selected else '#688a98',width=2 if i in selected else 1)
            c.create_text(max(2,x0+4),18,text=str(i+1)+(' 除外' if clip.excluded else ''),fill='#aaaaaa' if clip.excluded else 'white',anchor='nw')
        # Limit work to the visible samples and screen pixels, even at high zoom.
        first=max(0,int(start*self.rate));last=min(len(self.audio),int(end*self.rate)+1)
        edges=np.linspace(first,last,w+1).astype(int)
        for x in range(w):
            lo=edges[x];hi=min(len(self.audio),max(lo+1,edges[x+1]))
            if lo>=len(self.audio):break
            block=self.audio[lo:hi];y=float(np.max(np.abs(block)))*40
            c.create_line(x,h/2-y,x,h/2+y,fill='#79dfc0')
        if self.show_level.get():self.draw_levels(c,w,h,start,span,edges)
        if self.measurement:
            a,b=sorted(self.measurement)
            c.create_rectangle((a-start)/span*w,25,(b-start)/span*w,h-25,outline='#ff89d1',width=2)
        for i in selected:
            for t in (self.clips[i].start,self.clips[i].end):
                x=(t-start)/span*w
                c.create_rectangle(x-3,40,x+3,90,fill='#ffce70',outline='')
        x=(self.cursor-start)/span*w
        if 0<=x<=w:c.create_line(x,0,x,h-20,fill='#ffce70',width=2)
        for n in range(5):
            fraction=n/4
            c.create_text(fraction*(w-10)+5,h-10,text=f'{start+span*fraction:.3f}s',fill='white',anchor='w' if n==0 else 'e' if n==4 else 'center')
        c.create_text(w-5,5,text=f'{self.zoom:.2f}倍 / カーソル {self.cursor:.4f}s',fill='#ffce70',anchor='ne')

    LEVEL_FLOOR=-70.
    def draw_levels(self,c,w,h,start,span,edges):
        lv=self.levels()
        if lv is None or not len(lv):return
        top,bottom=10,h-26;floor=self.LEVEL_FLOOR
        y_of=lambda db:bottom-(min(0.,max(floor,db))-floor)/-floor*(bottom-top)
        for db in (-60,-40,-20):
            y=y_of(db);c.create_line(0,y,w,y,fill='#2c4256',dash=(2,4))
            c.create_text(w-4,y-1,text=f'{db}dB',fill='#6f8ea3',anchor='se',font=('',7))
        hop=self.rate*.01;points=[]
        for x in range(w):
            i=int(edges[x]/hop);j=max(i+1,int(math.ceil(edges[x+1]/hop)))
            if i>=len(lv):break
            points+=[x,y_of(float(lv[i:j].max()))]
        if len(points)>=4:c.create_line(*points,fill='#ff9f43',width=1)
        try:threshold=float(self.params['threshold'].get())
        except ValueError:threshold=None
        if threshold is not None and math.isfinite(threshold):
            y=y_of(threshold)
            c.create_line(0,y,w,y,fill='#ff5c5c',dash=(6,3))
            c.create_text(4,y-1,text=f'無音しきい値 {threshold:g} dBFS',fill='#ff8080',anchor='sw')
        if self.hover_time is not None and start<=self.hover_time<=start+span:
            x=(self.hover_time-start)/span*w;db=self.level_at(self.hover_time)
            c.create_line(x,top,x,bottom,fill='#ffffff',dash=(1,3))
            c.create_text(x+4,y_of(db)-2,text=f'{db:.1f} dB',fill='white',anchor='sw')

    def clip_at(self,t):
        # Later-starting interval wins where padding creates an overlap.
        matches=[i for i,c in enumerate(self.clips) if c.start<=t<c.end
                 or (t==len(self.audio)/self.rate and c.end==t)]
        return max(matches,key=lambda j:(self.clips[j].start,j)) if matches else None

    def select_on_wave(self,e,extend=False):
        """Ctrl+click: add/remove one interval. Ctrl+Shift+click: select the range from the last one."""
        duration,span=self.wave_extent()
        self.cursor=max(0,min(duration,self.view_start+e.x/max(1,self.canvas.winfo_width())*span))
        self.canvas.focus_set()
        i=self.clip_at(self.cursor)
        if i is None:self.draw();return
        current=set(self.selected());anchor=getattr(self,'_wave_anchor',None)
        if extend and anchor is not None and 0<=anchor<len(self.clips):
            current|=set(range(min(anchor,i),max(anchor,i)+1))
        elif i in current:current.discard(i)
        else:current.add(i)
        if not extend:self._wave_anchor=i
        self.tree.selection_set([str(j) for j in sorted(current)])
        if i in current:self.tree.focus(str(i));self.tree.see(str(i))
        self.select(move_cursor=False)
        self.status.set(f'{len(current)}区間を選択中。Shift+Sで結合（連続した区間のみ）。')

    def click_wave(self,e):
        if self.audio is None:return
        duration,span=self.wave_extent()
        self.cursor=max(0,min(duration,self.view_start+e.x/max(1,self.canvas.winfo_width())*span))
        self.canvas.focus_set()
        i=self.clip_at(self.cursor);self._wave_anchor=i
        if i is not None:
            self.tree.selection_set(str(i));self.tree.focus(str(i));self.tree.see(str(i))
        else:self.tree.selection_remove(*self.tree.selection())
        self.select(move_cursor=False)

    def play(self,from_cursor=False):
        if self.audio is None:return
        ids=self.selected()
        if not ids:return
        if from_cursor and len(ids)!=1:
            self.status.set('バーから再生する区間を1行だけ選択してください。');return
        c=self.clips[ids[0]]
        start=self.cursor if from_cursor else c.start
        if from_cursor and not c.start<=start<c.end:
            self.status.set('選択区間の内側に分割バーを置いてください。');return
        first=round(start*self.rate);last=round(c.end*self.rate)
        if first>=last:
            self.status.set('再生できる長さがありません。バーを少し前へ移動してください。');return
        try:
            import sounddevice as sd
            sd.stop();sd.play(self.audio[first:last],self.rate)
        except Exception as e:self.error(e)

    def stop(self):
        try:
            import sounddevice as sd
            sd.stop()
        except (ImportError, OSError):pass

    def split_here(self):
        if not self.ready():return
        ids=self.selected()
        if len(ids)!=1:return
        i=ids[0];c=self.clips[i];t=self.cursor
        if not c.start<t<c.end:self.error('選択した区間の内側をクリックしてください。');return
        self.checkpoint()
        self.clips[i:i+1]=[Clip(c.start,t,excluded=c.excluded),Clip(t,c.end,excluded=c.excluded)];self.refresh();self.status.set('区間を分割しました。2つの区間の歌詞・名前を設定してください。')

    def merge(self):
        if not self.ready():return
        ids=self.selected()
        if len(ids)<2:self.status.set('結合する区間を2つ以上選んでください（波形はCtrl＋クリックで複数選択）。');return
        if ids!=list(range(ids[0],ids[-1]+1)):self.error('連続した行を選択してください。');return
        group=[self.clips[i] for i in ids]
        if len({c.excluded for c in group})>1:
            self.error("除外行と有効行は混ぜて結合できません。先に除外状態をそろえてください。");return
        self.checkpoint()
        self.clips[ids[0]:ids[-1]+1]=[Clip(min(c.start for c in group),max(c.end for c in group),''.join(c.text for c in group),''.join(c.reading for c in group),excluded=group[0].excluded)];self.refresh()

    def delete(self):
        if not self.ready():return
        ids=self.selected()
        if not ids:return
        excluded=not all(self.clips[i].excluded for i in ids)
        self.checkpoint()
        for i in ids:self.clips[i].excluded=excluded
        self.refresh();self.tree.selection_set([str(i) for i in ids]);self.select()
        self.status.set('選択行を除外しました。連番は欠番になります。' if excluded else '選択行の除外を解除しました。')

    def snapshot(self):
        return {'clips':[asdict(c) for c in self.clips], 'selected':self.selected(), 'cursor':self.cursor}

    def checkpoint(self,token=None):
        self.history.record(self.snapshot(),token)

    def restore(self,state):
        self.stop()
        self.clips=[Clip(**c) for c in state['clips']];self.cursor=state['cursor']
        self.refresh()
        ids=[str(i) for i in state['selected'] if i<len(self.clips)]
        self.tree.selection_set(ids);self.select(move_cursor=False)
        if ids:self.tree.focus(ids[0]);self.tree.see(ids[0])
        self.tree.focus_set()

    def undo(self,event=None):
        if event is not None and event.widget is self.lyrics_text:
            try:self.lyrics_text.edit_undo()
            except tk.TclError:pass
            return 'break'
        if not self.ready():return 'break'
        state=self.history.undo(self.snapshot())
        if state is not None:self.restore(state);self.status.set('直前の編集を取り消しました。')
        return 'break'

    def redo(self,event=None):
        if event is not None and event.widget is self.lyrics_text:
            try:self.lyrics_text.edit_redo()
            except tk.TclError:pass
            return 'break'
        if not self.ready():return 'break'
        state=self.history.redo(self.snapshot())
        if state is not None:self.restore(state);self.status.set('編集をやり直しました。')
        return 'break'

    def recognition_targets(self,selected_only=False):
        return self.selected() if selected_only else [i for i,c in enumerate(self.clips) if not c.excluded]

    def apply_recognition(self,updates):
        self.checkpoint()
        for index,clip in updates:self.clips[index]=clip
        self.refresh()
        self.tree.selection_set([str(index) for index,_ in updates]);self.select()

    def recognize(self,selected_only=False):
        if not self.ready() or not self.clips:return
        ids=self.recognition_targets(selected_only)
        if not ids:
            self.status.set('再生成する行を選択してください。' if selected_only else '認識対象の有効な行がありません。');return
        if any(self.clips[i].text or self.clips[i].name for i in ids) and not messagebox.askyesno('AI認識',f'対象の{len(ids)}行の歌詞・読み・保存名をAIの結果で置き換えますか？'):return
        modelname=self.model.get();device=self.device.get()
        if modelname==ALIGN_LABEL:return self.align_lyrics(ids)
        clips=[Clip(**asdict(self.clips[i])) for i in ids];a=self.audio;rate=self.rate
        try:lyrics=self.lyrics_document()
        except Exception as e:self.error(e);return
        duration=len(a)/rate;dictionary=dict(self.reading_dictionary)
        cleanup=self.cleanup_romaji.get();hint=self.lyrics_hint();correct=self.post_correct.get()
        self.busy=True;self.status.set('モデルを準備しています。歌詞ヒントは認識後の照合だけに使用します。')
        def worker():
            try:
                from scipy.signal import resample_poly
                self.recognizer.load(modelname,device,lambda message:self.events.put(('status',message)))
                g=math.gcd(rate,16000);corrected=0
                for i,clip in enumerate(clips):
                    self.events.put(('status',f'{modelname}で認識中 {i+1}/{len(clips)}'))
                    validate_clip(clip,duration)
                    part=a[round(clip.start*rate):round(clip.end*rate)]
                    if not len(part):raise ValueError('音声サンプルを含まない短い区間があります。端を調整してください。')
                    channel=int(np.argmax(np.mean(part.astype(np.float64)**2,axis=0)))
                    mono=resample_poly(part[:,channel],16000//g,rate//g).astype(np.float32)
                    clip.raw_text=self.recognizer.transcribe(mono)
                    clip.text=remove_romaji_duplicates(clip.raw_text,dictionary,hint)[0] if cleanup else clip.raw_text
                    if correct and lyrics.lines:
                        clip.text,changed=correct_from_lyrics(lyrics,clip.text,clip.start,clip.end,duration)
                        corrected+=int(changed)
                    clip.reading='';clip.name=''
                    if clip.text:
                        clip.reading=get_reading(clip.text,dictionary)
                        try:clip.name=romanize(clip.reading)
                        except ValueError:clip.name=''
                self.events.put(('recognized',(list(zip(ids,clips)),corrected)))
            except Exception:
                self.recognizer.unload()
                self.events.put(('error',traceback.format_exc()))
        threading.Thread(target=worker,daemon=True).start()

    def align_lyrics(self,ids):
        """Qwen3-ForcedAligner: cut the full lyrics into the intervals (no recognition of the words)."""
        from lyric_aligner import lyric_lines, align_timed, align_plain, assign_words, texts_for_clips
        try:
            shift=self.lyrics_document().shift
            lines=lyric_lines(self.lyrics_hint(),shift)
        except Exception as e:self.error(e);return
        if not any(text for _t,text in lines):
            self.error('ForcedAlignerは歌詞を音声に合わせる方式です。「フル歌詞・LRC」欄に歌詞を入れてください（LRC推奨）。');return
        timed=lines[0][0] is not None
        allclips=[Clip(**asdict(c)) for c in self.clips];a=self.audio;rate=self.rate;device=self.device.get()
        duration=len(a)/rate;dictionary=dict(self.reading_dictionary);targets=set(ids)
        self.busy=True
        self.status.set('LRCの時刻を手がかりに歌詞を合わせます。' if timed else '時刻なしの歌詞：全区間をQwen3-ASRで認識してから歌詞を割り当てます。')
        def worker():
            try:
                from scipy.signal import resample_poly
                g=math.gcd(rate,16000)
                channel=int(np.argmax([np.dot(a[:,c].astype(np.float32),a[:,c].astype(np.float32)) for c in range(a.shape[1])]))
                mono=resample_poly(a[:,channel],16000//g,rate//g).astype(np.float32)
                transcripts=None
                if not timed:
                    # Recognised text only anchors the lyrics; reuse it for rows that are not regenerated.
                    need=[k for k,c in enumerate(allclips) if k in targets or not c.raw_text]
                    transcripts=[c.raw_text for c in allclips]
                    if need:
                        self.recognizer.load(QWEN_LABEL,device,lambda m:self.events.put(('status',m)))
                        for n,k in enumerate(need):
                            self.events.put(('status',f'歌詞の位置を探すため認識中 {n+1}/{len(need)}'))
                            c=allclips[k];part=mono[round(c.start*16000):round(c.end*16000)]
                            try:transcripts[k]=self.recognizer.transcribe(part)
                            except RuntimeError:transcripts[k]=''  # e.g. token limit: the aligner decides
                self.recognizer.load(ALIGN_LABEL,device,lambda m:self.events.put(('status',m)))
                if timed:
                    self.events.put(('status','LRCの各行を音声に合わせています…'))
                    source,words=align_timed(self.recognizer.align,mono,lines,duration)
                    groups=assign_words(words,allclips)
                else:
                    split=lambda text:self.recognizer.processor.split_words_for_alignment(text,'Japanese')
                    source,groups=align_plain(self.recognizer.align,split,mono,lines,allclips,transcripts,get_reading,
                        notify=lambda i,n:self.events.put(('status',f'区間の境目を調整中 {i}/{n}')))
                texts=texts_for_clips(source,groups)
                updates=[]
                for k in ids:
                    clip=allclips[k];clip.text=texts.get(k,'')
                    if transcripts is not None:clip.raw_text=transcripts[k]
                    clip.reading='';clip.name=''
                    if clip.text:
                        clip.reading=get_reading(clip.text,dictionary)
                        try:clip.name=romanize(clip.reading)
                        except ValueError:clip.name=''
                    updates.append((k,clip))
                self.events.put(('aligned',(updates,timed)))
            except Exception:
                self.recognizer.unload()
                self.events.put(('error',traceback.format_exc()))
        threading.Thread(target=worker,daemon=True).start()

    def poll(self):
        try:
            while True:
                kind,data=self.events.get_nowait()
                if kind=='status':self.status.set(data)
                elif kind=='segmented':
                    self.busy=False;clips,candidates,spacing,use_breath=data
                    if use_breath:self.status.set('ブレス候補を確認してください。');self.show_breath_preview(clips,candidates,spacing)
                    else:
                        self.checkpoint();self.clips=clips;self.refresh();self.status.set(f'{len(clips)}区間に無音分割しました。')
                elif kind=='recognized':
                    self.busy=False;updates,corrected=data;self.apply_recognition(updates)
                    self.status.set(f'認識完了。歌詞照合による補正：{corrected}行。「選択行の認識原文」で補正前を確認できます。')
                elif kind=='aligned':
                    self.busy=False;updates,timed=data;self.apply_recognition(updates)
                    empty=sum(1 for _k,c in updates if not c.text)
                    self.status.set(f'歌詞合わせ完了（{"LRC時刻" if timed else "認識結果"}を手がかり）：{len(updates)-empty}行に歌詞を割り当て、{empty}行は該当なし。境目を確認してください。')
                elif kind=='done':
                    self.busy=False;self.apply_recognition(data);self.status.set('AI認識が完了しました。歌詞・読み・分割位置を確認して保存してください。')
                else:
                    self.busy=False;self.status.set('処理に失敗しました。現在の一覧は保持されています。エラー内容を確認してください。');self.error(data)
        except queue.Empty:pass
        self.after(100,self.poll)

    def export(self):
        if not self.ready() or not self.clips:return
        if not any(not c.excluded for c in self.clips):
            self.status.set('全行が除外されています。除外を解除してください。');return
        if any(not c.name for c in self.clips if not c.excluded):
            if not messagebox.askyesno('未設定の保存名','名前が空の区間は xclip001x.wav などの仮名で保存します。続けますか？'):return
        folder=filedialog.askdirectory(title='WAVの保存先')
        if not folder:return
        try:
            paths=export_clips(self.audio,self.rate,self.clips,folder,self.subtype,self.numbered.get());self.status.set(f'{len(paths)}ファイルを保存しました。同名ファイルは上書きせず末尾に番号を付けます。')
            messagebox.showinfo('保存完了',f'{len(paths)}ファイルを保存しました。\n{folder}')
        except Exception as e:self.error(e)

    def save_project(self):
        if not self.ready() or self.audio is None:return
        p=filedialog.asksaveasfilename(defaultextension='.json',filetypes=[('作業プロジェクト','*.json')])
        if not p:return
        try:
            data={'version':7,'model':self.model.get(),'device':self.device.get(),'post_correct':self.post_correct.get(),'use_breath':self.use_breath.get(),'breath_settings':{k:float(v.get()) for k,v in self.breath_params.items()},'breath_position':self.breath_position.get(),'cleanup_romaji':self.cleanup_romaji.get(),'reading_dictionary':self.reading_dictionary,'source':self.path,'settings':asdict(self.settings()),'clips':[asdict(c) for c in self.clips],'numbered':self.numbered.get(),'lyrics_hint':self.lyrics_hint(),'lyrics_hint_expanded':self._lyrics_expanded,'lyrics_shift':self.lyrics_document().shift}
            Path(p).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8');self.status.set('プロジェクトを保存しました。元音声も保管してください。')
        except Exception as e:self.error(e)

    def load_project(self):
        if not self.ready():return
        p=filedialog.askopenfilename(filetypes=[('作業プロジェクト','*.json')])
        if not p:return
        try:
            data=json.loads(Path(p).read_text(encoding='utf-8'));source=data['source']
            if not Path(source).exists():source=filedialog.askopenfilename(title='元のWAV音声を指定',filetypes=[('WAV音声','*.wav')])
            if not source:return
            clips=[Clip(**c) for c in data['clips']]
            duration=sf.info(source).duration
            for c in clips:validate_clip(c,duration)
            dictionary=data.get('reading_dictionary',self.reading_dictionary)
            if not isinstance(dictionary,dict) or not all(isinstance(k,str) and isinstance(v,str) for k,v in dictionary.items()):
                raise ValueError('読み辞書の形式が不正です。')
            self.read_audio(source);self.clips=clips
            self.reading_dictionary=dict(dictionary)
            name=data.get('model',QWEN_LABEL)
            if name in ('tiny','base','small','medium','large-v3'):name='Whisper '+name
            self.model.set(name if name in MODELS else QWEN_LABEL)
            device=data.get('device','自動');self.device.set(device if device in ('自動','GPU','CPU') else '自動')
            self.post_correct.set(bool(data.get('post_correct',True)))
            self.use_breath.set(bool(data.get('use_breath',True)))
            self.breath_position.set(data.get('breath_position','ブレスの開始') if data.get('breath_position') in ('ブレスの開始','ブレスの中央') else 'ブレスの開始')
            for k,v in asdict(BreathSettings()).items():self.breath_params[k].set(str(data.get('breath_settings',{}).get(k,v)))
            self.cleanup_romaji.set(bool(data.get('cleanup_romaji',True)))
            self.restore_lyrics_hint(data)
            for k,v in data.get('settings',{}).items():
                if k in self.params:self.params[k].set(str(v))
            self.numbered.set(data.get('numbered',True));self.refresh()
        except Exception as e:self.error(e)

    def close(self):
        if self.busy and not messagebox.askyesno('終了','AI処理中です。終了しますか？'):return
        self.stop();self.destroy()

if __name__=='__main__':
    try:
        App().mainloop()
    except Exception:
        traceback.print_exc()
        try:
            messagebox.showerror('起動エラー', traceback.format_exc()[-1500:])
        except Exception:
            pass
        raise
