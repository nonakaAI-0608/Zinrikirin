# ボーカル自動切り分け（vocal_splitter）

ボーカル音源を、無音・ブレス候補・歌詞の行に合わせて自動で区間に切り分ける Windows 向け GUI ツールです。
歌詞の時刻合わせに faster-whisper、または Qwen3-ASR（任意）を使います。

## 必要なもの
- Windows 10 / 11
- Python 3.11（`py -3.11` で起動できること）または [uv](https://docs.astral.sh/uv/)
- Qwen3-ASR を使う場合：NVIDIA GPU 推奨（CPU でも可）

## インストール
1. このリポジトリを ZIP でダウンロードして展開するか、`git clone` します。
2. `start.bat` をダブルクリックします。初回はフォルダ内に `.venv` を作り、`requirements.txt` のライブラリを入れます。
   - uv があれば uv、無ければ `py -3.11 -m venv` を使います。
3. （任意）Qwen3-ASR を使う場合は、アプリを閉じてから次のどちらかを実行します。
   - `setup_qwen_gpu.bat` … PyTorch 2.6.0 CUDA 12.4 ＋ `requirements-qwen.txt`
   - `setup_qwen_cpu.bat` … PyTorch 2.6.0 CPU 版 ＋ `requirements-qwen.txt`

   モデルは初回使用時に `models/` フォルダへ自動でダウンロードされます（Qwen3-ASR 一式で数GB）。

### 依存ライブラリ
- `requirements.txt`：numpy, scipy, soundfile, sounddevice, fugashi, unidic-lite, faster-whisper, tkinterdnd2, alkana
- `requirements-qwen.txt`：transformers, accelerate, librosa, soundfile, nagisa（＋ setup_qwen.py が入れる torch）

## 使い方
詳しい操作説明・更新履歴は [README.txt](README.txt) を参照してください。
読みの辞書は、アプリで編集すると `reading_dictionary.json` に保存されます（リポジトリには含めません）。
モデルの出典は [SOURCES.txt](SOURCES.txt) にあります。

## ライセンス
MIT License（[LICENSE](LICENSE)）。モデル（Qwen3-ASR：Apache-2.0、faster-whisper）やライブラリはそれぞれのライセンスに従います。
