"""Run inside the app venv. Install the native HF Qwen backend for Windows/Turing."""
from pathlib import Path
import subprocess,sys,shutil


def install(mode):
    uv=shutil.which('uv')
    if uv:command=[uv,'pip','install','--python',sys.executable]
    else:
        subprocess.run([sys.executable,'-m','ensurepip','--upgrade'],check=True)
        command=[sys.executable,'-m','pip','install']
    index='https://download.pytorch.org/whl/'+('cu124' if mode=='gpu' else 'cpu')
    subprocess.run(command+['torch==2.6.0+'+('cu124' if mode=='gpu' else 'cpu'),'--index-url',index],check=True)
    subprocess.run(command+['-r',str(Path(__file__).with_name('requirements-qwen.txt'))],check=True)
    subprocess.run([sys.executable,'-c',
        'import torch, nagisa; from transformers import Qwen3ASRForConditionalGeneration, Qwen3ASRForTokenClassification, AutoProcessor; '
        'print("Qwen ready. CUDA:",torch.cuda.is_available())'],check=True)

if __name__=='__main__':
    try:install(sys.argv[1] if len(sys.argv)>1 else 'gpu')
    except Exception as e:
        print('Qwen setup failed:',e);sys.exit(1)
