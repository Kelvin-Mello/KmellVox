# 🎙️ KmellVox - Pipeline com IA para Dublagem, Clonagem de Voz e Sincronia Labial

O **KmellVox** é uma solução completa para localização e dublagem de vídeos com inteligência artificial, projetada com arquitetura modular de alto desempenho e seleção dinâmica de modelos conforme a VRAM disponível.

---

## 🏗️ Estrutura do Projeto

```text
KmellVox/
├── core/
│   ├── __init__.py
│   ├── hardware.py        # Detecção de VRAM, GPU e seleção de perfil (perfil_a, perfil_b, cpu)
│   ├── audio_extract.py   # Extração de áudio mono 16kHz via ffmpeg-python
│   ├── transcribe.py      # Transcrição com timestamps (faster-whisper) + exportação SRT + VRAM unload
│   ├── translate.py       # Tradução contextualizada em lote (llama-cpp-python + Qwen3 GGUF)
│   ├── voice_clone.py     # Clonagem de voz zero-shot (F5-TTS / IndexTTS-2)
│   ├── lipsync.py         # Sincronização labial facial (MuseTalk 1.5)
│   ├── assemble.py        # Remontagem final e muxing via ffmpeg
│   └── pipeline.py        # Orquestrador unificado de ponta a ponta
├── models/                # Pesos baixados dos modelos (ignorado no git)
│   └── .gitkeep
├── downloader/
│   ├── __init__.py
│   └── fetch_models.py    # Download e verificação de pesos conforme o perfil
├── ui/
│   ├── __init__.py
│   ├── main_window.py     # Interface gráfica moderna em PySide6 com drag-and-drop e fila
│   ├── queue_widget.py    # Fila de processamento em lote
│   └── settings_dialog.py # Diálogo de preferências, hardware e download de modelos
├── packaging/
│   ├── pyinstaller.spec   # Especificação de build com PyInstaller
│   └── installer.iss      # Script do instalador Windows (Inno Setup)
├── tests/                 # Suíte de 34 testes unitários automatizados
├── config.yaml            # Configurações globais, gpu_profile e caminhos locais
├── requirements.txt       # Dependências base do ambiente Python
├── main.py                # Ponto de entrada (GUI e CLI)
└── README.md
```

---

## ⚙️ Pré-requisitos & Requisitos de Sistema

1. **Sistema Operacional:** Windows 10/11 (x64) ou Linux (x64).
2. **Python:** Python 3.11 (versão recomendada para compatibilidade de CUDA e PySide6).
3. **Placa de Vídeo (GPU):** NVIDIA com suporte a CUDA (mínimo 5GB VRAM para aceleração completa por GPU).
4. **FFmpeg:** Binário estático do FFmpeg (com suporte a codecs de áudio/vídeo).
5. **Inno Setup 6+ (Opcional):** Para compilação do instalador executável `.exe` de distribuição no Windows.

---

## 🚀 Instalação & Configuração do Ambiente Virtual

### 1. Criar e Ativar o Ambiente Virtual (Python 3.11)

Abra o terminal (PowerShell) na raiz do projeto `KmellVox`:

```powershell
# Criação do ambiente virtual com Python 3.11
py -3.11 -m venv .venv

# Ativação no Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
```

### 2. Instalar o PyTorch com Suporte a CUDA (Obrigatório para Aceleração por GPU)

> [!IMPORTANT]
> **Atenção:** Instalar o `torch` padrão via `pip install torch` sem especificar o `--index-url` oficial do PyTorch instalará a versão **CPU-only**, quebrando a detecção de VRAM da GPU (`torch.cuda.get_device_properties`), a clonagem de voz F5-TTS/IndexTTS-2 e a sincronia labial MuseTalk.

Execute o comando correspondente à versão de CUDA suportada pela sua GPU:

#### Para GPUs Modernas (RTX 40 / 50 Series - CUDA 12.4):
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

#### Para GPUs com CUDA 12.1:
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

### 3. Instalar as Dependências do KmellVox

#### Opção A: Instalação Completa para GPU (Recomendada)
Instala todas as dependências de IA (F5-TTS, MuseTalk 1.5, IndexTTS-2, aceleração de tensores e processamento de mídia):
```powershell
pip install -r requirements-gpu.txt
```

#### Opção B: Instalação Base (Leve / Modo CPU)
```powershell
pip install -r requirements.txt
```

---

### ⚡ Aceleração por GPU no llama-cpp-python (CUDA / cuBLAS)

Para que o `llama-cpp-python` utilize a GPU NVIDIA com aceleração máxima cuBLAS/CUDA durante a tradução, defina a variável de ambiente `CMAKE_ARGS="-DGGML_CUDA=on"` antes da instalação:

#### Windows (PowerShell):
```powershell
$env:CMAKE_ARGS = "-DGGML_CUDA=on"
pip install llama-cpp-python --force-reinstall --no-cache-dir
```

#### Linux / macOS (Bash):
```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

---

## 🎬 Configuração do FFmpeg

O KmellVox busca o executável automaticamente nos seguintes locais:
1. No caminho configurado em `config.yaml` (`paths.ffmpeg_bin`): `tools/ffmpeg/bin/ffmpeg.exe`.
2. No PATH global do sistema Windows.

Se o FFmpeg for colocado na pasta `tools/ffmpeg/bin/`, o sistema o detecta imediatamente.

---

## 🧠 Perfis de Hardware & Modelos (`ModelProfile`)

O KmellVox detecta automaticamente a capacidade da GPU e resolve os modelos ideais:

| Perfil | Faixa de VRAM | Faster-Whisper | Compute Type | Modelo Tradutor LLM | IndexTTS-2 | MuseTalk `--use_float16` |
|---|---|---|---|---|---|---|
| **`perfil_a`** | $\ge$ 7.5 GB VRAM | `large-v3` | `float16` | Qwen3-8B-Instruct Q4_K_M | Habilitado (`True`) | Opcional (`False`) |
| **`perfil_b`** | 5.0 a 7.5 GB VRAM | `distil-large-v3` | `int8_float16` | Qwen3-4B-Instruct Q4_K_M | Desabilitado (`False`) | Obrigatório (`True`) |
| **`cpu`** | $<$ 5.0 GB / Sem CUDA | `small` | `int8` | Qwen3-1.5B-Instruct Q4_K_M | Desabilitado (`False`) | Desabilitado (`False`) |

---

## 📦 Download de Pesos dos Modelos

Para verificar e baixar os modelos correspondentes ao seu perfil de hardware:

```powershell
# Verificar o status dos modelos locais
python downloader/fetch_models.py --status

# Baixar apenas os modelos necessários para o perfil detectado
python main.py --fetch-models
```

---

## 🖥️ Execução

### Modo Interface Gráfica (PySide6)

```powershell
python main.py
```

### Modo Linha de Comando (CLI)

```powershell
python main.py --cli -i "caminho/do/video.mp4" -o "output/video_dublado.mp4" --target-lang pt
```

---

## 🏗️ Como Gerar o Instalador Final do Windows (.exe)

O processo de empacotamento do KmellVox é dividido em 2 etapas: geração do executável com o **PyInstaller** e empacotamento do instalador com o **Inno Setup**.

> **Nota:** Os pesos dos modelos (vários GBs) são **excluídos intencionalmente** do executável para manter o instalador leve (~80MB). Na primeira execução pós-instalação, o app abre diretamente a tela de instalação de modelos para o usuário baixá-los de forma sob demanda.

### Passo 1: Gerar a pasta de distribuição com o PyInstaller

Certifique-se de que o ambiente virtual está ativo e execute:

```powershell
# Instala o PyInstaller no ambiente virtual (se ainda não instalado)
pip install pyinstaller

# Compila o projeto a partir do spec
pyinstaller packaging/pyinstaller.spec --clean --noconfirm
```

Isso gerará a pasta `dist/KmellVox/` contendo `KmellVox.exe` e todas as DLLs e dependências necessárias.

### Passo 2: Compilar o Instalador com o Inno Setup

1. Baixe e instale o [Inno Setup 6](https://jrsoftware.org/isinfo.php) (se ainda não tiver no sistema).
2. Abra o arquivo `packaging/installer.iss` no Inno Setup Compiler e clique em **Compile (F9)**, ou execute via linha de comando:

```powershell
# Executa o compilador Inno Setup via linha de comando
& "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" packaging/installer.iss
```

3. O instalador final será gerado em:
   ```text
   packaging/dist/installer/KmellVox_Setup_v1.0.0.exe
   ```

### O que o instalador faz:
- Instala o KmellVox em `C:\Program Files\KmellVox` (ou na pasta escolhida pelo usuário).
- Cria atalhos no **Menu Iniciar** e na **Área de Trabalho**.
- Na primeira execução após a instalação, inicia o aplicativo com o parâmetro `--first-run`, abrindo imediatamente a janela de download dos modelos.

---

## 🧪 Testes Automatizados

Para executar toda a suíte de testes unitários do sistema:

```powershell
.\.venv\Scripts\python.exe -m unittest discover tests -v
```
