# 🎙️ Speech to Text Studio

A local Python + Streamlit speech-to-text application with:

1. **Audio file transcription**
2. **Video file transcription**
3. **Live microphone recording + transcription**
4. **Automatic language detection**
5. **Optional language selection**
6. **Plain TXT transcript download**
7. **Timestamped CSV download**
8. **Local Whisper inference using `faster-whisper`**

The project is designed to run locally. Your audio/video is processed by the local Whisper model rather than being uploaded to a transcription API.

---

## 1. Requirements

- Windows / Linux / macOS
- Python 3.10–3.13 recommended
- Internet connection for the first package/model download
- A working microphone for live transcription

### FFmpeg

`faster-whisper` normally handles decoding through PyAV, so a separate FFmpeg installation is generally not required for common media formats. If you encounter decoding problems with a particular video/audio format, install FFmpeg and add it to PATH.

---

## 2. Create a virtual environment

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Windows CMD

```cmd
python -m venv .venv
.venv\Scripts\activate
```

### Git Bash

```bash
python -m venv .venv
source .venv/Scripts/activate
```

---

## 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. Run the application

```bash
streamlit run app.py
```

Then open the URL shown by Streamlit, usually:

```text
http://localhost:8501
```

---

## 5. Audio / video transcription

Open **Audio / Video File**.

Supported examples:

- MP3
- WAV
- M4A
- AAC
- FLAC
- OGG
- OPUS
- MP4
- MOV
- MKV
- AVI
- WEBM
- MPEG

Upload the file, choose a language or Auto Detect, then click **Transcribe File**.

The application displays:

- Full transcript
- Detected language
- Timestamped segments
- Downloadable `.txt`
- Downloadable timestamped `.csv`

---

## 6. Live microphone transcription

Open **Live Microphone**.

1. Select the language or Auto Detect.
2. Click **Start recording**.
3. Speak into your microphone.
4. Click **Stop recording**.
5. Click **Convert Recording to Text**.

The recorded audio is then transcribed using the same local Whisper model.

### Important

This version is **near-real-time / record-then-transcribe**, not word-by-word streaming.

A true streaming version can be added later using short microphone chunks and incremental Whisper decoding.

---

## 7. Whisper model configuration

The default model is:

```text
small
```

You can change it using environment variables.

### Smaller / faster

```text
WHISPER_MODEL=base
```

### Better accuracy

```text
WHISPER_MODEL=medium
```

### CPU

```text
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
```

### NVIDIA GPU

If your CUDA/PyTorch environment is configured:

```text
WHISPER_DEVICE=cuda
WHISPER_COMPUTE_TYPE=float16
```

Example PowerShell:

```powershell
$env:WHISPER_MODEL="small"
$env:WHISPER_DEVICE="cpu"
$env:WHISPER_COMPUTE_TYPE="int8"
streamlit run app.py
```

---

## 8. Project structure

```text
speech_to_text_project/
│
├── app.py
├── requirements.txt
└── README.md
```

---

## 9. Architecture

```text
                    ┌─────────────────────┐
                    │   Streamlit UI      │
                    └──────────┬──────────┘
                               │
              ┌────────────────┴────────────────┐
              │                                 │
       Audio / Video File                 Microphone
              │                                 │
              ▼                                 ▼
        Temporary file                    WAV recording
              │                                 │
              └────────────────┬────────────────┘
                               ▼
                    ┌─────────────────────┐
                    │   faster-whisper    │
                    │   Whisper Model     │
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────┐
                    │   Transcription     │
                    │   + timestamps      │
                    └──────────┬──────────┘
                               ▼
             ┌─────────────────┴─────────────────┐
             │                                   │
        Text Transcript                    CSV timestamps
             │
             ▼
        TXT Download
```

---

## 10. Troubleshooting

### `ModuleNotFoundError`

Run:

```bash
pip install -r requirements.txt
```

### Microphone does not work

- Allow microphone permission in your browser.
- Use Chrome/Edge if your browser has compatibility issues.
- Restart Streamlit after installing `streamlit-mic-recorder`.

### Transcription is slow

Try:

```text
WHISPER_MODEL=base
```

or, for maximum speed:

```text
WHISPER_MODEL=tiny
```

### Accuracy is not sufficient

Try:

```text
WHISPER_MODEL=medium
```

A larger model generally needs more RAM/compute.

---

## 11. Future upgrades

This project is intentionally structured so these features can be added later:

- True real-time streaming transcription
- Speaker diarization: Speaker 1 / Speaker 2
- SRT subtitle generation
- VTT subtitle generation
- Word-level timestamps
- Translation
- Hindi/Marathi/English mixed-language handling
- Noise reduction
- Voice activity detection
- Search inside transcript
- AI transcript summarization
- Meeting minutes generation
- RAG over transcripts
- FastAPI backend + React frontend
- Database for storing transcription history
- Authentication
- Docker deployment
