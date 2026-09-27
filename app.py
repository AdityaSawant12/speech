import os
import tempfile
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="Speech to Text Studio",
    page_icon="🎙️",
    layout="wide",
)

st.title("🎙️ Speech to Text Studio")
st.caption("Transcribe audio/video files or convert live microphone speech to text.")

# Lazy imports so the UI can show useful setup errors.
try:
    from faster_whisper import WhisperModel
except Exception:
    WhisperModel = None

try:
    from streamlit_mic_recorder import mic_recorder
except Exception:
    mic_recorder = None

MODEL_SIZE = os.getenv("WHISPER_MODEL", "small")
DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")


@st.cache_resource
def load_model():
    if WhisperModel is None:
        raise RuntimeError(
            "faster-whisper is not installed. Run: pip install -r requirements.txt"
        )
    return WhisperModel(
        MODEL_SIZE,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
    )


def transcribe_file(file_path: str, language: str | None):
    model = load_model()
    segments, info = model.transcribe(
        file_path,
        language=language if language else None,
        beam_size=5,
        vad_filter=True,
    )

    rows = []
    text_parts = []

    for segment in segments:
        text = segment.text.strip()
        if text:
            rows.append(
                {
                    "start": round(segment.start, 2),
                    "end": round(segment.end, 2),
                    "text": text,
                }
            )
            text_parts.append(text)

    return "\n".join(text_parts), rows, info


def save_uploaded_file(uploaded_file):
    suffix = Path(uploaded_file.name).suffix or ".tmp"
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path


# ─────────────────────────────────────────────
# NLP helpers
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _load_spacy():
    """Load spaCy model once; auto-download en_core_web_sm if missing."""
    try:
        import spacy
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            from spacy.cli import download as spacy_download
            spacy_download("en_core_web_sm")
            return spacy.load("en_core_web_sm")
    except Exception:
        return None


def _sentiment(text: str) -> dict:
    """Return polarity, subjectivity and a label via TextBlob."""
    try:
        from textblob import TextBlob
        blob = TextBlob(text)
        pol = blob.sentiment.polarity        # -1 ... +1
        sub = blob.sentiment.subjectivity    #  0 ... 1
        if pol > 0.05:
            label, emoji, color = "Positive", "\U0001f60a", "#2ecc71"
        elif pol < -0.05:
            label, emoji, color = "Negative", "\U0001f61f", "#e74c3c"
        else:
            label, emoji, color = "Neutral", "\U0001f610", "#f39c12"
        return {"polarity": pol, "subjectivity": sub,
                "label": label, "emoji": emoji, "color": color}
    except Exception as exc:
        return {"error": str(exc)}


def _spacy_label_desc(label: str) -> str:
    """Human-readable description for common spaCy NER labels."""
    _map = {
        "PERSON": "Person", "ORG": "Organisation", "GPE": "Geo-political entity",
        "LOC": "Location", "DATE": "Date / Time", "TIME": "Time",
        "MONEY": "Monetary value", "PERCENT": "Percentage", "PRODUCT": "Product",
        "EVENT": "Event", "WORK_OF_ART": "Work of art", "LAW": "Law / regulation",
        "LANGUAGE": "Language", "NORP": "Nationality / religion / political group",
        "FAC": "Facility", "CARDINAL": "Cardinal number", "ORDINAL": "Ordinal number",
        "QUANTITY": "Quantity",
    }
    return _map.get(label, label)


def _ner(text: str) -> list:
    """Return deduplicated list of entity dicts using spaCy."""
    nlp = _load_spacy()
    if nlp is None:
        return []
    doc = nlp(text)
    seen = set()
    entities = []
    for ent in doc.ents:
        key = (ent.text.strip(), ent.label_)
        if key not in seen:
            seen.add(key)
            entities.append({
                "Entity": ent.text.strip(),
                "Type": ent.label_,
                "Description": _spacy_label_desc(ent.label_),
            })
    return entities


def _keywords(text: str, n: int = 10) -> list:
    """Return top-n (keyword, relevance 0-100) pairs using YAKE."""
    try:
        import yake
        kw_extractor = yake.KeywordExtractor(
            lan="en", n=2, dedupLim=0.7, top=n, features=None
        )
        keywords = kw_extractor.extract_keywords(text)
        if not keywords:
            return []
        max_score = max(s for _, s in keywords) or 1
        return [(kw, round((1 - s / max_score) * 100, 1)) for kw, s in keywords]
    except Exception as exc:
        return [(f"Error: {exc}", 0.0)]


# ─────────────────────────────────────────────
# NLP panel rendered after every transcription
# ─────────────────────────────────────────────

def nlp_panel(text: str, key_prefix: str = ""):
    """Render Sentiment / NER / Keywords analysis tabs for a transcript."""
    if not text or not text.strip():
        return

    st.markdown("---")
    st.subheader("\U0001f9e0 NLP Analysis")

    nlp_t1, nlp_t2, nlp_t3 = st.tabs(
        ["\U0001f4ac Sentiment", "\U0001f3f7\ufe0f Named Entities", "\U0001f511 Keywords"]
    )

    # ── Sentiment ─────────────────────────────
    with nlp_t1:
        with st.spinner("Analysing sentiment..."):
            result = _sentiment(text)

        if "error" in result:
            st.warning(f"Sentiment analysis unavailable: {result['error']}")
            st.info("Run `pip install textblob` and restart Streamlit.")
        else:
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Overall Sentiment",
                          f"{result['emoji']} {result['label']}")
            with col_b:
                st.metric("Polarity Score",
                          f"{result['polarity']:+.3f}",
                          help="Range: -1 (most negative) to +1 (most positive)")
            with col_c:
                st.metric("Subjectivity",
                          f"{result['subjectivity']:.3f}",
                          help="0 = fully objective, 1 = fully subjective")

            st.markdown("**Polarity gauge** (Negative ← | → Positive)")
            filled = int((result["polarity"] + 1) / 2 * 20)
            bar = "\u2588" * filled + "\u2591" * (20 - filled)
            _c = result["color"]
            _lb = result["label"]
            st.markdown(
                f"<p style='font-family:monospace;font-size:1.2rem;color:{_c}'>"
                + bar
                + f" {_lb}</p>",
                unsafe_allow_html=True,
            )
            st.progress((result["polarity"] + 1) / 2)

    # ── NER ───────────────────────────────────
    with nlp_t2:
        with st.spinner("Extracting named entities..."):
            entities = _ner(text)

        if not entities:
            st.info(
                "No named entities found, or spaCy model is unavailable.\n\n"
                "Install with:\n```\npip install spacy\n"
                "python -m spacy download en_core_web_sm\n```"
            )
        else:
            import pandas as pd
            df_ent = pd.DataFrame(entities)
            st.markdown(f"**{len(entities)} unique entities found**")
            st.dataframe(df_ent, use_container_width=True, hide_index=True)

            type_counts = df_ent["Type"].value_counts().reset_index()
            type_counts.columns = ["Entity Type", "Count"]
            with st.expander("Entity type summary"):
                st.dataframe(type_counts, use_container_width=True, hide_index=True)

    # ── Keywords ──────────────────────────────
    with nlp_t3:
        n_kw = st.slider("Number of keywords", 5, 20, 10,
                         key=f"{key_prefix}_kw_slider")
        with st.spinner("Extracting keywords..."):
            keywords = _keywords(text, n=n_kw)

        if not keywords or str(keywords[0][0]).startswith("Error"):
            st.info(
                "Keyword extraction unavailable.\n\n"
                "Install YAKE with:\n```\npip install yake\n```"
            )
        else:
            st.markdown(f"**Top {len(keywords)} keywords / phrases (YAKE)**")
            for rank, (kw, score) in enumerate(keywords, 1):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"`{rank}.` **{kw}**")
                    st.progress(score / 100)
                with col2:
                    st.markdown(
                        f"<div style='text-align:right;padding-top:6px'>"
                        f"<b>{score:.1f}</b>/100</div>",
                        unsafe_allow_html=True,
                    )

            import pandas as pd
            kw_df = pd.DataFrame(keywords, columns=["Keyword", "Relevance (0-100)"])
            kw_df.index += 1
            with st.expander("Full keyword table"):
                st.dataframe(kw_df, use_container_width=True)


tab1, tab2 = st.tabs(["📁 Audio / Video File", "🎤 Live Microphone"])

with tab1:
    st.subheader("Transcribe an audio or video file")

    uploaded = st.file_uploader(
        "Upload a file",
        type=[
            "mp3", "wav", "m4a", "aac", "flac", "ogg", "opus",
            "mp4", "mov", "mkv", "avi", "webm", "mpeg", "mpg"
        ],
        help="Video files are supported because Whisper extracts and transcribes their audio.",
    )

    language_options = {
        "Auto Detect": None,
        "English": "en",
        "Hindi": "hi",
        "Marathi": "mr",
        "Tamil": "ta",
        "Telugu": "te",
        "Kannada": "kn",
        "Bengali": "bn",
        "Gujarati": "gu",
        "Spanish": "es",
        "French": "fr",
        "German": "de",
    }

    language_name = st.selectbox("Language", list(language_options.keys()))
    language = language_options[language_name]

    if uploaded:
        st.audio(uploaded)

        if st.button("🚀 Transcribe File", type="primary"):
            temp_path = None
            try:
                with st.spinner("Transcribing..."):
                    temp_path = save_uploaded_file(uploaded)
                    text, rows, info = transcribe_file(temp_path, language)

                st.success("Transcription completed.")

                detected = getattr(info, "language", "unknown")
                probability = getattr(info, "language_probability", None)
                if probability is not None:
                    st.info(f"Detected language: **{detected}** ({probability:.1%} confidence)")
                else:
                    st.info(f"Detected language: **{detected}**")

                st.subheader("Transcript")
                st.text_area("Text", text, height=350)

                txt_bytes = text.encode("utf-8")
                st.download_button(
                    "⬇️ Download TXT",
                    data=txt_bytes,
                    file_name=f"{Path(uploaded.name).stem}_transcript.txt",
                    mime="text/plain",
                )

                if rows:
                    import pandas as pd
                    df = pd.DataFrame(rows)
                    csv_bytes = df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇️ Download timestamped CSV",
                        data=csv_bytes,
                        file_name=f"{Path(uploaded.name).stem}_timestamps.csv",
                        mime="text/csv",
                    )

                    with st.expander("Timestamped segments"):
                        st.dataframe(df, use_container_width=True)

                # ── NLP panel ──
                nlp_panel(text, key_prefix="file")

            except Exception as e:
                st.error(f"Transcription failed: {e}")
                st.exception(e)
            finally:
                if temp_path:
                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass

with tab2:
    st.subheader("Real-time microphone speech-to-text")
    st.write(
        "Click the microphone button, speak, stop recording, and the recorded speech "
        "will be transcribed locally using Whisper."
    )

    live_language_name = st.selectbox(
        "Microphone language",
        list(language_options.keys()),
        key="live_language",
    )
    live_language = language_options[live_language_name]

    if mic_recorder is None:
        st.warning(
            "Microphone component is not installed. Run: "
            "`pip install -r requirements.txt` and restart Streamlit."
        )
    else:
        audio = mic_recorder(
            start_prompt="🎙️ Start recording",
            stop_prompt="⏹️ Stop recording",
            just_once=False,
            use_container_width=True,
            format="wav",
            key="mic_recorder",
        )

        if audio and audio.get("bytes"):
            st.audio(audio["bytes"], format="audio/wav")

            if st.button("📝 Convert Recording to Text", type="primary"):
                temp_path = None
                try:
                    fd, temp_path = tempfile.mkstemp(suffix=".wav")
                    os.close(fd)
                    with open(temp_path, "wb") as f:
                        f.write(audio["bytes"])

                    with st.spinner("Converting microphone recording..."):
                        text, rows, info = transcribe_file(temp_path, live_language)

                    st.success("Microphone transcription completed.")
                    st.text_area("Live transcription result", text, height=250)

                    st.download_button(
                        "⬇️ Download microphone transcript",
                        data=text.encode("utf-8"),
                        file_name="microphone_transcript.txt",
                        mime="text/plain",
                    )

                    # ── NLP panel ──
                    nlp_panel(text, key_prefix="mic")

                except Exception as e:
                    st.error(f"Microphone transcription failed: {e}")
                    st.exception(e)
                finally:
                    if temp_path:
                        try:
                            os.remove(temp_path)
                        except OSError:
                            pass

st.divider()

with st.expander("⚙️ Current configuration"):
    st.write(f"Whisper model: `{MODEL_SIZE}`")
    st.write(f"Device: `{DEVICE}`")
    st.write(f"Compute type: `{COMPUTE_TYPE}`")
    st.write(
        "For CPU, `small + int8` is a practical default. If you have a compatible "
        "NVIDIA GPU, you can use `WHISPER_DEVICE=cuda` and `WHISPER_COMPUTE_TYPE=float16`."
    )
