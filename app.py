import os
import json
import math
import tempfile
import subprocess
import shutil
from pathlib import Path

import streamlit as st
import torch
import torchaudio
from google import genai
from pyannote.audio import Pipeline
from speechbrain.inference.classifiers import EncoderClassifier


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Neon Meeting AI",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = Path(__file__).resolve().parent
PROFILE_DIR = APP_DIR / "speaker_profiles"
PROFILE_DIR.mkdir(parents=True, exist_ok=True)

PROFILE_DB = PROFILE_DIR / "profiles.json"

EMBEDDING_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"

# A conservative starting point. Tune this with your own recordings.
MATCH_THRESHOLD = 0.55


# ============================================================
# PREMIUM GLASS UI
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at 15% 15%, rgba(70, 80, 120, .32), transparent 30%),
            radial-gradient(circle at 85% 75%, rgba(20, 130, 150, .22), transparent 30%),
            #080b12;
        color: #f5f7fb;
    }

    [data-testid="stSidebar"] {
        background: rgba(255,255,255,.045);
        border-right: 1px solid rgba(255,255,255,.10);
    }

    .glass {
        padding: 24px;
        border-radius: 22px;
        background: rgba(255,255,255,.065);
        border: 1px solid rgba(255,255,255,.12);
        backdrop-filter: blur(18px);
        margin-bottom: 18px;
    }

    .hero {
        padding: 32px;
        border-radius: 28px;
        background: linear-gradient(
            135deg,
            rgba(255,255,255,.10),
            rgba(255,255,255,.035)
        );
        border: 1px solid rgba(255,255,255,.14);
        box-shadow: 0 18px 55px rgba(0,0,0,.28);
        margin-bottom: 24px;
    }

    .hero-title {
        font-size: 42px;
        font-weight: 850;
        letter-spacing: -1.5px;
    }

    .hero-subtitle {
        color: #aeb8ca;
        font-size: 16px;
    }

    .profile-card {
        padding: 18px;
        border-radius: 18px;
        background: rgba(255,255,255,.055);
        border: 1px solid rgba(255,255,255,.10);
        margin-bottom: 12px;
    }

    div.stButton > button {
        border-radius: 14px;
        min-height: 44px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SECRETS / CLIENTS
# ============================================================

def get_secret(name: str):
    value = os.environ.get(name)
    if value:
        return value
    try:
        return st.secrets[name]
    except Exception:
        return None


GEMINI_API_KEY = get_secret("GEMINI_API_KEY")
HF_TOKEN = get_secret("HF_TOKEN")
AUTH_USERNAME = get_secret("AUTH_USERNAME") or "admin"
AUTH_PASSWORD = get_secret("AUTH_PASSWORD") or "admin123"


# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "authenticated": False,
    "page": 1,
    "meeting_data": None,
    "audio_name": None,
    "speaker_results": [],
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# LOGIN
# ============================================================

if not st.session_state.authenticated:
    st.write("")
    st.write("")
    st.write("")

    st.markdown(
        '<div class="hero">'
        '<div class="hero-title">💠 NEON MEETING AI</div>'
        '<div class="hero-subtitle">AI MEETING TO ACTION INTELLIGENCE</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.subheader("🔐 SIGN IN")
        st.caption("Enter your credentials to access the meeting intelligence system.")

        username = st.text_input(
            "USERNAME",
            placeholder="Enter username",
            key="login_username",
        )

        password = st.text_input(
            "PASSWORD",
            type="password",
            placeholder="Enter password",
            key="login_password",
        )

        if st.button("🔓 LOGIN", type="primary", use_container_width=True):
            if username == AUTH_USERNAME and password == AUTH_PASSWORD:
                st.session_state.authenticated = True
                st.session_state.page = 1
                st.rerun()
            else:
                st.error("❌ Invalid username or password.")

    st.caption("🔒 Secure access • Speaker Recognition • Meeting Intelligence")
    st.stop()


# Optional runtime check: FFmpeg is supplied by packages.txt on Streamlit Cloud.
# No UI element is added here, so the existing design remains unchanged.
FFMPEG_PATH = shutil.which("ffmpeg")

# ============================================================
# PROFILE DATABASE
# ============================================================

def load_profiles():
    if not PROFILE_DB.exists():
        return {}

    try:
        data = json.loads(PROFILE_DB.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_profiles(profiles):
    PROFILE_DB.write_text(
        json.dumps(profiles, indent=2),
        encoding="utf-8",
    )


def delete_profile(name):
    profiles = load_profiles()
    profiles.pop(name, None)
    save_profiles(profiles)


# ============================================================
# ML LOADERS
# ============================================================

@st.cache_resource(show_spinner=False)
def load_speaker_encoder():
    return EncoderClassifier.from_hparams(
        source=EMBEDDING_MODEL,
        savedir=str(APP_DIR / "models" / "ecapa"),
        run_opts={"device": "cpu"},
    )


@st.cache_resource(show_spinner=False)
def load_diarization_pipeline(hf_token):
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN is missing. Add a Hugging Face access token "
            "and accept the pyannote Community-1 model conditions."
        )

    return Pipeline.from_pretrained(
        DIARIZATION_MODEL,
        token=hf_token,
    )


# ============================================================
# AUDIO HELPERS
# ============================================================

def save_uploaded_audio(uploaded, suffix=".wav"):
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    ) as f:
        f.write(uploaded.getbuffer())
        return f.name


def convert_to_wav(input_path):
    """
    Convert any supported meeting audio/video format to mono 16 kHz WAV.

    FFmpeg is installed as a system package on Streamlit Cloud through
    packages.txt, so this function calls the ffmpeg executable directly.
    """
    output_path = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".wav",
    ).name

    # Find the FFmpeg executable installed by packages.txt.
    ffmpeg_path = shutil.which("ffmpeg")

    if not ffmpeg_path:
        raise RuntimeError(
            "FFmpeg is not available. Make sure packages.txt contains "
            "'ffmpeg' and reboot/redeploy the Streamlit app."
        )

    command = [
        ffmpeg_path,
        "-y",
        "-i",
        input_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        output_path,
    ]

    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        error_detail = (exc.stderr or "").strip()
        raise RuntimeError(
            "FFmpeg could not convert the audio. "
            f"Details: {error_detail[-1000:]}"
        ) from exc

    return output_path


def load_16k_mono(path):
    waveform, sample_rate = torchaudio.load(path)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(
            waveform,
            sample_rate,
            16000,
        )

    return waveform


def duration_seconds(waveform):
    return waveform.shape[-1] / 16000.0


def embedding_from_waveform(waveform, encoder):
    with torch.no_grad():
        embedding = encoder.encode_batch(
            waveform,
            normalize=True,
        )

    embedding = embedding.squeeze().detach().cpu()
    embedding = embedding / (
        torch.linalg.vector_norm(embedding) + 1e-8
    )

    return embedding


def cosine_similarity(a, b):
    a = torch.tensor(a, dtype=torch.float32)
    b = torch.tensor(b, dtype=torch.float32)

    a = a / (torch.linalg.vector_norm(a) + 1e-8)
    b = b / (torch.linalg.vector_norm(b) + 1e-8)

    return float(torch.dot(a, b))


def match_embedding(embedding, profiles):
    best_name = "Unknown"
    best_score = -1.0

    for name, profile in profiles.items():
        stored = profile.get("embedding")
        if not stored:
            continue

        score = cosine_similarity(
            embedding,
            stored,
        )

        if score > best_score:
            best_score = score
            best_name = name

    if best_score < MATCH_THRESHOLD:
        return "Unknown", best_score

    return best_name, best_score


def make_speaker_audio(waveform, segments, max_seconds=20.0):
    """
    Collect several diarized segments for one speaker.
    We intentionally cap the total audio used for matching.
    """
    chunks = []
    total = 0.0

    for start, end in segments:
        start_i = max(0, int(start * 16000))
        end_i = min(
            waveform.shape[-1],
            int(end * 16000),
        )

        if end_i <= start_i:
            continue

        chunk = waveform[:, start_i:end_i]
        length = chunk.shape[-1] / 16000.0

        remaining = max_seconds - total

        if remaining <= 0:
            break

        if length > remaining:
            chunk = chunk[:, : int(remaining * 16000)]
            length = chunk.shape[-1] / 16000.0

        if length >= 0.8:
            chunks.append(chunk)
            total += length

    if not chunks:
        return None

    return torch.cat(chunks, dim=-1)


def play_recorded_audio(recording):
    """
    Display the microphone recording using Streamlit's native
    audio player and return the original recording object.
    """
    if recording is None:
        return None

    try:
        audio_bytes = recording.getvalue()

        if not audio_bytes:
            st.error("❌ No audio data was received from the microphone.")
            return None

        st.success("🎙️ Recording captured successfully.")

        # Use Streamlit's native audio player instead of manually
        # constructing a Base64 HTML audio element.
        st.audio(recording, format="audio/wav")

        st.caption(
            f"🎧 Recording size: {len(audio_bytes) / 1024:.1f} KB"
        )

        return recording

    except Exception as exc:
        st.error(f"❌ Audio playback error: {exc}")
        return None


# ============================================================
# SIDEBAR
# ============================================================

profiles = load_profiles()

with st.sidebar:
    st.markdown("## 💠 NEON MEETING AI")
    st.caption("AI Meeting to Action Intelligence")

    st.divider()

    if st.button(
        "🏠 HOME",
        use_container_width=True,
    ):
        st.session_state.page = 1
        st.rerun()

    if st.button(
        "🎙️ VOICE PROFILES",
        use_container_width=True,
    ):
        st.session_state.page = 2
        st.rerun()

    if st.button(
        "🎧 ANALYZE MEETING",
        use_container_width=True,
    ):
        st.session_state.page = 3
        st.rerun()

    st.divider()

    st.write("### 👤 SESSION")
    st.success("Logged in")

    st.metric(
        "Registered Speakers",
        len(profiles),
    )

    if st.button("🚪 LOGOUT", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.meeting_data = None
        st.session_state.speaker_results = []
        st.session_state.page = 1
        st.rerun()


# ============================================================
# PAGE 1 — HOME
# ============================================================

if st.session_state.page == 1:

    st.markdown(
        '<div class="hero">'
        '<div class="hero-title">💠 NEON MEETING AI</div>'
        '<div class="hero-subtitle">'
        'AI MEETING TO ACTION INTELLIGENCE'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="glass">'
        '<h3>⚡ CONVERSATION → INTELLIGENCE</h3>'
        '<p>Turn meetings into identified speakers, tasks, promises, deadlines and decisions.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("🎙️ SPEAKER ID", "ON")

    with c2:
        st.metric("🧠 AI ANALYSIS", "ON")

    with c3:
        st.metric("🔗 COMMITMENTS", "ON")

    with c4:
        st.metric("◈ DEADLINES", "ON")

    st.write("")

    if st.button(
        "🎙️ REGISTER VOICE",
        type="primary",
        use_container_width=True,
    ):
        st.session_state.page = 2
        st.rerun()

    if st.button(
        "🎧 ANALYZE A MEETING",
        use_container_width=True,
    ):
        st.session_state.page = 3
        st.rerun()


# ============================================================
# PAGE 2 — VOICE REGISTRATION
# ============================================================

elif st.session_state.page == 2:

    st.title("🎙️ VOICE PROFILE")

    st.caption(
        "Register a speaker once. The app stores the speaker embedding, "
        "not the raw registration audio."
    )

    if not HF_TOKEN:
        st.warning(
            "HF_TOKEN is required for the speaker model. "
            "Add it to Streamlit Secrets before registering voices."
        )

    st.markdown(
        '<div class="glass">',
        unsafe_allow_html=True,
    )

    name = st.text_input(
        "SPEAKER NAME",
        placeholder="Example: Venkatesh",
    )

    st.write("### 🎤 Record voice")

    recording = st.audio_input(
        "Speak naturally for about 10–20 seconds",
        sample_rate=16000,
        key="voice_registration_recording",
    )

    recording_object = play_recorded_audio(recording)

    if recording_object:
        if st.button(
            "💾 REGISTER SPEAKER",
            type="primary",
            use_container_width=True,
        ):
            temp_path = None

            try:
                if not HF_TOKEN:
                    raise RuntimeError(
                        "HF_TOKEN is not configured. "
                        "Add it to Streamlit Secrets."
                    )

                clean_name = name.strip()

                if len(clean_name) < 2:
                    raise RuntimeError(
                        "Please enter a valid speaker name."
                    )

                temp_path = save_uploaded_audio(
                    recording_object,
                    ".wav",
                )

                waveform = load_16k_mono(temp_path)
                seconds = duration_seconds(waveform)

                if seconds < 5:
                    raise RuntimeError(
                        "Please record at least 5 seconds of clear speech."
                    )

                with st.spinner(
                    "Loading speaker model and creating voice profile..."
                ):
                    encoder = load_speaker_encoder()
                    embedding = embedding_from_waveform(
                        waveform,
                        encoder,
                    )

                profiles = load_profiles()

                profiles[clean_name] = {
                    "embedding": embedding.tolist(),
                    "created_at": __import__("datetime").datetime.now().isoformat(
                        timespec="seconds"
                    ),
                    "model": EMBEDDING_MODEL,
                }

                save_profiles(profiles)

                st.success(
                    f"✅ {clean_name} voice profile registered."
                )

                st.info(
                    "Only the voice embedding is stored by this app. "
                    "You can re-register the speaker later to replace the profile."
                )

                st.rerun()

            except Exception as exc:
                st.error(f"❌ Registration failed: {exc}")

            finally:
                if temp_path:
                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )

    st.subheader("👥 REGISTERED SPEAKERS")

    profiles = load_profiles()

    if not profiles:
        st.info("No speaker profiles registered yet.")

    else:
        for speaker_name, profile in profiles.items():

            with st.container(border=True):
                col1, col2, col3 = st.columns([3, 1, 1])

                with col1:
                    st.write(f"### 🎙️ {speaker_name}")
                    st.caption(
                        f"Registered: {profile.get('created_at', 'Unknown')}"
                    )

                with col2:
                    st.success("ACTIVE")

                with col3:
                    if st.button(
                        "🗑️ DELETE",
                        key=f"delete_{speaker_name}",
                    ):
                        delete_profile(speaker_name)
                        st.rerun()

    st.divider()

    st.warning(
        "Voice profiles are biometric information. Get the speaker's "
        "consent before registering and storing a voice profile."
    )


# ============================================================
# PAGE 3 — MEETING ANALYSIS
# ============================================================

elif st.session_state.page == 3:

    st.title("🎧 MEETING ANALYSIS")

    profiles = load_profiles()

    if not profiles:
        st.warning(
            "No voice profiles exist yet. Register at least one speaker first."
        )

        if st.button(
            "🎙️ GO TO VOICE REGISTRATION",
            type="primary",
            use_container_width=True,
        ):
            st.session_state.page = 2
            st.rerun()

        st.stop()

    st.caption(
        f"{len(profiles)} registered speaker profile(s) available for matching."
    )

    audio = st.file_uploader(
        "UPLOAD MEETING RECORDING",
        type=["mp3", "wav", "m4a", "mp4", "webm"],
    )

    if audio:
        st.audio(audio)

    if audio and st.button(
        "🧠 ANALYZE MEETING",
        type="primary",
        use_container_width=True,
    ):

        temp_input = None
        temp_wav = None

        try:
            if not HF_TOKEN:
                raise RuntimeError(
                    "HF_TOKEN is missing. Add it to Streamlit Secrets."
                )

            if not GEMINI_API_KEY:
                raise RuntimeError(
                    "GEMINI_API_KEY is missing."
                )

            extension = Path(audio.name).suffix.lower() or ".mp3"

            temp_input = save_uploaded_audio(
                audio,
                extension,
            )

            with st.status(
                "Running speaker recognition and meeting intelligence...",
                expanded=True,
            ) as status:

                st.write("1/5 Preparing meeting audio...")
                temp_wav = convert_to_wav(temp_input)

                st.write("2/5 Detecting speaker turns...")
                diarizer = load_diarization_pipeline(HF_TOKEN)
                diarization_output = diarizer(temp_wav)

                annotation = diarization_output.speaker_diarization

                speaker_segments = {}

                for turn, _, speaker_label in annotation.itertracks(
                    yield_label=True
                ):
                    speaker_segments.setdefault(
                        speaker_label,
                        [],
                    ).append(
                        (turn.start, turn.end)
                    )

                st.write(
                    f"Detected {len(speaker_segments)} distinct speaker(s)."
                )

                st.write("3/5 Matching speakers to registered profiles...")

                meeting_waveform = load_16k_mono(temp_wav)
                encoder = load_speaker_encoder()

                results = []
                speaker_transcript_context = []

                for speaker_label, segments in speaker_segments.items():

                    speaker_audio = make_speaker_audio(
                        meeting_waveform,
                        segments,
                        max_seconds=20.0,
                    )

                    if speaker_audio is None:
                        name = "Unknown"
                        score = -1.0
                    else:
                        embedding = embedding_from_waveform(
                            speaker_audio,
                            encoder,
                        )

                        name, score = match_embedding(
                            embedding.tolist(),
                            profiles,
                        )

                    results.append(
                        {
                            "speaker_label": speaker_label,
                            "name": name,
                            "score": score,
                            "segments": len(segments),
                        }
                    )

                    speaker_transcript_context.append(
                        f"{speaker_label} = {name}"
                    )

                st.session_state.speaker_results = results

                st.write("4/5 Sending meeting audio to Gemini...")

                client = genai.Client(
                    api_key=GEMINI_API_KEY
                )

                uploaded_file = client.files.upload(
                    file=temp_wav
                )

                mapping_text = "\n".join(
                    speaker_transcript_context
                )

                prompt = f"""
You are an AI Meeting to Action Intelligence Agent.

Analyze the uploaded meeting audio.

A separate speaker-recognition system has produced this
speaker mapping:

{mapping_text}

Use these names when the mapping has a confident match.
If a speaker is mapped to Unknown, use Unknown.
Do not invent a person's identity.

Extract:

1. Concise meeting summary
2. Tasks
3. Promises / commitments
4. Deadlines
5. Important decisions

For every task or promise, identify the speaker when the
audio provides enough evidence.

Return ONLY valid JSON.

Use exactly:

{{
  "summary": "Short meeting summary",
  "commitments": [
    {{
      "speaker": "Venkatesh",
      "task": "Complete the backend",
      "deadline": "Tomorrow",
      "type": "Task"
    }}
  ],
  "decisions": [
    "Decision made"
  ]
}}

Rules:
- Do not invent information.
- Use "Unknown" when identity is unavailable.
- Use "Not specified" when no deadline is mentioned.
- Keep the summary concise.
- Return valid JSON only.
"""

                interaction = client.interactions.create(
                    model="gemini-3.6-flash",
                    input=[
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {
                            "type": "audio",
                            "uri": uploaded_file.uri,
                            "mime_type": uploaded_file.mime_type,
                        },
                    ],
                )

                result_text = interaction.output_text.strip()

                if result_text.startswith("```json"):
                    result_text = result_text[7:].strip()

                if result_text.startswith("```"):
                    result_text = result_text[3:].strip()

                if result_text.endswith("```"):
                    result_text = result_text[:-3].strip()

                meeting_data = json.loads(result_text)

                meeting_data.setdefault(
                    "summary",
                    "No summary available.",
                )
                meeting_data.setdefault(
                    "commitments",
                    [],
                )
                meeting_data.setdefault(
                    "decisions",
                    [],
                )

                meeting_data["speaker_results"] = results

                st.session_state.meeting_data = meeting_data
                st.session_state.audio_name = audio.name

                st.write("5/5 Finalizing intelligence report...")
                status.update(
                    label="✅ Meeting analysis complete",
                    state="complete",
                )

            st.rerun()

        except json.JSONDecodeError:
            st.error(
                "Gemini returned invalid JSON. Please try the meeting again."
            )

        except Exception as exc:
            st.error(
                f"❌ Analysis failed: {exc}"
            )

        finally:
            for path in [temp_input, temp_wav]:
                if path:
                    try:
                        os.remove(path)
                    except OSError:
                        pass


# ============================================================
# RESULTS
# ============================================================

if st.session_state.meeting_data:

    data = st.session_state.meeting_data

    st.divider()
    st.title("📊 MEETING INTELLIGENCE")

    speaker_results = data.get(
        "speaker_results",
        st.session_state.speaker_results,
    )

    st.subheader("🎙️ SPEAKER RECOGNITION")

    st.metric(
        "TOTAL SPEAKERS",
        len(speaker_results),
    )

    for item in speaker_results:

        score = item.get("score", -1)

        if score >= 0:
            score_text = f"{score:.3f}"
        else:
            score_text = "N/A"

        if item["name"] == "Unknown":
            st.warning(
                f"🎤 {item['speaker_label']} → **Unknown** "
                f"(similarity: {score_text})"
            )
        else:
            st.success(
                f"🎤 {item['speaker_label']} → **{item['name']}** "
                f"(similarity: {score_text})"
            )

    st.subheader("📝 SUMMARY")

    with st.container(border=True):
        st.write(
            data.get(
                "summary",
                "No summary available.",
            )
        )

    commitments = data.get(
        "commitments",
        [],
    )

    st.subheader("✅ TASKS & PROMISES")

    if commitments:

        for index, item in enumerate(
            commitments,
            start=1,
        ):

            with st.container(border=True):

                st.write(
                    f"### {index}. "
                    f"{item.get('type', 'Commitment')}"
                )

                st.write(
                    f"👤 **Speaker:** "
                    f"{item.get('speaker', 'Unknown')}"
                )

                st.write(
                    f"📌 **Action:** "
                    f"{item.get('task', 'Not specified')}"
                )

                st.write(
                    f"◈ **Deadline:** "
                    f"{item.get('deadline', 'Not specified')}"
                )

    else:
        st.info("No tasks or promises detected.")

    st.subheader("💡 DECISIONS")

    decisions = data.get(
        "decisions",
        [],
    )

    if decisions:
        for decision in decisions:
            st.write(f"• {decision}")
    else:
        st.info("No important decisions detected.")

    st.subheader("📥 EXPORT")

    report = [
        "NEON MEETING AI",
        "AI MEETING TO ACTION INTELLIGENCE",
        "=" * 55,
        "",
        "SPEAKERS",
        "-" * 30,
    ]

    for item in speaker_results:
        score = item.get("score", -1)
        report.append(
            f"{item['speaker_label']} -> "
            f"{item['name']} "
            f"(similarity: {score:.3f})"
            if score >= 0
            else
            f"{item['speaker_label']} -> Unknown"
        )

    report.extend(
        [
            "",
            "SUMMARY",
            "-" * 30,
            data.get("summary", ""),
            "",
            "TASKS / PROMISES",
            "-" * 30,
        ]
    )

    for item in commitments:
        report.append(
            f"{item.get('speaker', 'Unknown')}: "
            f"{item.get('task', 'Not specified')} | "
            f"{item.get('deadline', 'Not specified')} | "
            f"{item.get('type', 'Commitment')}"
        )

    report.extend(
        [
            "",
            "DECISIONS",
            "-" * 30,
        ]
    )

    report.extend(
        [str(x) for x in decisions]
    )

    st.download_button(
        "⬇️ DOWNLOAD INTELLIGENCE REPORT",
        data="\n".join(report),
        file_name="neon_meeting_intelligence.txt",
        mime="text/plain",
        use_container_width=True,
    )
