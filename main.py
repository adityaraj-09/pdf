import streamlit as st
import librosa
import json
import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path

# Streamlit UI
st.title("Few Shot Language Agnostic Keyword Spotting (FSLAKWS)")

# Step 1: Upload Five Audio Files
st.header("Step 1: Upload Five Audio Files")
uploaded_files = st.file_uploader("Upload 5 audio files", type=["wav", "mp3", "flac"], accept_multiple_files=True)

if len(uploaded_files) == 5:
    st.success("All 5 audio files uploaded successfully!")
    
    # Load and display waveforms of all 5 files
    for idx, audio_file in enumerate(uploaded_files):
        st.write(f"**Audio File {idx + 1}: {audio_file.name}**")
        audio_data, sr = librosa.load(audio_file, sr=None)
        st.audio(audio_file, format="audio/wav")
        
        # Plot waveform
        fig, ax = plt.subplots()
        ax.plot(np.linspace(0, len(audio_data) / sr, len(audio_data)), audio_data, color='blue')
        ax.set_title(f"Waveform of {audio_file.name}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.grid(True)
        st.pyplot(fig)

    # Step 2: Upload Query Audio File
    st.header("Step 2: Upload Query Audio File")
    query_audio = st.file_uploader("Upload a query audio file", type=["wav", "mp3", "flac"])

    if query_audio:
        st.success("Query audio uploaded successfully!")
        # Load and display query audio waveform
        query_audio_data, sr = librosa.load(query_audio, sr=None)
        st.audio(query_audio, format="audio/wav")
        
        # Plot query audio waveform
        fig, ax = plt.subplots()
        ax.plot(np.linspace(0, len(query_audio_data) / sr, len(query_audio_data)), query_audio_data, color='blue')
        ax.set_title(f"Waveform of {query_audio.name}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.grid(True)
        st.pyplot(fig)

        # Step 3: Simulate Processing
        st.header("Step 3: Processing Files")
        with st.spinner("Processing audio files..."):
            time.sleep(2)  # Simulate processing delay
        st.success("Processing complete!")

        # Step 4: Read JSON File
        st.header("Step 4: Keyword Occurrences")
        json_path = Path("./test.json")
        if json_path.exists():
            with open(json_path, "r") as f:
                json_data = json.load(f)
            detections = json_data.get("detections", {})
            st.success(f"Loaded data from {json_path.name}")
        else:
            st.error(f"JSON file {json_path.name} not found in the folder!")
            st.stop()

        # Step 5: Visualize Keyword Occurrences
        st.header("Step 5: Visualize Keyword Occurrences")
        audio_duration = librosa.get_duration(y=query_audio_data, sr=sr)  # Get query audio duration

        for keyword, data in detections.items():
            st.subheader(f"Keyword: {keyword}")
            st.write(f"Total Count: {data['count']}")

            # Prepare a number line for each keyword occurrence
            time_range = np.linspace(0, 35, 1000)
            timeline = np.zeros_like(time_range)

            for occ in data["occurrences"]:
                start_idx = np.where(time_range >= occ["start_time_sec"])[0][0]
                end_idx = np.where(time_range >= occ["end_time_sec"])[0][0]
                timeline[start_idx:end_idx] = 1  # Mark the presence of keyword

            # Plotting occurrences
            fig, ax = plt.subplots(figsize=(12, 2))
            ax.plot(time_range, timeline, color='green', lw=2, label='Keyword Present')
            ax.fill_between(time_range, 0, 1, where=(timeline == 1), color='green', alpha=0.5)
            ax.fill_between(time_range, 0, 1, where=(timeline == 0), color='red', alpha=0.3)

            # Style enhancements
            ax.set_title(f"Occurrences of {keyword}", fontsize=14, fontweight='bold')
            ax.set_xlabel("Time (s)", fontsize=12)
            ax.set_ylabel("Presence", fontsize=12)
            ax.set_xticks(np.arange(0, audio_duration + 1, max(1, audio_duration // 10)))
            ax.set_yticks([])
            ax.legend(loc="upper right")
            ax.grid(True, linestyle='--', alpha=0.6)

            st.pyplot(fig)

            # Display details for each occurrence
            for occ in data["occurrences"]:
                st.markdown(
                    f"<span style='color: green;'>Occurrence from {occ['start_time_sec']:.1f}s to {occ['end_time_sec']:.1f}s "
                    f"with confidence {occ['confidence']:.3f}</span>",
                    unsafe_allow_html=True,
                )
    else:
        st.warning("Please upload a query audio file to proceed.")
else:
    st.warning("Please upload exactly 5 audio files to proceed.")
