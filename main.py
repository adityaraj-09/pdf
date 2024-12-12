import streamlit as st
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path
import os
import zipfile
from extract_timeStamps import *
from inference import *
import pickle
import librosa
# Streamlit UI
st.title("Few Shot Language Agnostic Keyword Spotting (FSLAKWS)")
@st.cache_data
def process_audio_files(found_wav_files,id_key):
    result_list = []
    for idx, wav_file in enumerate(found_wav_files[:3]):
        timestamps = extract_keyword_timestamps(wav_file)
        kw_ts = []
        for idx, (start, end) in enumerate(timestamps):
            segment_name = f"{os.path.basename(wav_file).split('.')[0]}_start_{start}_end_{end}_{idx + 1}"
            cut_audio_segment(wav_file, start, end, "audio_samples", segment_name)
            model = EmbeddingClassifier()
            final_idx=inference(model, "./audio_samples/"+segment_name+".wav", device)
            kw_ts.append({
                "start_time": start,
                "end_time": end,
                "keyword": id_key[final_idx]
            })

        result_list.append({
            wav_file: kw_ts
        })
    return result_list

# Step 1: Upload Five Audio Files
st.header("Step 1: Upload a ZIP File")
uploaded_zip = st.file_uploader("Upload a ZIP file", type=["zip"])
result_list=[]

if uploaded_zip:
    # Define the temporary extraction directory
    extract_path = './DATA'
    os.makedirs(extract_path, exist_ok=True)

    # Extract the zip file into the temporary directory
    zip_path = os.path.join(extract_path, uploaded_zip.name)
    with open(zip_path, 'wb') as f:
        f.write(uploaded_zip.getvalue())
    
    # Extract the zip file into the temporary directory
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    
    # Check for .wav files in the innermost folder
    found_wav_files = []
    kw_to_id_path = None
    
   

    for root, _, files in os.walk(extract_path):
        for file in files:
            if file.lower().endswith('.wav'):
                found_wav_files.append(os.path.join(root, file))
            elif file.lower() == 'kw_to_id.pkl':
                kw_to_id_path = os.path.join(root, file)
            elif file.lower().endswith('.pkl'):
                p = os.path.join(root, file)
                
                  
              
            
                    

    # Display number of .wav files found
    st.success(f"Found {len(found_wav_files)} .wav files ")
    if p:
        with open(p, 'rb') as kw_file:
            dummy = pickle.load(kw_file)
        
    #     with open("dummy.json", 'w') as json_file:
    #                 json.dump(dummy, json_file, indent=4)    
        
    if kw_to_id_path:
        with open(kw_to_id_path, 'rb') as kw_file:
            kw_to_id = pickle.load(kw_file)
            print(kw_to_id)
            print(type(kw_to_id))

            id_to_kw = {kw_to_id[k]:k for k in kw_to_id}
    #     json_output_path = "kw_to_id.json"
    
    # # Dump the data into a JSON file
    #     with open(json_output_path, 'w') as json_file:
    #         json.dump(kw_to_id, json_file, indent=4)    
    else:
        st.error("Could not find 'kw_to_id.pkl' in the ZIP archive.")        
            
    with st.spinner("Processing audio..."):
        result_list = process_audio_files(found_wav_files,id_to_kw)

# Show a success message once processing is complete
    accuracy(dummy,result_list)
    st.success("Finished working!")
            
    with open("result.json", 'w') as json_file:
        json.dump(result_list, json_file, indent=4) 
                    

    
    

    if result_list:

        # Step 5: Visualize Keyword Occurrences
        st.header("Step 5: Visualize Keyword Occurrences")
        file_options = ["Select a file"] + [f"File {i}: {list(result.keys())[0]}" for i, result in enumerate(result_list)]
        selected_index = st.selectbox("Select a file from result_list", range(-1, len(result_list)), format_func=lambda x: file_options[x + 1])

        if selected_index != -1:
            selected_file_data = result_list[int(selected_index)]

            single_result, file_key = group_keywords(selected_file_data)
            st.audio(file_key, format='audio/wav')

            # Prepare a number line for all keyword occurrences
            time_range = np.linspace(0, 30, 1000)  # Simulated timeline (0 to 30 seconds)
            timeline_matrix = np.zeros((len(single_result), len(time_range)))  # Rows for each keyword
            occurrence_times = []  # Store times for custom ticks

            keyword_colors = ["blue", "red", "green", "purple", "orange"]  # Add more colors as needed
            fig, ax = plt.subplots(figsize=(12, 4))

            for idx, (keyword, data) in enumerate(single_result.items()):
                for occ in data:
                    start_time = max(0, min(30, occ["start_time"]))
                    end_time = max(0, min(30, occ["end_time"]))
                    occurrence_times.extend([start_time, end_time])

                    # Find indices safely
                    start_idx = np.where(time_range >= start_time)[0][0]
                    end_idx = np.where(time_range >= end_time)[0][0]
                    timeline_matrix[idx, start_idx:end_idx] = 1  # Mark presence for the keyword

                # Plot each keyword's timeline
                ax.fill_between(
                    time_range,
                    idx,
                    idx + 1,
                    where=(timeline_matrix[idx] == 1),
                    color=keyword_colors[idx % len(keyword_colors)],
                    alpha=0.5,
                    label=keyword,
                )

            # Style enhancements
            ax.set_title("Keyword Occurrences", fontsize=14, fontweight="bold")
            ax.set_xlabel("Time (s)", fontsize=12)
            ax.set_ylabel("Keywords", fontsize=12)
            ax.set_yticks(range(len(single_result)))
            ax.set_yticklabels(list(single_result.keys()))

            # Combine default ticks with occurrence times
            default_ticks = np.arange(0, 30 + 1, max(1, 30 // 10))
            custom_ticks = sorted(set(default_ticks).union(occurrence_times))
            ax.set_xticks(custom_ticks)
            ax.set_xticklabels([f"{tick:.1f}s" for tick in custom_ticks], rotation=45)
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.legend(loc="upper right")

            # Show the plot in Streamlit
            st.pyplot(fig)

        else:
            st.warning("Please select an audio file to proceed.")