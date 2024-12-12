import librosa
import numpy as np
import os
import soundfile as sf

def cut_audio_segment(audio_path, start_time, end_time, output_folder, segment_name):
    """
    Cuts out a portion of the audio from start_time to end_time and saves it to the specified folder.

    Parameters:
        audio_path (str): Path to the original audio file.
        start_time (float): Start time in seconds.
        end_time (float): End time in seconds.
        output_folder (str): Folder where the audio segment will be saved.
        segment_name (str): Name for the output segment file.

    Returns:
        str: Path to the saved audio segment.
    """
    # Load the audio file
    audio, sr = librosa.load(audio_path, sr=None)
    
    # Convert start and end times to sample indices
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    
    # Extract the audio segment
    audio_segment = audio[start_sample:end_sample]
    
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Construct the output file path
    output_path = os.path.join(output_folder, f"{segment_name}.wav")
    
    # Save the audio segment
    sf.write(output_path, audio_segment, sr)
    
    return output_path

def random_int():
    return np.random.randint(250)

def group_keywords(input_dict):
    result = {}

    for file_key, details in input_dict.items():  # Iterate through keys and values in input_dict
        for entry in details:
            keyword = entry["keyword"]
            if keyword not in result:
                result[keyword] = []
            result[keyword].append({
                "start_time": entry["start_time"],
                "end_time": entry["end_time"]
            })

    return result,file_key

def extract_keyword_timestamps(audio_path):
    """
    Processes an audio file to find breaks and returns timestamps for enclosed audio keywords.

    Parameters:
        audio_path (str): Path to the audio file.

    Returns:
        list of tuple: A list of (start_time, end_time) tuples for the detected audio keyword segments.
    """
    # Load the audio file
    audio, sr = librosa.load(audio_path, sr=None)

    # Define the window size and hop length in samples
    window_size = int(0.001 * sr)  # 10 ms window size
    hop_length = int(0.0001 * sr)  # 0.1 ms hop length

    # Compute the STFT
    stft = librosa.stft(audio, n_fft=window_size, hop_length=hop_length, win_length=window_size)

    # Convert the STFT to magnitude and compute the mean across dimension 1
    stft_magnitude = np.abs(stft)
    mean_stft_magnitude = np.mean(stft_magnitude, axis=0)
    mean_stft_magnitude = np.log(mean_stft_magnitude / np.max(mean_stft_magnitude))

    def get_segments(array):
        """Find start and end points of segments in the mean STFT magnitude array."""
        starts = []
        ends = []
        started = False
        ended = False
        keyword = False

        for point in range(len(array) - 1):
            if array[point] < -20 and not started:
                started = True
                starts.append(point)

            if array[point] > -5 and not keyword and started:
                keyword = True

            if array[point] < -20 and keyword and started:
                ended = True

            if array[point] > -5 and ended and started and keyword:
                ends.append(point)
                started = False
                ended = False
                keyword = False
                
            if started and (point - starts[-1] >= 1. * sr/hop_length):
                ends.append(point)
                started = False
                ended = False
                keyword = False

        return starts, ends

    # Get start and end points of segments
    starts, ends = get_segments(mean_stft_magnitude)

    # Convert sample indices to timestamps
    timestamps = [(start * hop_length / sr, end * hop_length / sr) for start, end in zip(starts, ends)]
    return timestamps

def accuracy(ground_truth,result_data):
    correct_matches = 0
    total_predictions = 0
    total_ground_truth = 0

    # Iterate through the results
    for file, predictions in result_data:
        for el in ground_truth_annotations:
            if el.keys()[0] == file:
        ground_truth_annotations = ground_truth[file]
        total_ground_truth += len(ground_truth_annotations)
        total_predictions += len(predictions)
        
        # Match predictions with ground truth
        for pred in predictions:
            for gt in ground_truth_annotations:
                # Check for keyword match and time overlap
                if pred["keyword"] == gt["keyword"]:
                    if not (pred["end_time"] < gt["start_time"] or pred["start_time"] > gt["end_time"]):
                        correct_matches += 1
                        break

    # Calculate accuracy
    accuracy = correct_matches / total_predictions if total_predictions > 0 else 0

    print(f"Correct Matches: {correct_matches}")
    print(f"Total Predictions: {total_predictions}")
    print(f"Accuracy: {accuracy:.2%}")