import torch.nn as nn
import librosa
import numpy as np
import soundfile as sf
from typing import Union, Optional, Tuple
import torch
from transformers import WhisperModel, WhisperFeatureExtractor
from transformers import WhisperProcessor, WhisperModel

class AudioPreprocessor:
    def __init__(
        self, 
        target_sr: int = 16000, 
        normalize: bool = True,
        trim_silence: bool = True,
        max_duration: Optional[float] = None,
        mono: bool = True
    ):
        self.target_sr = target_sr
        self.normalize = normalize
        self.trim_silence = trim_silence
        self.max_duration = max_duration
        self.mono = mono
        
        
    def process(self, audio_path: Union[str, np.ndarray]) -> np.ndarray:
        if isinstance(audio_path, str):
            audio, orig_sr = librosa.load(audio_path, sr=None, mono=False)
            if self.mono and audio.ndim > 1:
                audio = librosa.to_mono(audio)
        elif isinstance(audio_path, np.ndarray):
            audio, orig_sr = audio_path, self.target_sr
        else:
            raise TypeError("Input must be a file path or numpy array")

        if orig_sr != self.target_sr:
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=self.target_sr)
            
        peak_normalization = audio / np.max(np.abs(audio))
        rms_normalization = audio / np.sqrt(np.mean(audio**2))
        audio = peak_normalization * 0.5 + rms_normalization * 0.5
        
        if self.trim_silence:
            audio, _= librosa.effects.trim(audio)
        if self.max_duration is not None:
            max_length = int(self.max_duration * self.target_sr)
            audio = audio[:max_length]
        audio = np.array(audio, dtype=np.float16)
        
        if len(audio) < self.target_sr:
            pad_length = self.target_sr - len(audio)
            audio = np.pad(audio, (0, pad_length), mode='constant')
        else:
            audio = audio[:self.target_sr]
            
        return audio
    

    def save_processed_audio(
        self, 
        audio: np.ndarray, 
        output_path: str, 
        format: str = 'wav'
    ):
        sf.write(output_path, audio, self.target_sr, format=format)

class EmbeddingClassifier(nn.Module):
    def __init__(self, input_size=25600, num_classes=441):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )
    
    def forward(self, _, embeddings, __):
        return self.classifier(embeddings)

def inference(model, audio_path, device):
    model.load_state_dict(torch.load('best_model_cpu.pth'))
    model.to(device).eval()
    processor = WhisperProcessor.from_pretrained("openai/whisper-base")
    modelWhisper = WhisperModel.from_pretrained("openai/whisper-base").encoder.to(device)
    
    with torch.no_grad():
        audio_processor = AudioPreprocessor()
        audio_array = audio_processor.process(audio_path)
        inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        encoder_outputs = modelWhisper(inputs['input_features'])
        embeddings = encoder_outputs.last_hidden_state.cpu().numpy()[:,:50,:]
        embeddings = torch.tensor(embeddings).to(device)
        embeddings = embeddings.view(embeddings.size(0), -1)
        outputs = model(None, embeddings, None)
    
    _, predicted = torch.max(outputs, 1)
    return predicted.item()

device = 'cpu'
