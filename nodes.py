import torch
import torchaudio
import os
import numpy as np

try:
    from transformers import AutoProcessor, CsmForConditionalGeneration
except ImportError:

    print("Error: transformers library not found or version too old. Please install transformers>=4.49.0")
    AutoProcessor = None
    CsmForConditionalGeneration = None

class CSM_Transformers_Node:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        self.current_model_id = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "Hello from Sesame."}),
                "speaker_id": ("STRING", {"default": "0"}),
                "model_id": ("STRING", {"default": "sesame/csm-1b"}),
            },
            "optional": {
                "ref_audio": ("AUDIO",),
                "ref_text": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "SesameAI/CSM"

    def load_model(self, model_id):
        if AutoProcessor is None:
            raise ImportError("transformers>=4.49.0 is required for CSM node.")

        if self.model is None or self.processor is None or self.current_model_id != model_id:
            print(f"Loading CSM model: {model_id}")
            if self.model is not None:
                self.model.to("cpu")
                del self.model
                torch.cuda.empty_cache()
            
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = CsmForConditionalGeneration.from_pretrained(model_id, device_map=self.device)
            self.current_model_id = model_id
        
        return self.model, self.processor

    def generate(self, text, speaker_id, model_id, ref_audio=None, ref_text=None):
        model, processor = self.load_model(model_id)
        
        conversation = []
        
        if ref_audio is not None and ref_text and ref_text.strip():
            waveform = ref_audio['waveform']
            if waveform.dim() == 3:
                waveform = waveform[0]
            
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            target_sr = 24000
            source_sr = ref_audio.get('sample_rate', 44100)
            
            if source_sr != target_sr:
                resampler = torchaudio.transforms.Resample(orig_freq=source_sr, new_freq=target_sr)
                waveform = resampler(waveform)
            
            waveform_np = waveform.squeeze().cpu().numpy()

            conversation.append({
                "role": str(speaker_id), 
                "content": [
                    {"type": "text", "text": ref_text},
                    {"type": "audio", "path": waveform_np} 
                ]
            })
        
        conversation.append({
            "role": str(speaker_id),
            "content": [{"type": "text", "text": text}]
        })
        
        inputs = processor.apply_chat_template(
            conversation,
            tokenize=True,
            return_dict=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, output_audio=True)
        
        generated_audio = outputs[0].cpu().float()
        
        if generated_audio.dim() == 1:
            generated_audio = generated_audio.unsqueeze(0).unsqueeze(0)
        elif generated_audio.dim() == 2:
            generated_audio = generated_audio.unsqueeze(0)
            
        return ({"waveform": generated_audio, "sample_rate": 24000},)



NODE_CLASS_MAPPINGS = {
    "CSM_Transformers_Node": CSM_Transformers_Node
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CSM_Transformers_Node": "CSM Transformers"
}
