import torch
import gc
import gradio as gr
from safetensors.torch import load_file, save_file
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextConfig
import os

class QuantumCLIPExtractor:
    @classmethod
    def extract_from_checkpoint(cls, checkpoint_path: str) -> tuple[dict, dict]:
        state_dict = load_file(checkpoint_path)
        components = {"clip_g": {}, "clip_l": {}}
        
        for key in state_dict:
            clean_key = key.replace("conditioner.embedders.0.", "").replace("cond_stage_model.", "")
            if 'text_model.encoder.layers.23' in clean_key or 'text_projection' in clean_key:
                components["clip_g"][clean_key] = state_dict[key]
            elif 'text_model.encoder.layers' in clean_key:
                components["clip_l"][clean_key] = state_dict[key]
                
        return (
            cls.process_component(components["clip_g"]),
            cls.process_component(components["clip_l"])
        )

    @staticmethod
    def process_component(component: dict) -> dict:
        processed = {}
        replacements = {
            "layer_norm1": "self_attn_layer_norm",
            "layer_norm2": "final_layer_norm",
            "mlp.fc1": "fc1",
            "mlp.fc2": "fc2",
            "positional_embedding": "embeddings.position_embedding.weight",
            "token_embedding": "embeddings.token_embedding.weight"
        }
        
        for key in component:
            new_key = key
            for old, new in replacements.items():
                new_key = new_key.replace(old, new)
            processed[new_key] = component[key]
        return processed

def load_custom_clip(ckpt_path: str) -> CLIPTextModel:
    clip_g, clip_l = QuantumCLIPExtractor.extract_from_checkpoint(ckpt_path)
    merged_state = {**clip_g, **clip_l}
    config = CLIPTextConfig.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel(config)
    
    model_state = text_encoder.state_dict()
    filtered = {k: v for k, v in merged_state.items() if k in model_state}
    model_state.update(filtered)
    text_encoder.load_state_dict(model_state, strict=False)
    return text_encoder.eval().to("cuda")

def process_fft_chunked(param1_half, param2_half, hyper_out, decoherence_mask, chunk_size=32):
    orig_shape = param1_half.shape
    flat_shape = (-1, orig_shape[-1])
    flat1 = param1_half.view(flat_shape)
    flat2 = param2_half.view(flat_shape)
    flat_mask = decoherence_mask.view(flat_shape)
    processed_chunks = []
    
    for i in tqdm(range(0, flat1.shape[0], chunk_size), desc="Processing FFT chunks", leave=False):
        with torch.no_grad():
            chunk1 = flat1[i:i+chunk_size].float()
            chunk2 = flat2[i:i+chunk_size].float()
            mask_chunk = flat_mask[i:i+chunk_size].to('cuda', non_blocking=True)

            fft1 = torch.fft.rfft(chunk1, dim=-1)
            fft2 = torch.fft.rfft(chunk2, dim=-1)
            freq_dim = fft1.shape[-1]

            if hyper_out.shape[-1] < freq_dim:
                coeff = hyper_out.repeat(1, freq_dim // hyper_out.shape[-1] + 1)[:, :freq_dim]
            else:
                coeff = hyper_out[:, :freq_dim]
            coeff = coeff.expand(chunk1.size(0), -1).float()

            magnitude_blend = torch.sigmoid(coeff * 5)
            phase_blend = torch.sigmoid(coeff * 3 - 1)

            blended_fft_real = magnitude_blend * fft1.real + (1 - magnitude_blend) * fft2.real
            blended_fft_imag = phase_blend * fft1.imag + (1 - phase_blend) * fft2.imag
            blended_fft = torch.complex(blended_fft_real, blended_fft_imag)

            blended_chunk = torch.fft.irfft(blended_fft, n=chunk1.shape[-1], dim=-1)
            avg = (chunk1 + chunk2) / 2
            blended_chunk[mask_chunk] = avg[mask_chunk]

            blended_chunk = blended_chunk.half().cpu()
            processed_chunks.append(blended_chunk)

            del chunk1, chunk2, fft1, fft2, blended_fft, avg, mask_chunk, magnitude_blend, phase_blend, coeff
    
    blended_flat = torch.cat(processed_chunks, dim=0)
    return blended_flat.view(orig_shape)

def quantum_merge_models(base_model_path, secondary_model_path, clip_source, prompt, output_path, entanglement=0.7714, chunk_size=2048, add_vpred=False):
    try:
        model1 = load_file(base_model_path)
        model2 = load_file(secondary_model_path)
        
        text_encoder = load_custom_clip(base_model_path if clip_source == "Base" else secondary_model_path)
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        
        hypernet = torch.nn.Sequential(
            torch.nn.Linear(768, 1024),
            torch.nn.GELU(),
            torch.nn.Linear(1024, 256),
            torch.nn.Tanh()
        ).cuda().half()

        with torch.no_grad():
            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt"
            )
            text_input_ids = text_inputs.input_ids.to("cuda")
            text_emb = text_encoder(text_input_ids).pooler_output.half()
            hyper_out = hypernet(text_emb).float()

        merged_model = {}
        keys = list(model1.keys())
        
        for key in tqdm(keys, desc="Merging parameters"):
            if key in model2:
                param1 = model1[key].cuda().half()
                param2 = model2[key].cuda().half()

                if 'weight' in key:
                    seed = abs(hash(prompt + key)) % (2**32)
                    torch.manual_seed(seed)
                    decoherence_mask = torch.rand(param1.shape, device='cpu') < 0.2

                    blended = process_fft_chunked(param1, param2, hyper_out, decoherence_mask, chunk_size)
                    merged = (blended.float() * entanglement + 
                             (param1.cpu().float() * (1 - entanglement) + 
                              param2.cpu().float() * (1 - entanglement)) / 2).half()
                else:
                    merged = (param1 + param2) / 2

                merged_model[key] = merged.cpu()
                del param1, param2, merged
                if 'weight' in key: del blended
                gc.collect()

            else:
                merged_model[key] = model1[key]

        save_file(merged_model, output_path)
        
        # Add v_pred tensor if requested
        if add_vpred:
            try:
                state_dict = load_file(output_path)
                state_dict['v_pred'] = torch.tensor([])
                vpred_path = output_path.replace('.safetensors', '_s.safetensors')
                save_file(state_dict, vpred_path)
                return True, f"Merge successful! Saved to {vpred_path}"
            except Exception as e:
                return False, f"v_pred addition failed: {str(e)}"

        return True, f"Merge successful! Saved to {output_path}"
    except Exception as e:
        return False, f"Error: {str(e)}"

def wrapper(base_file, secondary_file, clip_source, prompt, output_name, entanglement, chunk_size, add_vpred):
    try:
        # Get actual file paths from Gradio file objects
        base_path = base_file.name
        secondary_path = secondary_file.name
        
        success, message = quantum_merge_models(
            base_path,
            secondary_path,
            clip_source,
            prompt,
            output_name,
            entanglement,
            chunk_size,
            add_vpred
        )
        
        output_path = output_name.replace('.safetensors', '_s.safetensors') if add_vpred else output_name
        return (output_path if success else None), message
    except Exception as e:
        return None, f"Wrapper error: {str(e)}"

def create_interface():
    with gr.Blocks(title="Quantum Model Merger") as interface:
        gr.Markdown("# 🧪 Quantum Model Merger")
        
        with gr.Row():
            with gr.Column():
                base_model = gr.Textbox(
                    label="1. Base Model Path",
                    placeholder="E:/Models/base_model.safetensors"
                )
                secondary_model = gr.Textbox(
                    label="2. Secondary Model Path",
                    placeholder="E:/Models/secondary_model.safetensors"
                )
                clip_source = gr.Radio(
                    ["Base", "Secondary"], 
                    value="Base",
                    label="3. CLIP Source Model"
                )
                prompt = gr.Textbox(
                    label="4. Fusion Prompt",
                    value="1girl, solo, best quality, masterpiece",
                    lines=3
                )
                output_name = gr.Textbox(
                    label="5. Output Path",
                    value="E:/Models/merged_model.safetensors"
                )
                vpred_check = gr.Checkbox(
                    label="Add v_pred Tensor (for v-prediction models)",
                    value=False
                )
                entanglement = gr.Slider(0.0, 1.0, value=0.77)
                chunk_size = gr.Slider(128, 4096, value=2048, step=128)
                merge_btn = gr.Button("🚀 Start Merge")

            with gr.Column():
                output_file = gr.File(label="Merged Model")
                logs = gr.Textbox(label="Logs", interactive=False, lines=8)

        def wrapper(base_path, secondary_path, clip_source, prompt, output_path, entanglement, chunk_size, add_vpred):
            try:
                # Validate paths
                if not all([path.endswith('.safetensors') for path in [base_path, secondary_path, output_path]]):
                    return None, "Error: All paths must end with .safetensors"
                    
                if not os.path.exists(base_path):
                    return None, f"Base model not found: {base_path}"
                    
                if not os.path.exists(secondary_path):
                    return None, f"Secondary model not found: {secondary_path}"

                # Run merge
                success, message = quantum_merge_models(
                    base_path,
                    secondary_path,
                    clip_source,
                    prompt,
                    output_path,
                    entanglement,
                    chunk_size,
                    add_vpred
                )
                
                # Return final output path
                final_path = output_path.replace('.safetensors', '_s.safetensors') if add_vpred else output_path
                return (final_path if os.path.exists(final_path) else None), message
                
            except Exception as e:
                return None, f"Merge failed: {str(e)}"

        merge_btn.click(
            wrapper,
            [base_model, secondary_model, clip_source, prompt, output_name, entanglement, chunk_size, vpred_check],
            [output_file, logs]
        )
    
    return interface


if __name__ == "__main__":
    interface = create_interface()
    interface.launch()
