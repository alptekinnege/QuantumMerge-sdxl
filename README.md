# Quantum Model Merger

This script merges two Stable Diffusion models using a novel technique inspired by quantum mechanics principles, including entanglement, decoherence, and frequency-domain processing.  It uses a hypernetwork conditioned on a text prompt to guide the merging process, and incorporates chunked FFT processing for memory efficiency, particularly with large models.
Used for the development of:
* https://civitai.com/models/1245906/illustrious-bunny
* https://civitai.com/models/1220203/bunnygunny-v2-noob-pony

## Features

*   **Quantum-Inspired Merging:** Combines models using a blend of:
    *   **Entanglement Blending:**  Linearly interpolates the model parameters, controlled by `entanglement_strength`.
    *   **Decoherence Mask:**  Selectively averages parameters based on a random mask (akin to quantum decoherence), controlled by `decoherence_factor`.
    *   **Frequency Domain Processing:**  Blends the models in the frequency domain (using FFT) guided by a hypernetwork, allowing for nuanced, frequency-dependent merging.
*   **Prompt-Conditioned Hypernetwork:**  A small neural network (hypernetwork) generates blending coefficients based on a text prompt. This allows the prompt to influence *how* the models are merged.
*   **Memory Optimization:**
    *   **Chunked FFT Processing:**  The FFT blending is performed in chunks to avoid large memory allocations, making it suitable for GPUs with limited VRAM.
    *   **Half-Precision (FP16):**  Uses half-precision floating-point numbers where possible to reduce memory usage.
    *   **Memory Mapping:** Uses `safetensors`' memory mapping for loading models, minimizing RAM usage.
    *   **Explicit Garbage Collection:**  Includes explicit deletion of tensors and calls to `gc.collect()` to manage memory aggressively.
* **CLIP Extraction**: merges and loads `clip_g` and `clip_l` from either the base or secondary model, and uses it to encode the prompt.

## Requirements
```
torch==2.1.2
transformers==4.38.2
safetensors==0.4.2
tqdm==bars
gradio
gc
```
Install them using pip (venv preffered):

```bash
pip install torch==2.1.2 transformers==4.38.2 safetensors==0.4.2 tqdm gradio gc
```
You also will need to install a correct version of torch that supports your CUDA version, if you intend to use GPU.

## Usage

1.  **Prepare your Models:**  You need two Stable Diffusion models saved in the `.safetensors` format.
2.  **Run the script, it will open a Gradio interface and with the paths:**

3.  **Adjust Parameters (Optional):**  You can fine-tune the merging process by modifying these parameters within the `quantum_merge_models` function call:

    *   `entanglement_strength` (default: 0.7714): Controls the linear interpolation between the blended FFT result and the average of the original parameters.  Higher values favor the FFT-blended result.
    *   `decoherence_factor` (default: 0.2):  Controls the probability of a parameter being directly averaged instead of using the FFT blend.  Higher values result in more averaging.
    *   `chunk_size` (default: 4096):  The size of chunks used in the FFT processing.  Smaller chunks reduce peak memory usage but might increase processing time.  Adjust this based on your available GPU memory.

## Detailed Explanation

### `process_fft_chunked(param1_half, param2_half, hyper_out, decoherence_mask, chunk_size=32)`

This function performs the core frequency-domain blending.  It takes two model parameters (`param1_half`, `param2_half`), the hypernetwork output (`hyper_out`), a decoherence mask (`decoherence_mask`), and a chunk size (`chunk_size`).

1.  **Chunking:**  The input tensors are divided into smaller chunks along the first dimension to limit memory usage.
2.  **FFT:**  The Fast Fourier Transform (FFT) is applied to each chunk of both parameters, transforming them into the frequency domain.
3.  **Frequency Blending:** The hypernetwork output (`hyper_out`) is used to generate blending coefficients (magnitude and phase).  These coefficients determine how the real and imaginary components of the FFTs are combined.  This is where the prompt's influence is applied.
4.  **Inverse FFT:**  The inverse FFT is applied to the blended frequency representation, converting it back to the spatial domain.
5.  **Decoherence Mask Application:**  The decoherence mask is used to selectively replace elements of the blended result with the average of the original parameters.
6.  **Memory Management:** The function uses `torch.no_grad()`, moves chunks to the GPU only when needed, converts data to half-precision where appropriate, and explicitly deletes intermediate tensors and calls `gc.collect()` to free up memory.

### `quantum_merge_models(...)`

This is the main function that orchestrates the entire merging process.

1.  **Load Models:** Loads the two input models using `safetensors.torch.load_file` with memory mapping.
2.  **Hypernetwork Initialization:** Creates a small hypernetwork (a simple feed-forward neural network) that will generate blending coefficients.  It's initialized on the GPU and set to half-precision.
3.  **Prompt Encoding:**  Uses a tokenizer and the custom loaded CLIP text encoder to convert the input `prompt` into text embeddings.
4.  **Hypernetwork Output:**  The text embeddings are passed through the hypernetwork to generate the `hyper_out` tensor, which will be used to control the frequency blending.
5.  **Parameter Iteration:**  Iterates through the parameters of the first model (using `model1.keys()`).
6.  **Merging Logic:**
    *   **Parameter Matching:** Checks if the current parameter key exists in both models.
    *   **Weight Parameter Handling:** If the parameter key contains "weight," it's considered a weight parameter and processed using the FFT blending logic:
        *   **Decoherence Mask Generation:**  A random mask is generated *on the CPU* to save VRAM.
        *   **FFT Blending:**  Calls `process_fft_chunked` to perform the frequency-domain blending.
        *   **Entanglement Blending:** The result of the FFT blend is combined with the average of the original parameters using the `entanglement_strength`.
    *   **Non-Weight Parameter Handling:** If the parameter is not a weight, it's simply averaged.
    *   **Memory Management:**  Parameters are moved to the GPU, processed, and then moved back to the CPU.  Intermediate tensors are deleted, and `gc.collect()` is called.
7.  **Save Merged Model:** Saves the merged model to the specified `output_path` using `safetensors.torch.save_file`.
8. **Final Cleanup**: Deletes some of the leftover variables and calls `gc.collect()` again.

## Update log
*24/03/2025 - added gradio gui and auto clip extraction

## Potential Improvements / Future Work

*   **GUI:**  A graphical user interface would make the script more user-friendly.
*   **More Sophisticated Hypernetwork:**  Experiment with different hypernetwork architectures (e.g., using attention mechanisms).
*   **Different Blending Strategies:** Explore alternative blending techniques in the frequency domain.
*   **Layer-Specific Parameters:** Allow for different `entanglement_strength` and `decoherence_factor` values for different layers of the model.
*   **Dynamic Chunk Size:**  Automatically adjust the `chunk_size` based on available GPU memory.

## Special thanks
Special thanks to the ComfyUI team for the clip extraction code
