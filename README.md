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
* **Custom CLIP loading**: merges and loads `clip_g` and `clip_l` CLIP checkpoints, and uses it to encode the prompt.

## Dependencies

*   **PyTorch:**  `torch` (tested with 2.1.2, but likely works with other recent versions).  CUDA is highly recommended for performance.
*   **Transformers:**  `transformers` (tested with 4.38.2) for CLIP model and tokenizer.
*   **Safetensors:** `safetensors` (tested with 0.4.2) for efficient model loading and saving.
*   **tqdm:** `tqdm` for progress bars.

Install them using pip:

```bash
pip install torch transformers safetensors tqdm
```
You also will need to install a correct version of torch that supports your CUDA version, if you intend to use GPU.

## Usage

1.  **Prepare your Models:**  You need two Stable Diffusion models saved in the `.safetensors` format and CLIP checkpoints.
2.  **Set File Paths:** Modify the following variables in the script to point to your model files and desired output path:

    ```python
    clip_g              = "E:\Apps\...\clip_clip_g_00001_.safetensors"  # Path to your CLIP G .safetensors file
    clip_l              = "E:\Apps\...\clip_clip_l_00001_.safetensors"  # Path to your CLIP L .safetensors file
    prompt_to_follow    = "1girl, spiderman"   # The prompt that will guide the merge
    base_model          = "E:\Apps\...\base_model.safetensors"      # Path to your first (base) model
    secondary_model     = "E:\Apps\...\secondary_model.safetensors"   # Path to your second model
    output_path         = "E:\Apps\...\output.safetensors"          # Path to save the merged model
    ```
3.  **Adjust Parameters (Optional):**  You can fine-tune the merging process by modifying these parameters within the `quantum_merge_models` function call:

    *   `entanglement_strength` (default: 0.7714): Controls the linear interpolation between the blended FFT result and the average of the original parameters.  Higher values favor the FFT-blended result.
    *   `decoherence_factor` (default: 0.2):  Controls the probability of a parameter being directly averaged instead of using the FFT blend.  Higher values result in more averaging.
    *   `chunk_size` (default: 4096):  The size of chunks used in the FFT processing.  Smaller chunks reduce peak memory usage but might increase processing time.  Adjust this based on your available GPU memory.

    Example of changing parameters:

    ```python
    quantum_merge_models(
        model1_path=base_model,
        model2_path=secondary_model,
        prompt=prompt_to_follow,
        output_path=output_path,
        entanglement_strength=0.8,  # Increased entanglement
        decoherence_factor=0.1,     # Reduced decoherence
        chunk_size=2048            # Smaller chunk size
    )
    ```

4.  **Run the Script:** Execute the Python script.  The merged model will be saved to the specified `output_path`.

## Detailed Explanation

### `load_custom_clip(g_path, l_path)`

This function loads a custom CLIP model by merging the state dictionaries from two separate safetensors files (`g_path` and `l_path`).  It then loads these weights into a `CLIPTextModel` instance.  This allows you to use a CLIP model that has been fine-tuned or modified separately from the main Stable Diffusion model.  The function also includes error checking to report any missing or unexpected keys during the weight loading process.

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

## Key Concepts

*   **Hypernetwork:**  A small neural network that generates the parameters (or weights) of another, larger network.  In this case, the hypernetwork generates coefficients that control the blending process.
*   **FFT (Fast Fourier Transform):**  An algorithm that transforms a signal from the spatial domain (e.g., pixel values) to the frequency domain (e.g., frequencies of sine waves).  This allows us to manipulate the model's parameters based on their frequency characteristics.
*   **Entanglement:** In quantum mechanics, entanglement is a phenomenon where two particles become linked and share the same fate, no matter how far apart they are.  Here, it's used metaphorically to describe the linear interpolation between the FFT-blended result and the average of the original parameters.
*   **Decoherence:** In quantum mechanics, decoherence is the loss of quantum coherence, which leads to classical behavior.  Here, it's used metaphorically to describe the process of selectively averaging parameters based on a random mask.
* **Safetensors**: a file format for storing tensors that allows loading using memory mapping. This significantly decreases RAM usage.
* **CLIP**: Contrastive Language-Image Pre-training. A neural network by OpenAI.

## Potential Improvements / Future Work

*   **GUI:**  A graphical user interface would make the script more user-friendly.
*   **Colab Support:**  A graphical user interface on free colab space.
*   **More Sophisticated Hypernetwork:**  Experiment with different hypernetwork architectures (e.g., using attention mechanisms).
*   **Different Blending Strategies:** Explore alternative blending techniques in the frequency domain.
*   **Layer-Specific Parameters:** Allow for different `entanglement_strength` and `decoherence_factor` values for different layers of the model.
*   **Dynamic Chunk Size:**  Automatically adjust the `chunk_size` based on available GPU memory.



## Notes:

* I am not sure if the prompt following part of the script works as it should, the code was written with deepseek's help
* To get clip_l and clip_g you can use comfy with a simple set-up, load checkpoint -> save clip (you can use noobai for the clip etc)
