# AI-Powered Text-to-Image Generation with Latent Diffusion

This project implements a text-to-image generation system using a latent diffusion framework. It fine-tunes the CLIP model on the Flickr30k dataset to improve semantic understanding of natural language prompts, and leverages pre-trained components from CompVis and OpenAI to generate realistic images from text descriptions.

---

## Project Highlights

- Fine-tuned OpenAI CLIP using Flickr30k image-caption pairs for improved text-image alignment.
- Built a complete image generation workflow using latent diffusion with VAE, U-Net, and a PNDM scheduler.
- Extracted normalized text and image embeddings to optimize caption-image similarity matching.
- Generated diverse images from prompts using Hugging Face’s `diffusers` and `transformers` libraries.
- Achieved an 18% improvement in alignment accuracy after CLIP fine-tuning.

---

## Dataset: Flickr30k

- 31,783 images
- 158,915 human-written captions (approximately 5 per image)
- Captions vary in length and subject matter, covering diverse real-world scenes

---

## Pretrained Models

| Model | Description |
|-------|-------------|
| **VAE** | Encodes and reconstructs images as latent vectors (CompVis/stable-diffusion-v1-4) |
| **U-Net** | Denoises latent representations during diffusion (CompVis/stable-diffusion-v1-4) |
| **CLIP** | Aligns text and image features for multimodal understanding (openai/clip-vit-base-patch32) |
| **PNDM Scheduler** | Controls the noise schedule in the diffusion process |

---

## Project Structure

| File | Description |
|------|-------------|
| `eda_flickr30k.ipynb` | Exploratory data analysis and CLIP fine-tuning |
| `prompt_to_image_generation.ipynb` | Text-to-image generation using latent diffusion |
| `generated_images/` | Sample output images paired with their prompts |

---

## Sample Results

| Prompt | Generated Image |
|--------|-----------------|
| "Dog playing on a beach" | ![](./generated_images/dog_beach.png) |
| "A man giving food to cats" | ![](./generated_images/man_feeding_cats.png) |
| "Dog and cat playing on a beach" | ![](./generated_images/dog_cat_beach.png) |
| "City damage after earthquake" | ![](./generated_images/city_earthquake.png) |

---

## How to Run

This project is designed to run in Jupyter, Colab, or similar environments.

1. Open the following notebooks in order:
   - `eda_flickr30k.ipynb` (data preparation and CLIP fine-tuning)
   - `prompt_to_image_generation.ipynb` (image generation)

2. Follow inline instructions and use provided `pip install` commands as needed.

---

## Author

**Foram Trivedi**  
Generative AI · Diffusion Models · CLIP · PyTorch  
[GitHub Profile](https://github.com/trivedif)

---

## Acknowledgements

- [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
- [OpenAI CLIP](https://github.com/openai/CLIP)
- [Flickr30k Dataset](http://shannon.cs.illinois.edu/DenotationGraph/)
