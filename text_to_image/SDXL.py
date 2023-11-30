from diffusers import DiffusionPipeline
import torch

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

example_prompt = "A picture of Berlin in Winter"


def load_base_and_refiner(model_path=r"D:\models\stable-diffusion-xl-base-1.0"):
    # load both base & refiner
    base = DiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    ).to(DEVICE)

    refiner = DiffusionPipeline.from_pretrained(
        model_path,
        text_encoder_2=base.text_encoder_2,
        vae=base.vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    ).to(DEVICE)

    return base, refiner


def create_image(base, refiner, prompt, n_steps=50, high_noise_frac=0.8):
    # run both experts
    image = base(
        prompt=prompt,
        num_inference_steps=n_steps,
        denoising_end=high_noise_frac,
        output_type="latent",
    ).images

    image = refiner(
        prompt=prompt,
        num_inference_steps=n_steps,
        denoising_start=high_noise_frac,
        image=image,
    ).images[0]

    return image


def main():
    print("Using device:", DEVICE)

    print("Loading base and refiner...")
    # load both base & refiner
    base, refiner = load_base_and_refiner()

    while True:
        prompt = input("You: ")

        # Check if the user wants to exit the conversation
        if prompt.lower() == 'exit':
            print("Goodbye!")
            break

        # create image
        image = create_image(base, refiner, prompt)

        # show image in new window
        image.show()


if __name__ == "__main__":
    main()
