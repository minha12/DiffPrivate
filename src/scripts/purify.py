import os
import numpy as np
from PIL import Image, ImageFilter
from tqdm import tqdm
from art.defences.preprocessor import (
    JpegCompression,
    TotalVarMin,
    FeatureSqueezing,
    SpatialSmoothing,
)
import hydra
from omegaconf import DictConfig


def apply_preprocessor(preprocessor, image_path):
    image = Image.open(image_path)
    x = np.array(image)
    x = x.reshape((1,) + x.shape)
    clip_values = (0, 255)

    if preprocessor == "jpeg_compression":
        preprocessor_obj = JpegCompression(quality=50, clip_values=clip_values)
    elif preprocessor == "total_var_min":
        preprocessor_obj = TotalVarMin(clip_values=clip_values)
    elif preprocessor == "feature_squeezing":
        preprocessor_obj = FeatureSqueezing(bit_depth=4, clip_values=clip_values)
    elif preprocessor == "spatial_smoothing":
        preprocessor_obj = SpatialSmoothing(window_size=3, channels_first=False)
    else:  # For custom preprocessors not in ART, handle differently
        if preprocessor == "gaussian_blur":
            sigma = 2.5
            return image.filter(ImageFilter.GaussianBlur(radius=sigma))
        elif preprocessor == "random_noise":
            mean = 0
            stddev = 5
            noise = np.random.normal(mean, stddev, x.shape[1:]).astype(np.uint8)
            return Image.fromarray(
                np.clip(np.array(image) + noise, 0, 255).astype("uint8")
            )

    x_preprocessed, _ = preprocessor_obj(x=x)
    return Image.fromarray(x_preprocessed[0].astype("uint8"))


@hydra.main(version_base=None, config_path="../../configs", config_name="purify_config")
def main(cfg: DictConfig):
    preprocessors = cfg.defense.preprocessors
    input_path = cfg.defense.input_path
    output_path = cfg.defense.output_path

    for preprocessor in preprocessors:
        print(f"Processing with {preprocessor}...")
        output_folder = os.path.join(output_path, preprocessor)
        os.makedirs(output_folder, exist_ok=True)

        processed_images = set()
        for image_name in os.listdir(input_path):
            if not image_name.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            
            if "adv" in image_name:
                prefix = image_name[:5]
            if prefix in processed_images:
                continue
                
            processed_images.add(prefix)
            # Process all images with the same prefix where one has "adv"
            for related_img in os.listdir(input_path):
                if related_img.startswith(prefix) and "adv" in related_img:
                    img_path = os.path.join(input_path, related_img)
                    preprocessed_image = apply_preprocessor(preprocessor, img_path)
                    # Extract the original extension
                    ext = os.path.splitext(related_img)[1]
                    # Save with prefix + original extension
                    output_filename = prefix + ext
                    preprocessed_image.save(os.path.join(output_folder, output_filename))


if __name__ == "__main__":
    main()
