import time
import logging
import boto3
from PIL import Image
import numpy as np
import random
from io import BytesIO

# Configure root logger for CloudWatch
logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3 = boto3.client('s3')

# ------------------------------
# Generic timing decorator
# ------------------------------

def timeit(label: str):
    """Decorator that logs the execution time of the wrapped function."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            elapsed_ms = (time.time() - start) * 1000
            logger.info(f"{label} took {elapsed_ms:.2f} ms")
            return result

        return wrapper

    return decorator


# ------------------------------
# Image helpers
# ------------------------------

@timeit("get_image")
def get_image(bucket: str, key: str) -> bytes:
    """Download an object from S3 and return its raw bytes."""
    obj = s3.get_object(Bucket=bucket, Key=key)
    return obj["Body"].read()

@timeit("save_to_buffer")
def save_to_buffer(processed_img):
    """Save PIL image to an in‑memory buffer and rewind it."""
    buffer = BytesIO()
    processed_img.save(buffer, format="JPEG")
    buffer.seek(0)
    return buffer


@timeit("upload_to_s3")
def upload_to_s3(buffer, bucket_name, output_key):
    """Upload the buffer to the given S3 bucket/key."""
    s3.put_object(Bucket=bucket_name, Key=output_key, Body=buffer, ContentType="image/jpeg")


@timeit("transform_image_to_PIL")
def transform_image_to_PIL(image_bytes: bytes) -> Image.Image:
    """Convert raw bytes to a PIL Image."""
    return Image.open(BytesIO(image_bytes))


@timeit("transform_image_from_PIL_to_np")
def transform_image_from_PIL_to_np(img: Image.Image) -> np.ndarray:
    """Convert PIL Image to NumPy array."""
    return np.array(img)


@timeit("compute_color_distance")
def compute_color_distance(img_array: np.ndarray, color: np.ndarray) -> np.ndarray:
    """Return per‑pixel Manhattan distance from the given RGB color."""
    return np.sum(np.abs(img_array - color), axis=2)


@timeit("apply_mask")
def apply_mask(img_array: np.ndarray, mask: np.ndarray) -> Image.Image:
    """Gray out pixels outside the mask and return a PIL Image."""
    R, G, B = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
    gray = (0.2989 * R + 0.5870 * G + 0.1140 * B).astype(np.uint8)
    img_array[mask] = np.stack([gray[mask]] * 3, axis=-1)
    return Image.fromarray(img_array)


@timeit("decolor")
def decolor(image_bytes: bytes, key: str, bucket_name: str, color: np.ndarray) -> str:
    """Apply decolor transformation and upload result to S3, returning the output key."""
    img = transform_image_to_PIL(image_bytes)
    img_array = transform_image_from_PIL_to_np(img)

    # Build mask and decolor
    summed_temp = compute_color_distance(img_array, color)
    mask = summed_temp > 60
    processed_img = apply_mask(img_array, mask)

    # Save to buffer
    buffer=save_to_buffer(processed_img)

    # Upload to S3
    output_key = f"out/{key}"
    upload_to_s3(buffer, bucket_name, output_key)

    return output_key


# ------------------------------
# Lambda entry point
# ------------------------------

def lambda_handler(event, context):
    bucket_name = "imagesbucketforwork"
    prefix = "4k/"  # Folder inside the bucket
    overall_start = time.time()

    try:
        # List objects under the prefix
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        if "Contents" not in response:
            return {"statusCode": 404, "body": "No images found in the folder."}

        image_keys = [item["Key"] for item in response["Contents"] if item["Key"] != prefix]
        images_data = []

        key=random.choice(image_keys)
        
        # Download and process each image
        image_bytes = get_image(bucket_name, key)
        output_key = decolor(image_bytes ,key, bucket_name, color=np.array([40,50,40]))
        images_data.append({"original_key": key, "processed_key": output_key})

        overall_elapsed = (time.time() - overall_start) * 1000
        logger.info(f"lambda_handler completed in {overall_elapsed:.2f} ms")

        return {
            "statusCode": 200,
            "body": {"total_images": len(images_data), "images_info": images_data},
        }

    except Exception as exc:
        logger.exception("Unhandled error in lambda_handler")
        return {"statusCode": 500, "body": f"Error: {str(exc)}"}