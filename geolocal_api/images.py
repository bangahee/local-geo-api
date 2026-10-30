import base64
import binascii
from io import BytesIO
from pathlib import Path
import warnings


MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_PIXELS = 25_000_000


class InvalidImage(ValueError):
    pass


def prepare_image(encoded: str, destination: Path):
    """Validate, orient, and write RGB pixels without the source metadata."""
    from PIL import Image, ImageOps, UnidentifiedImageError

    if encoded.startswith("data:"):
        header, separator, encoded = encoded.partition(",")
        if not separator or header not in {
            "data:image/jpeg;base64", "data:image/png;base64", "data:image/webp;base64"
        }:
            raise InvalidImage("Use a JPEG, PNG, or WebP base64 data URL")
    if len(encoded) > 4 * ((MAX_IMAGE_BYTES + 2) // 3):
        raise InvalidImage("Image exceeds the 8 MiB limit")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise InvalidImage("Image must be valid base64") from exc
    if not raw or len(raw) > MAX_IMAGE_BYTES:
        raise InvalidImage("Image must contain between 1 byte and 8 MiB")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(raw)) as original:
                if original.format not in {"JPEG", "PNG", "WEBP"}:
                    raise InvalidImage("Only JPEG, PNG, and WebP images are supported")
                if original.width * original.height > MAX_PIXELS:
                    raise InvalidImage("Image exceeds the 25 megapixel limit")
                if getattr(original, "n_frames", 1) != 1:
                    raise InvalidImage("Animated images are not supported")
                original.load()
                oriented = ImageOps.exif_transpose(original).convert("RGB")
                # Copy pixels into a fresh image: exclude EXIF, GPS, ICC, and text.
                clean = Image.new("RGB", oriented.size)
                clean.paste(oriented)
                clean.thumbnail((2048, 2048))
                clean.save(destination, format="PNG")
    except InvalidImage:
        raise
    except (
        UnidentifiedImageError, OSError, ValueError,
        Image.DecompressionBombError, Image.DecompressionBombWarning,
    ) as exc:
        raise InvalidImage("Image is corrupt, unsupported, or too large") from exc
