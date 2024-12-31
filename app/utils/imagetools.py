import itertools
from fractions import Fraction
from io import BytesIO

from PIL import Image


def rgb_to_hex(rgb):
    """Convert RGB color to HEX."""
    r = min(255, max(0, rgb[0]))
    g = min(255, max(0, rgb[1]))
    b = min(255, max(0, rgb[2]))
    return "#{:02x}{:02x}{:02x}".format(int(r), int(g), int(b))


def rgb_to_xyz(rgb):
    r, g, b = rgb

    # Normalize the RGB values to the range [0, 1]
    r = r / 255.0
    g = g / 255.0
    b = b / 255.0

    # Apply the gamma correction (inverse of sRGB companding)
    if r > 0.04045:
        r = ((r + 0.055) / 1.055) ** 2.4
    else:
        r = r / 12.92

    if g > 0.04045:
        g = ((g + 0.055) / 1.055) ** 2.4
    else:
        g = g / 12.92

    if b > 0.04045:
        b = ((b + 0.055) / 1.055) ** 2.4
    else:
        b = b / 12.92

    # Convert to XYZ using the RGB to XYZ matrix transformation
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

    return (x, y, z)


def xyz_to_lab(xyz):
    x, y, z = xyz

    # Define the reference white point
    x_ref = 0.95047
    y_ref = 1.00000
    z_ref = 1.08883

    # Normalize the XYZ values by the reference white point
    x = x / x_ref
    y = y / y_ref
    z = z / z_ref

    # Apply the LAB transformation
    def f(t):
        if t > 0.008856:
            return t ** (1 / 3)
        else:
            return (7.787 * t) + (16 / 116)

    l = 116 * f(y) - 16
    a = 500 * (f(x) - f(y))
    b = 200 * (f(y) - f(z))

    return (l, a, b)


def rgb_to_lab(rgb):
    xyz = rgb_to_xyz(rgb)
    lab = xyz_to_lab(xyz)
    return lab


def add_watermark_to_image(
    background_image_path: str | bytes | Image.Image,
    watermark_image_path: str,
    position: tuple[int, int] = (0, 0),
    resize: tuple[int, int] = None,
) -> Image:
    """
    Add an watermark image to a background image at a specified position using PIL.

    :param background_image_path: Path to the background image.
    :param watermark_image_path: Path to the watermark image.
    :param position: A tuple (x, y) representing the position to place the watermark.
    :param resize: A tuple (w, h) representing the target size of placed watermark.
    :return: An Image object with the watermark added.
    """

    background = Image.open(background_image_path)
    w, h = background.size
    watermark = Image.open(watermark_image_path)

    position = (w + position[0]) % w, (h + position[1]) % h

    if resize:
        watermark = watermark.resize((50, 50))

    background.paste(watermark, position[0], position[1], watermark)

    return background


def get_aspect_ratio_str(width: int, height: int) -> str:
    fr = Fraction(height, width)
    return f"{fr.denominator}:{fr.numerator}"


def resize_image(image: Image.Image, new_width=384, new_height=None) -> Image.Image:
    if new_width is None and new_height is None:
        return image

    original_width, original_height = image.size
    aspect_ratio = original_height / original_width

    if new_height is None:
        new_height = int(aspect_ratio * new_width)
    elif new_width is None:
        new_width = int(new_height / aspect_ratio)

    resized_image = image.resize((new_width, new_height))
    return resized_image


def crop_image(image: Image.Image, sections=(2, 2), **kwargs) -> list[Image.Image]:
    parts = []
    for i, j in itertools.product(range(sections[0]), range(sections[1])):
        x = j * image.width // sections[0]
        y = i * image.height // sections[1]
        region = image.crop(
            (x, y, x + image.width // sections[0], y + image.height // sections[1])
        )
        parts.append(region)
    return parts


def convert_format_bytes(
    image: Image.Image, convert_format: str = "jpg", quality: float | None = None
) -> BytesIO:
    convert_formats = {
        "jpeg": "JPEG",
        "jpg": "JPEG",
        "webp": "WEBP",
        "png": "PNG",
        "gif": "GIF",
        "bmp": "BMP",
        "tiff": "TIFF",
        "ico": "ICO",
    }
    
    image_bytes = BytesIO()
    if convert_format == "jpg":
        image = image.convert("RGB")
        
    image.save(
        image_bytes,
        format=convert_formats.get(convert_format.lower(), "JPEG"),
        **{"quality": quality} if quality else {},
    )
    image_bytes.seek(0)
    return image_bytes
