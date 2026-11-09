"""Convert white ink on black into a normalized MNIST-sized image."""
import base64
import binascii

import cv2
import numpy as np


class InvalidDrawing(ValueError):
    """A message that can be shown on the page."""


def prepare_drawing(data):
    if not isinstance(data, str) or not data.startswith('data:image/png;base64,'):
        raise InvalidDrawing('Please send a PNG drawing from the canvas.')
    try:
        raw = base64.b64decode(data.split(',', 1)[1], validate=True)
        if not raw.startswith(b'\x89PNG\r\n\x1a\n'):
            raise ValueError()
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
        if image is None or image.dtype != np.uint8:
            raise ValueError()
        height, width = image.shape[:2]
        if not (28 <= width <= 1024 and 28 <= height <= 1024):
            raise ValueError()
    except (ValueError, binascii.Error, cv2.error):
        raise InvalidDrawing('Invalid PNG image. Clear the canvas and draw a digit.') from None

    if image.ndim == 3:
        gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
        if image.shape[2] == 4:
            # Transparent pixels contribute no ink on the black background.
            gray = (gray.astype(np.float32) * image[:, :, 3] / 255).astype(np.uint8)
    else:
        gray = image

    mask = gray > 20
    ys, xs = np.nonzero(mask)
    if len(xs) < 12 or np.ptp(xs) < 2 or np.ptp(ys) < 2:
        raise InvalidDrawing('The canvas is empty or the drawing is too small. Draw one digit.')

    # Keep the proportions and fit the digit inside a 20 x 20 box, like MNIST.
    gray = np.where(mask, gray, 0).astype(np.uint8)
    crop = gray[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    scale = 20 / max(crop.shape)
    size = (max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale)))
    resized = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
    result = np.zeros((28, 28), np.float32)
    height, width = resized.shape
    top, left = (28 - height) // 2, (28 - width) // 2
    result[top:top + height, left:left + width] = resized / 255.0

    # Move the centre of mass to the centre of the model's input.
    moments = cv2.moments(result)
    dx = 13.5 - moments['m10'] / moments['m00']
    dy = 13.5 - moments['m01'] / moments['m00']
    result = cv2.warpAffine(result, np.float32([[1, 0, dx], [0, 1, dy]]), (28, 28))
    return result[np.newaxis, ...].astype(np.float32)
