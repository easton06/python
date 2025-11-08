#!/usr/bin/env python3
from PIL import Image
import numpy as np
import minilzo

all_frames = [
    "66 21 00 1B 2F 03 01 00 01 00 01 2B 01 55 00 03 03 00 02 00 00 00 00 20 CB 02 00 07 FF 20 0C 01 80",
    "66 21 00 1B 2F 03 01 00 01 00 01 2B 01 55 00 02 00 E0 20 00 00 00 00 00 00 00 00 00 00 00 00 00 A6",
    "66 21 00 1B 2F 03 01 00 01 00 01 2B 01 55 00 01 00 00 DE BC 00 20 BB 2C 44 00 02 00 00 00 00 00 C0",
    "66 23 00 1B 2F 03 01 00 01 00 01 2B 01 2C 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 11 00 00 BE"
]

def lzo_decompress_ble_frame(data: bytes) -> bytes | None:
    try:
        # convert hex bytes to actual bytes
        frame_bytes = bytes.fromhex(data)
        # skip 17 bytes (header), then discard last byte (checksum)
        compressed_data = frame_bytes[17:-1]
        dst_len = 9999 # weird API, max decompressed length has to be known beforehand or we fail.
        # TODO: find a better python API for LZO-1X decompress
        decompressed = minilzo.decompress(compressed_data, dst_len)
        print(f"[✓] Decompressed {len(compressed_data)} bytes to {len(decompressed)} bytes")
        return decompressed
    except Exception as e:
        print(f"[-] Failed to decompress raw LZO ({len(compressed_data)}B)", e)
        return None

def convert_decompressed_bytes_to_image(data: bytes, output_path: str):
    # byteswap 16-bit words
    swapped = bytearray()
    for i in range(0, len(data), 2):
        if i + 1 < len(data):
            swapped += data[i + 1:i + 2] + data[i:i + 1]
        else:
            swapped += data[i:i + 1]
    data = swapped
    # flip every bit (black <-> white inverted)
    data = bytes([x ^ 0xff for x in data])
    # assuem static height of 96 pixels and adjust width to match.
    height = 96
    total_bits = len(data) * 8
    width = total_bits // height

    print(f"Decoded dimensions: {width}x{height}")

    # Convert to bit array
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))

    # Reshape as [width, height], scanning columns bottom-to-top
    image_data = np.zeros((height, width), dtype=np.uint8)

    for x in range(width):
        col_bits = bits[x * height:(x + 1) * height]
        image_data[:, x] = col_bits[::-1] * 255  # invert vertically (bottom→top)


    # Create and save grayscale image
    img = Image.fromarray(image_data, mode="L")
    print(f"Saving to {output_path}")
    img.save(output_path)
    img.show()

def main():
    # get decompressed data from all frames
    all_data = b""
    for frame in all_frames:
        decompressed = lzo_decompress_ble_frame(frame)
        if decompressed is not None:
            all_data += decompressed
    convert_decompressed_bytes_to_image(all_data, "reconstructed.png")

if __name__ == '__main__':
    main()
