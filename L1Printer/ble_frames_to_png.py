#!/usr/bin/env python3
from PIL import Image
import numpy as np
import minilzo

all_frames = [
    "66 6E 00 1B 2F 03 01 00 01 00 01 80 01 60 00 03 02 FF FF FF FF FF 91 00 7F 94 00 B1 00 BF B8 00 95 00 DF 94 00 B1 00 EF B8 00 95 00 F7 94 00 B1 00 FB B8 00 95 00 FD 94 00 B1 00 FE B8 00 B4 00 20 3C 80 01 20 3C 70 01 7C 17 20 3C 80 01 20 3C 70 01 20 00 00 6D F4 02 0E FF FF FF FF FF FF FE FF FF FF FF FF FF FF FF FF FF 11 00 00 93",
    "66 32 00 1B 2F 03 01 00 01 00 01 80 01 60 00 02 02 FF FF FF FF FF 20 00 00 00 00 4F 10 00 0C FF FF FF FF FF FF FF FF FF FF FF FF FF FF FF 11 00 00 AB",
    "66 32 00 1B 2F 03 01 00 01 00 01 80 01 60 00 01 02 FF FF FF FF FF 20 00 00 00 00 4F 10 00 0C FF FF FF FF FF FF FF FF FF FF FF FF FF FF FF 11 00 00 AC",
    "66 60 00 1B 2F 03 01 00 01 00 01 80 01 60 00 00 03 FF FE FF FF FF FF C1 00 FD DC 00 79 00 FB 70 00 CD 00 F7 DC 00 79 00 EF 70 00 CD 00 DF DC 00 79 00 BF 70 00 CD 00 7F DE 00 FF FF 20 3F 78 01 61 0D FE 70 00 CC 00 20 00 00 00 86 04 03 0A FF FF FF FF FF FF FF FF FF FF FF 7F FF 11 00 00 04"
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
