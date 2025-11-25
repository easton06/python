#!/usr/bin/env python3
"""
MakeID L1 Printer - CORRECT Protocol Implementation
Split FIRST, then compress each chunk independently
"""

import numpy as np
import minilzo
import math

# === PRINTER SETTINGS ===
PRINTER_ID = bytes([0x1B, 0x2F, 0x03, 0x01, 0x00, 0x01, 0x00, 0x01])
IMAGE_WIDTH = 384
IMAGE_HEIGHT = 96

def calculate_checksum(frame_bytes):
    """Calculate frame checksum (sum subtraction)"""
    checksum = 0
    for byte in frame_bytes[:-1]:
        checksum = (checksum - byte) & 0xFF
    return checksum

def create_test_bitmap(width, height, pattern="border"):
    """Create test bitmap"""
    bitmap = np.zeros((height, width), dtype=np.uint8)
    
    if pattern == "border":
        border_size = 5
        bitmap[0:border_size, :] = 255
        bitmap[-border_size:, :] = 255
        bitmap[:, 0:border_size] = 255
        bitmap[:, -border_size:] = 255
    elif pattern == "all_white":
        bitmap[:] = 255
    elif pattern == "all_black":
        bitmap[:] = 0
    elif pattern == "diagonal":
        for i in range(min(width, height)):
            bitmap[i, i] = 255
            if i < width:
                bitmap[i, width - 1 - i] = 255
    
    return bitmap

def bitmap_to_bytes(bitmap):
    """Convert numpy bitmap to byte array (row-major, 1 bit per pixel)"""
    height, width = bitmap.shape
    flat = bitmap.flatten()
    
    # Pack 8 pixels per byte (MSB first)
    bytes_array = bytearray()
    for i in range(0, len(flat), 8):
        byte = 0
        for bit in range(8):
            if i + bit < len(flat) and flat[i + bit] > 0:
                byte |= (0x80 >> bit)
        bytes_array.append(byte)
    
    return bytes(bytes_array)

def transform_to_printer_format(bitmap_bytes, width, height):
    """
    Transform bitmap to printer format:
    1. Row-major → Column-major (bottom-to-top)
    2. 16-bit word swap
    3. Bit inversion
    """
    BYTES_PER_COLUMN = height // 8  # 12 bytes for 96 pixels
    total_bytes = (width * height) // 8
    
    # Step 1: Convert to column-major (bottom-to-top)
    result = bytearray(total_bytes)
    
    for x in range(width):
        for y in range(height):
            # Read pixel from row-major bitmap
            row_idx = y * width + x
            byte_idx = row_idx // 8
            bit_pos = row_idx % 8
            pixel = (bitmap_bytes[byte_idx] >> (7 - bit_pos)) & 1
            
            # Write to column-major (bottom-to-top)
            col_idx = x * BYTES_PER_COLUMN + (height - 1 - y) // 8
            bit_pos_out = (height - 1 - y) % 8
            
            if pixel:
                result[col_idx] |= (1 << bit_pos_out)
    
    # Step 2: 16-bit word swap
    for i in range(0, total_bytes, 2):
        if i + 1 < total_bytes:
            result[i], result[i + 1] = result[i + 1], result[i]
    
    # Step 3: Invert all bits (0xFF = black for thermal printer)
    for i in range(total_bytes):
        result[i] ^= 0xFF
    
    return bytes(result)

def compress_and_generate_frames(bitmap, image_name):
    height, width = bitmap.shape
    print(f"\n{'='*70}")
    print(f"Generating frames for: {image_name} ({width}x{height})")
    print(f"{'='*70}")
    
    # Step 1: Convert bitmap to bytes
    bitmap_bytes = bitmap_to_bytes(bitmap)
    print(f"1. Bitmap → Bytes: {len(bitmap_bytes)} bytes")
    
    # Step 2: Transform to printer format (column-major, bottom-to-top)
    printer_format = transform_to_printer_format(bitmap_bytes, width, height)
    print(f"2. Transformed to printer format: {len(printer_format)} bytes")
    
    # Step 3: Calculate width split (in pixels/columns)
    BYTES_PER_COLUMN = height // 8  # 96 pixels = 12 bytes per column
    
    default_width = 85  # maximum width of each chunk
    remaining_width = width % 85
    number_of_chunks = math.ceil(width / 85)
    
    print(f"3. Splitting {width} columns into {number_of_chunks} chunks:")
    print(f"   Base width per chunk: {default_width} columns ({default_width * BYTES_PER_COLUMN} bytes)")
    
    # Step 4: Split bitmap by width (columns), then compress each chunk
    frames = []
    column_offset = 0
    total_compressed = 0
    
    print(f"\n4. Processing chunks (splitting by width):")

    frames_remaining = number_of_chunks
    
    for chunk_idx in range(number_of_chunks):
        frames_remaining -= 1

        # Calculate width for this chunk (distribute remainder evenly)
        chunk_width = default_width
        
        # Calculate byte range for this chunk
        # In column-major format: each column is BYTES_PER_COLUMN bytes
        byte_offset = column_offset * BYTES_PER_COLUMN
        chunk_bytes = chunk_width * BYTES_PER_COLUMN
        
        # Extract THIS chunk (vertical slice of columns)
        chunk_data = printer_format[byte_offset:byte_offset + chunk_bytes]
        
        # Compress THIS chunk independently
        try:
            compressed_chunk = minilzo.compress(chunk_data)
        except Exception as e:
            print(f"   Chunk {chunk_idx + 1} compression failed: {e}")
            return None
        
        total_compressed += len(compressed_chunk)
        
        frame_width = default_width if frames_remaining else remaining_width
        
        # Create BLE frame
        frame = create_ble_frame(
            compressed_chunk, 
            frames_remaining,
            frame_width
        )
        
        frames.append(frame)
        
        print(f"   Chunk {chunk_idx + 1}/{number_of_chunks}: "
              f"width={chunk_width} cols, "
              f"uncompressed={chunk_bytes}B → compressed={len(compressed_chunk)}B "
              f"({len(compressed_chunk)*100/chunk_bytes:.1f}%), "
              f"frame_total={len(frame)}B, "
              f"remaining={frames_remaining}")
        
        column_offset += chunk_width
    
    print(f"\n5. Summary:")
    print(f"   Total chunks: {number_of_chunks}")
    print(f"   Total columns processed: {column_offset}/{width}")
    print(f"   Total uncompressed: {len(printer_format)}B")
    print(f"   Total compressed: {total_compressed}B ({total_compressed*100/len(printer_format):.1f}%)")
    
    return frames

def create_ble_frame(compressed_chunk, frames_remaining, chunk_width):
    frame = bytearray()
    
    # Header starts with Magic number 0x66

    frame.append(0x66)
    length = 17 + len(compressed_chunk) + 1
    frame.append(length & 0xFF)
    frame.append((length >> 8) & 0xFF)
    frame.extend(PRINTER_ID)
    frame.append(IMAGE_WIDTH & 0xFF)
    frame.append((IMAGE_WIDTH >> 8) & 0xFF)
    frame.append(chunk_width)
    frame.append((frames_remaining >> 8) & 0xFF)
    frame.append(frames_remaining & 0xFF)

    # End of Header
    frame.append(0x00)
    
    frame.extend(compressed_chunk)
    
    # Checksum (last byte)
    checksum = calculate_checksum(frame + b'\x00')
    frame.append(checksum)
    
    return bytes(frame)

def generate_python_frames_output(frames, test_name):
    """Generate Python code with frames"""
    print(f"\n{'='*70}")
    print(f"Python code for: {test_name}")
    print(f"{'='*70}")
    print("all_frames = [")
    
    for i, frame in enumerate(frames):
        hex_string = ' '.join(f"{b:02X}" for b in frame)
        if i < len(frames) - 1:
            print(f'    "{hex_string}",')
        else:
            print(f'    "{hex_string}"')
    
    print("]")

def generate_esp32_code(frames, test_name):
    """Generate ESP32 C++ code"""
    print(f"\n{'='*70}")
    print(f"ESP32 code for: {test_name}")
    print(f"{'='*70}")
    
    for i, frame in enumerate(frames):
        print(f"\n// Frame {i+1} - {len(frame)} bytes")
        print("printFrames.push_back({")
        
        for j in range(0, len(frame), 16):
            line = frame[j:j+16]
            hex_str = ', '.join(f"0x{b:02X}" for b in line)
            if j + 16 >= len(frame):
                print(f"    {hex_str}")
            else:
                print(f"    {hex_str},")
        
        print("});")

def verify_roundtrip(frames, expected_width):
    print(f"\n{'='*70}")
    print("Verifying roundtrip (decompress frames)")
    print(f"{'='*70}")
    
    decompressed_data = bytearray()
    BYTES_PER_COLUMN = IMAGE_HEIGHT // 8
    
    for i, frame in enumerate(frames):
        compressed_chunk = frame[17:-1]
        
        try:
            dst_len = 9999 # weird API, max decompressed length has to be known beforehand or we fail.
            decompressed = minilzo.decompress(compressed_chunk, dst_len)
            decompressed_data.extend(decompressed)
            
            columns = len(decompressed) // BYTES_PER_COLUMN
            print(f"   Frame {i+1}: {len(compressed_chunk)}B → {len(decompressed)}B "
                  f"({columns} columns)")
        except Exception as e:
            print(f"   Frame {i+1}: Decompression failed - {e}")
            return False
    
    expected_size = (expected_width * IMAGE_HEIGHT) // 8
    
    if len(decompressed_data) == expected_size:
        print(f"\nSUCCESS! Decompressed {len(decompressed_data)} bytes "
              f"(expected {expected_size})")
        return True
    else:
        print(f"\nFAILED! Decompressed {len(decompressed_data)} bytes "
              f"(expected {expected_size})")
        return False

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("╔" + "="*68 + "╗")
    print("║" + "  MakeID L1 Printer - CORRECT Frame Generator".center(68) + "║")
    print("║" + "  Split FIRST, then compress each chunk".center(68) + "║")
    print("╚" + "="*68 + "╝")
    
    # Test configurations
    test_configs = [
        ("border", "Border Pattern"),
        ("all_white", "All White"),
        ("all_black", "All Black"),
        ("diagonal", "Diagonal Lines"),
    ]
    
    for pattern, name in test_configs:
        # Create bitmap
        bitmap = create_test_bitmap(IMAGE_WIDTH, IMAGE_HEIGHT, pattern)
        
        frames = compress_and_generate_frames(bitmap, name)
        
        if frames:
            # Verify roundtrip
            if verify_roundtrip(frames, IMAGE_WIDTH):
                # Generate code
                generate_python_frames_output(frames, name)
                generate_esp32_code(frames, name)
            
            print(f"\n{'='*70}\n")
        else:
            print(f"Failed to generate frames for {name}\n")
    
    print("╔" + "="*68 + "╗")
    print("║" + "  Generation Complete!".center(68) + "║")
    print("╚" + "="*68 + "╝")
