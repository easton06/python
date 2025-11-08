#!/usr/bin/env python3
import numpy as np
import minilzo

# === PRINTER SETTINGS ===
PRINTER_ID = bytes([0x1B, 0x2F, 0x03, 0x01, 0x00, 0x01, 0x00, 0x01])

def calculate_checksum(frame_bytes):
    checksum = 0
    for byte in frame_bytes[:-1]:
        checksum = (checksum - byte) & 0xFF
    return checksum

def create_test_bitmap(width, height, pattern="border"):
    """Create test bitmap with border pattern"""
    bitmap = np.zeros((height, width), dtype=np.uint8)
    if pattern == "border":
        border_size = min(5, height // 10, width // 10)
        bitmap[0:border_size, :] = 255
        bitmap[-border_size:, :] = 255
        bitmap[:, 0:border_size] = 255
        bitmap[:, -border_size:] = 255
    elif pattern == "all_white":
        bitmap[:] = 255
    elif pattern == "all_black":
        bitmap[:] = 0
    elif pattern == "checker":
        bitmap[:, ::2] = 255
        bitmap[::2, :] ^= 255
    return bitmap

def bitmap_to_bytes_optimized(bitmap):
    """Convert bitmap to 1bpp byte array"""
    height, width = bitmap.shape
    bytes_per_row = (width + 7) // 8
    out = bytearray()
    
    for y in range(height):
        for x in range(0, width, 8):
            byte_val = 0
            for bit in range(8):
                if x + bit < width and bitmap[y, x + bit] > 0:
                    byte_val |= (1 << (7 - bit))
            out.append(byte_val)
    
    return bytes(out)

def transform_to_printer_format_optimized(bitmap_bytes, width, height):
    """
    Transform bitmap to printer format
    """
    bytes_per_row = (width + 7) // 8
    total_bytes = height * bytes_per_row
    
    # Simple inversion (common in thermal printers)
    result = bytearray()
    for byte in bitmap_bytes:
        result.append(byte ^ 0xFF)  # Invert bits
    
    # Add the common header
    final_output = bytearray()
    final_output.extend(b'\x00\x02')  # Common header
    final_output.extend(result)
    
    return bytes(final_output)

def compress_and_generate_frames(bitmap, job_id, final_magic, image_name):
    """Compress bitmap and generate frames"""
    height, width = bitmap.shape
    print(f"\n=== Generating {image_name} ({width}x{height}) ===")
    
    # Convert to bytes
    bitmap_bytes = bitmap_to_bytes_optimized(bitmap)
    print(f"Original bitmap: {len(bitmap_bytes)} bytes")
    
    # Transform to printer format
    printer_format = transform_to_printer_format_optimized(bitmap_bytes, width, height)
    print(f"Transformed format: {len(printer_format)} bytes")
    
    # Compress using LZO
    try:
        compressed_data = minilzo.compress(printer_format)
        print(f"Compressed: {len(compressed_data)} bytes")
    except Exception as e:
        print(f"Compression failed: {e}")
        return None
    
    # Split into 4 chunks
    TARGET_FRAMES = 4
    chunks = []
    
    if len(compressed_data) <= TARGET_FRAMES:
        base_size = max(1, len(compressed_data) // TARGET_FRAMES)
        for i in range(TARGET_FRAMES):
            start = i * base_size
            end = start + base_size if i < TARGET_FRAMES - 1 else len(compressed_data)
            if start < len(compressed_data):
                chunks.append(compressed_data[start:end])
            else:
                chunks.append(b'')
    else:
        base_size = len(compressed_data) // TARGET_FRAMES
        for i in range(TARGET_FRAMES):
            start = i * base_size
            if i < TARGET_FRAMES - 1:
                end = start + base_size
            else:
                end = len(compressed_data)
            chunks.append(compressed_data[start:end])
    
    print(f"Split into {len(chunks)} chunks: {[len(c) for c in chunks]}")
    
    # Create frames
    frames = []
    for i, chunk in enumerate(chunks):
        is_final = (i == TARGET_FRAMES - 1)
        frames_remaining = TARGET_FRAMES - i - 1
        
        frame = bytearray()
        frame.append(0x66)  # Magic
        
        # Calculate length
        length = 16 + len(chunk) + 4
        frame.extend([length & 0xFF, (length >> 8) & 0xFF])
        
        # Printer ID
        frame.extend(PRINTER_ID)
        
        # Job ID (little-endian)
        frame.extend([job_id & 0xFF, (job_id >> 8) & 0xFF])
        
        # Frame magic
        frame_magic = final_magic if is_final else 0x55
        frame.append(frame_magic)
        
        # Remaining frames (big-endian)
        frame.extend([(frames_remaining >> 8) & 0xFF, frames_remaining & 0xFF])
        
        # Payload
        frame.extend(chunk)
        
        # End marker
        frame.extend([0x11, 0x00, 0x00])
        
        # Calculate checksum
        temp_frame = frame + b'\x00'
        checksum = calculate_checksum(temp_frame)
        frame.append(checksum)
        
        frames.append(bytes(frame))
        print(f"Frame {i+1}: {len(chunk)}B payload, {len(frame)}B total")
    
    return frames

def generate_esp32_code(frames, test_name):
    print(f"\n// ===========================================")
    print(f"// {test_name} - Generated with current algorithm")
    print(f"// ===========================================")
    print("std::vector<std::vector<uint8_t>> printFrames;")
    
    for i, frame in enumerate(frames):
        print(f"\n// Frame {i+1} - {len(frame)} bytes")
        print("printFrames.push_back({")
        
        # Format with proper indentation (16 bytes per line)
        hex_lines = []
        for j in range(0, len(frame), 16):
            line = frame[j:j+16]
            hex_str = ', '.join(f"0x{b:02X}" for b in line)
            if j + 16 >= len(frame):
                hex_lines.append(f"    {hex_str}")
            else:
                hex_lines.append(f"    {hex_str},")
        
        print('\n'.join(hex_lines))
        print("});")
    
    print(f"\n// Total: {len(frames)} frames")

def debug_frame_details(frames):
    print(f"\n=== Frame Details ===")
    for i, frame in enumerate(frames):
        payload = frame[16:-4]  # Extract payload
        print(f"Frame {i+1}:")
        print(f"  Total: {len(frame)} bytes")
        print(f"  Payload: {len(payload)} bytes")
        print(f"  Payload hex: {payload.hex()}")
        print(f"  Checksum: 0x{frame[-1]:02X}")

def generate_frames_string_format(frames, test_name):
    print(f"all_frames = [")
    
    frame_strings = []
    for i, frame in enumerate(frames):
        # Convert frame to space-separated hex string
        hex_string = ' '.join(f"{b:02X}" for b in frame)
        frame_strings.append(f'    "{hex_string}"')
    
    # Join with newlines and commas
    print(',\n'.join(frame_strings))
    print("]")

# Test configurations
test_configs = [
    (384, 96, "border", 0x012B, 0x2C, "Border_384x96"),
    # (384, 96, "all_white", 0x012B, 0x2C, "White_384x96"), 
    # (384, 96, "all_black", 0x012B, 0x2C, "Black_384x96"),
    # (384, 96, "checker", 0x012B, 0x2C, "Checker_384x96"),
]

if __name__ == "__main__":
    print("MakeID L1 Printer - ESP32 Code Generator")
    print("Using current compression algorithm")
    
    for width, height, pattern, job_id, final_magic, name in test_configs:
        # Create bitmap
        bitmap = create_test_bitmap(width, height, pattern)
        
        # Generate frames using current algorithm
        frames = compress_and_generate_frames(bitmap, job_id, final_magic, name)
        
        if frames:
            # Generate ESP32 code
            generate_esp32_code(frames, f"{name} (JobID: 0x{job_id:04X})")
            generate_frames_string_format(frames, f"{name} (JobID: 0x{job_id:04X})")
            
            # Show frame details for debugging
            debug_frame_details(frames)
            
            print(f"\n{'='*60}")
