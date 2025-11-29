#!/usr/bin/env python3
import numpy as np
import minilzo
import math

# === PRINTER SETTINGS ===
PRINTER_CONSTANT = bytes([0x1B, 0x2F, 0x03, 0x01, 0x00, 0x01, 0x00, 0x01])
IMAGE_WIDTH = 291
IMAGE_HEIGHT = 96

def calculate_checksum(frame_bytes):
    checksum = 0
    for byte in frame_bytes[:-1]:
        checksum = (checksum - byte) & 0xFF
    return checksum

def create_test_bitmap(width, height, pattern="border", text=None, text_x=10, text_y=10):
    bitmap = np.zeros((height, width), dtype=np.uint8)  # Start all white (0)

    if pattern == "border":
        border_size = 5
        bitmap[0:border_size, :] = 1                    # Top border
        bitmap[-border_size:, :] = 1                    # Bottom border  
        bitmap[:, 0:border_size] = 1                    # Left border
        bitmap[:, -border_size:] = 1                    # Right border

    elif pattern == "all_white":
        bitmap[:] = 0  # All white (0)
    elif pattern == "all_black":
        bitmap[:] = 1  # All black (1)
    elif pattern == "diagonal":
        for i in range(min(width, height)):
            bitmap[i, i] = 1  # Main diagonal 
            bitmap[i, width - 1 - i] = 1  # Anti-diagonal

    elif pattern == "stripes":
        stripe_width = 10
        for x in range(width):
            if (x // stripe_width) % 2 == 0:
                bitmap[:, x] = 1  # Black stripes

    elif pattern == "horizontal_stripes":
        stripe_height = 8
        for y in range(height):
            if (y // stripe_height) % 2 == 0:
                bitmap[y, :] = 1  # Black stripes

    elif pattern == "checkerboard":
        square_size = 10
        for y in range(height):
            for x in range(width):
                if ((x // square_size) + (y // square_size)) % 2 == 0:
                    bitmap[y, x] = 1  # Black squares
    
    elif pattern == "text" and text:
        # Create text bitmap once
        text_bmp = create_text_bitmap(text)
        text_height, text_width = text_bmp.shape
        
        # Repeat text across the entire bitmap with specified spacing
        for start_y in range(0, height, text_y + text_height):
            for start_x in range(0, width, text_x + text_width):
                # Copy text onto the main bitmap at this position
                for y in range(text_height):
                    for x in range(text_width):
                        pixel_y = start_y + y
                        pixel_x = start_x + x
                        if (pixel_y < height and pixel_x < width and 
                            text_bmp[y, x] == 1):
                            bitmap[pixel_y, pixel_x] = 1  # Set pixel to black for text

    elif pattern == "letter" and text:
        # Create text bitmap once for the entire string
        text_bmp = create_text_bitmap(text)
        text_height, text_width = text_bmp.shape
        
        # Repeat the entire text string across the bitmap
        for start_y in range(0, height, text_y + text_height):
            for start_x in range(0, width, text_x + text_width):
                # Copy the entire text string onto the main bitmap at this position
                for y in range(text_height):
                    for x in range(text_width):
                        pixel_y = start_y + y
                        pixel_x = start_x + x
                        if (pixel_y < height and pixel_x < width and 
                            text_bmp[y, x] == 1):
                            bitmap[pixel_y, pixel_x] = 1  # Set pixel to black for text

    return bitmap

def create_letter_bitmap(letter):
    """Create a bitmap for a single letter"""
    # Simple 5x7 font for uppercase letters
    font = {
        'A': [0x0E, 0x11, 0x1F, 0x11, 0x11],
        'B': [0x1E, 0x11, 0x1E, 0x11, 0x1E],
        'C': [0x0E, 0x11, 0x10, 0x11, 0x0E],
        'D': [0x1E, 0x11, 0x11, 0x11, 0x1E],
        'E': [0x1F, 0x10, 0x1E, 0x10, 0x1F],
        'F': [0x1F, 0x10, 0x1E, 0x10, 0x10],
        'G': [0x0E, 0x11, 0x13, 0x11, 0x0F],
        'H': [0x11, 0x11, 0x1F, 0x11, 0x11],
        'I': [0x1F, 0x04, 0x04, 0x04, 0x1F],
        'J': [0x07, 0x02, 0x02, 0x12, 0x0C],
        'K': [0x11, 0x12, 0x1C, 0x12, 0x11],
        'L': [0x10, 0x10, 0x10, 0x10, 0x1F],
        'M': [0x11, 0x1B, 0x15, 0x11, 0x11],
        'N': [0x11, 0x19, 0x15, 0x13, 0x11],
        'O': [0x0E, 0x11, 0x11, 0x11, 0x0E],
        'P': [0x1E, 0x11, 0x1E, 0x10, 0x10],
        'Q': [0x0E, 0x11, 0x11, 0x15, 0x0F],
        'R': [0x1E, 0x11, 0x1E, 0x14, 0x13],
        'S': [0x0F, 0x10, 0x0E, 0x01, 0x1E],
        'T': [0x1F, 0x04, 0x04, 0x04, 0x04],
        'U': [0x11, 0x11, 0x11, 0x11, 0x0E],
        'V': [0x11, 0x11, 0x11, 0x0A, 0x04],
        'W': [0x11, 0x11, 0x15, 0x15, 0x0A],
        'X': [0x11, 0x0A, 0x04, 0x0A, 0x11],
        'Y': [0x11, 0x0A, 0x04, 0x04, 0x04],
        'Z': [0x1F, 0x02, 0x04, 0x08, 0x1F],
    }
    
    if letter.upper() not in font:
        # Return empty 5x7 bitmap for unknown characters
        return np.zeros((7, 5), dtype=np.uint8)
    
    char_data = font[letter.upper()]
    bitmap = np.zeros((7, 5), dtype=np.uint8)
    
    for row in range(5):  # 5 rows of font data
        for col in range(5):  # 5 columns
            if char_data[row] & (1 << (4 - col)):
                bitmap[row + 1, col] = 1  # Offset by 1 row for better centering
    
    return bitmap

def create_text_bitmap(text, spacing=1):
    """Create a bitmap for text string"""
    letters = []
    max_height = 7  # 5x7 font + 1 pixel top/bottom padding
    
    # Create bitmaps for each letter
    for char in text:
        if char == ' ':
            # Add space (empty column)
            letters.append(np.zeros((max_height, 3), dtype=np.uint8))
        else:
            letters.append(create_letter_bitmap(char))
    
    # Calculate total width
    total_width = sum(letter.shape[1] for letter in letters) + (len(letters) - 1) * spacing
    
    # Create final bitmap
    result = np.zeros((max_height, total_width), dtype=np.uint8)
    
    # Composite letters
    x_offset = 0
    for letter in letters:
        height, width = letter.shape
        result[0:height, x_offset:x_offset + width] = letter
        x_offset += width + spacing
    
    return result

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
    3. Flip horizontally
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
    
    # Step 2: 16-bit word swap WRONG LOGIC
    # for i in range(0, total_bytes, 2):
    #     if i + 1 < total_bytes:
    #         result[i], result[i + 1] = result[i + 1], result[i]
    
    # Step 3: Invert all bits (0xFF = black for thermal printer) WRONG logic
    # for i in range(total_bytes):
    #     result[i] ^= 0xFF

    # Step 3: Flip Horizontally (since the printer format is drawn from bottom right to top left)
    flipped_result = bytearray(total_bytes)

    for x in range(width):
        src_col = x
        dst_col = width - 1 - x
        src_start = src_col * BYTES_PER_COLUMN
        dst_start = dst_col * BYTES_PER_COLUMN
        flipped_result[dst_start:dst_start + BYTES_PER_COLUMN] = result[src_start:src_start + BYTES_PER_COLUMN]
    
    return bytes(flipped_result)

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
    
    for chunk_idx in range(number_of_chunks-1, -1, -1):
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
        
        # inversely
        frame_width = remaining_width if chunk_idx + 1 == number_of_chunks else default_width
        frames_remaining = number_of_chunks - chunk_idx - 1
        
        # Create BLE frame
        frame = create_ble_frame(
            compressed_chunk,
            frames_remaining,
            frame_width
        )
        
        frames.insert(0, frame)
        
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
    frame.extend(PRINTER_CONSTANT)
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

def visualize_bitmap(bitmap, max_width=80, max_height=40):
    """Simple ASCII visualization of the bitmap"""
    height, width = bitmap.shape
    scale_x = width / max_width
    scale_y = height / max_height
    
    print("Bitmap Visualization:")
    for y in range(0, min(height, max_height), max(1, int(scale_y))):
        line = ""
        for x in range(0, min(width, max_width), max(1, int(scale_x))):
            if bitmap[y, x] > 0:
                line += "█"  # Black pixel
            else:
                line += " "  # White pixel
        print(line)

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
        ("stripes", "Vertical Stripes"),
        ("horizontal_stripes", "Horizontal Stripes"),
        ("checkerboard", "Checkerboard"),
        ("text", "Text Hello", "HELLO"),  # New: Text pattern
        ("letter", "Letters ABC", "ABC123"),  # New: Individual letters
    ]
    
    for config in test_configs:
        pattern = config[0]
        name = config[1]
        # Create bitmap
        if pattern in ["text", "letter"] and len(config) > 2:
            text_param = config[2]
            bitmap = create_test_bitmap(IMAGE_WIDTH, IMAGE_HEIGHT, pattern, text=text_param, text_x=10, text_y=10)
        else:
            bitmap = create_test_bitmap(IMAGE_WIDTH, IMAGE_HEIGHT, pattern)
        
        frames = compress_and_generate_frames(bitmap, name)
        
        if frames:
            # Verify roundtrip
            if verify_roundtrip(frames, IMAGE_WIDTH):
                # Generate code
                generate_python_frames_output(frames, name)
                generate_esp32_code(frames, name)
                visualize_bitmap(bitmap)
            
            print(f"\n{'='*70}\n")
        else:
            print(f"Failed to generate frames for {name}\n")

    print("╔" + "="*68 + "╗")
    print("║" + "  Generation Complete!".center(68) + "║")
    print("╚" + "="*68 + "╝")
