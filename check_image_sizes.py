import os
from PIL import Image

def get_min_max_sizes(split_dir):
    image_dir = os.path.join("kvasir-seg", split_dir, "images")
    if not os.path.exists(image_dir):
        return None, None, 0
        
    min_w, min_h = float('inf'), float('inf')
    max_w, max_h = 0, 0
    min_area = float('inf')
    max_area = 0
    
    min_area_size = (0, 0)
    max_area_size = (0, 0)
    
    count = 0
    
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            filepath = os.path.join(image_dir, filename)
            try:
                with Image.open(filepath) as img:
                    w, h = img.size
                    
                    # Track absolute min/max for width and height individually
                    if w < min_w: min_w = w
                    if h < min_h: min_h = h
                    if w > max_w: max_w = w
                    if h > max_h: max_h = h
                    
                    # Also track min/max based on area to give a definitive "smallest" and "largest" resolution
                    area = w * h
                    if area < min_area:
                        min_area = area
                        min_area_size = (w, h)
                    if area > max_area:
                        max_area = area
                        max_area_size = (w, h)
                        
                    count += 1
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
                
    if count == 0:
        return None, None, 0
        
    return (min_w, min_h), (max_w, max_h), min_area_size, max_area_size, count

def main():
    splits = ["train", "validation", "test"]
    
    print("Kvasir-SEG Dataset Image Size Statistics\n" + "="*40)
    for split in splits:
        print(f"Split: {split.capitalize()}")
        res = get_min_max_sizes(split)
        if res[0] is None:
            print("  -> Directory not found or no images.\n")
        else:
            (min_w, min_h), (max_w, max_h), min_area_size, max_area_size, count = res
            print(f"  Total Images: {count}")
            print(f"  Smallest Dimensions (min width, min height): {min_w} x {min_h}")
            print(f"  Largest Dimensions (max width, max height): {max_w} x {max_h}")
            print(f"  Smallest Resolution (W x H): {min_area_size[0]} x {min_area_size[1]}")
            print(f"  Largest Resolution (W x H): {max_area_size[0]} x {max_area_size[1]}\n")

if __name__ == "__main__":
    main()
