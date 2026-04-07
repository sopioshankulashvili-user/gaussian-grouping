# #rename png files in /data/sopio/small_city_50/25/object_masks to remove the first mask_ from the filename
# import os

# for filename in os.listdir("/data/sopio/small_city_50/25/object_mask_original"):
#     # go through images and remove mask_ prefix from filename and rename the file to exclude _png* suffix
#     if filename.endswith(".png"):
#         # new_filename = filename[len("mask_"):]  # Remove the "mask_" prefix
#         new_filename = filename.split("_png")[0] + ".png"  # Remove the "_png*" suffix
#         os.rename(os.path.join("/data/sopio/small_city_50/25/object_mask_original", filename), 
#                   os.path.join("/data/sopio/small_city_50/25/object_mask_original", new_filename))

# # chsnge resolution of the images in this folder "/data/sopio/small_city_50/25/object_mask as follows
# # resolution = round(orig_w/(2)), round(orig_h/(2))
# # resized_image_rgb = PILtoTorch(image, resolution)


# import os
# from PIL import Image
# import torch
# import numpy as np

# # Paths
# input_dir = "/data/sopio/small_city_50/25/object_mask_original"
# output_dir = "/data/sopio/small_city_50/25/object_mask" # Recommended to save to a new folder first

# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# print(f"Resizing masks in {input_dir}...")

# for filename in os.listdir(input_dir):
#     if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.npy')):
#         img_path = os.path.join(input_dir, filename)
        
#         # Load image
#         img = Image.open(img_path)
#         orig_w, orig_h = img.size
        
#         # Calculate new resolution (scale by 2 as per your request)
#         # Note: In PIL, resolution is (width, height)
#         new_resolution = (round(orig_w / 1), round(orig_h / 1))
        
#         # Resize using NEAREST to preserve class IDs
#         resized_img = img.resize(new_resolution, resample=Image.NEAREST)
        
#         # Save
#         resized_img.save(os.path.join(output_dir, filename))

# print(f"Done! Resized masks are in: {output_dir}")


# #in the output_dir replace pixel values 255 with 1s
# for filename in os.listdir(output_dir):
#     if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.npy')):
#         img_path = os.path.join(output_dir, filename)
        
#         # Load image
#         img = Image.open(img_path)
#         img_array = torch.from_numpy(np.array(img))
        
#         # Replace 255 with 1
#         img_array[img_array == 255] = 1
        
#         # Save the modified image
#         modified_img = Image.fromarray(img_array.numpy())
#         modified_img.save(os.path.join(output_dir, filename))

# print(f"Done! Pixel values replaced in: {output_dir}")