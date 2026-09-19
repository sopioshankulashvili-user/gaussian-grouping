import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
import lpips
from tqdm import tqdm

# --- Configuration ---
# Map dataset directories to their respective plot labels
DATASETS = {
    # "output/milan/2/train/ours_7000": "univrses_1",
    # "output/milan/3/train/ours_7000": "univrses_2",
    # "output/milan/4/train/ours_7000": "univrses_3",
    # "output/small_city_50/25/train/ours_7000": "smallcity_1",
    # "output/small_city_50/crack_reduced/train/ours_7000": "smallcity_2"
    "output/milan/2/test/ours_7000": "univrses_1",
    "output/milan/3/test/ours_7000": "univrses_2",
    "output/milan/4/test/ours_7000": "univrses_3",
    "output/small_city_50/25/test/ours_7000": "smallcity_1",
    "output/small_city_50/crack_reduced/test/ours_7000": "smallcity_2"
}

# Set up PyTorch device for LPIPS
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize LPIPS model (AlexNet is standard for perceptual similarity)
loss_fn_alex = lpips.LPIPS(net='alex').to(device)

def load_image(path):
    """Loads an image in RGB format."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def image_to_tensor(img):
    """Converts a [0, 255] numpy image to a [-1, 1] PyTorch tensor for LPIPS."""
    # Convert to float and scale to [-1, 1]
    img_float = (img.astype(np.float32) / 255.0) * 2.0 - 1.0
    # Change shape from HWC to CHW and add batch dimension
    tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)

def compute_miou(pred_img, gt_img):
    """
    Computes Mean Intersection over Union (mIoU) for color-coded mask images.
    Converts RGB to a single 24-bit integer to easily identify unique classes.
    """
    # Compress RGB channels into a single 1D integer array per pixel
    pred_1d = pred_img[:, :, 0].astype(np.int32) * 65536 + pred_img[:, :, 1].astype(np.int32) * 256 + pred_img[:, :, 2].astype(np.int32)
    gt_1d = gt_img[:, :, 0].astype(np.int32) * 65536 + gt_img[:, :, 1].astype(np.int32) * 256 + gt_img[:, :, 2].astype(np.int32)
    
    unique_classes = np.unique(gt_1d)
    ious = []
    
    for c in unique_classes:
        intersection = np.sum((pred_1d == c) & (gt_1d == c))
        union = np.sum((pred_1d == c) | (gt_1d == c))
        if union > 0:
            ious.append(intersection / union)
            
    return np.mean(ious) if ious else 0.0

def main():
    # Dictionaries to store results and image counts for plotting
    all_metrics = {
        'PSNR': {label: [] for label in DATASETS.values()},
        'SSIM': {label: [] for label in DATASETS.values()},
        'LPIPS': {label: [] for label in DATASETS.values()},
        'mIoU': {label: [] for label in DATASETS.values()}
    }
    
    image_counts_quality = {label: 0 for label in DATASETS.values()}
    image_counts_miou = {label: 0 for label in DATASETS.values()}

    for path, label in DATASETS.items():
        print(f"\n--- Processing Dataset: {label} ({path}) ---")
        
        # Paths
        gt_dir = os.path.join(path, "gt")
        render_dir = os.path.join(path, "renders")
        gt_obj_dir = os.path.join(path, "gt_objects_color")
        pred_obj_dir = os.path.join(path, "objects_pred")
        
        # Ensure GT directory exists to avoid crashes
        if not os.path.exists(gt_dir):
            print(f"Directory missing: {gt_dir}. Skipping dataset {label}.")
            continue

        filenames = os.listdir(gt_dir)
        
        for filename in tqdm(filenames, desc=f"Evaluating {label}"):
            gt_path = os.path.join(gt_dir, filename)
            render_path = os.path.join(render_dir, filename)
            
            # 1. Compute Image Quality Metrics
            if os.path.exists(render_path):
                gt_img = load_image(gt_path)
                render_img = load_image(render_path)
                
                # Normalize to [0, 1] for PSNR and SSIM
                gt_norm = gt_img.astype(np.float32) / 255.0
                render_norm = render_img.astype(np.float32) / 255.0
                
                psnr_val = compute_psnr(gt_norm, render_norm, data_range=1.0)
                ssim_val = compute_ssim(gt_norm, render_norm, data_range=1.0, channel_axis=2)
                
                # Compute LPIPS
                gt_tensor = image_to_tensor(gt_img)
                render_tensor = image_to_tensor(render_img)
                with torch.no_grad():
                    lpips_val = loss_fn_alex(render_tensor, gt_tensor).item()
                
                all_metrics['PSNR'][label].append(psnr_val)
                all_metrics['SSIM'][label].append(ssim_val)
                all_metrics['LPIPS'][label].append(lpips_val)
                
                image_counts_quality[label] += 1
            
            # 2. Compute mIoU (if object mask directories exist)
            gt_obj_path = os.path.join(gt_obj_dir, filename)
            pred_obj_path = os.path.join(pred_obj_dir, filename)
            
            if os.path.exists(gt_obj_path) and os.path.exists(pred_obj_path):
                gt_obj_img = load_image(gt_obj_path)
                pred_obj_img = load_image(pred_obj_path)
                
                miou_val = compute_miou(pred_obj_img, gt_obj_img)
                all_metrics['mIoU'][label].append(miou_val)
                
                image_counts_miou[label] += 1

        # Print summaries for the current dataset
        print(f"Results for {label}:")
        for metric in ['PSNR', 'SSIM', 'LPIPS', 'mIoU']:
            if all_metrics[metric][label]:
                mean_val = np.mean(all_metrics[metric][label])
                print(f"  {metric}: {mean_val:.4f}")

    # --- Plotting Image Quality Metrics (PSNR, SSIM, LPIPS) ---
    print("\nGenerating image quality plots...")
    fig1, axes1 = plt.subplots(1, 3, figsize=(16, 6))

    quality_configs = [
        ('PSNR', axes1[0], 'higher is better'),
        ('SSIM', axes1[1], 'higher is better'),
        ('LPIPS', axes1[2], 'lower is better')
    ]

    for metric_name, ax, metric_label in quality_configs:
        valid_labels = [label for label in DATASETS.values() if all_metrics[metric_name][label]]
        data_to_plot = [all_metrics[metric_name][label] for label in valid_labels]
        
        # Format X labels with image counts under the subset ID
        x_labels = [f"{label}\n({image_counts_quality[label]} images)" for label in valid_labels]
        
        if data_to_plot:
            ax.boxplot(data_to_plot, labels=x_labels, patch_artist=True)
            ax.set_title(f'{metric_name} ({metric_label})')
            ax.set_xlabel('Subset ID')
            ax.set_ylabel(metric_name)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            ax.tick_params(axis='x', rotation=45)
        else:
            ax.set_title(f'{metric_name} (No Data Available)')
            ax.axis('off')

    plt.tight_layout()
    quality_filename = 'combined_datasets_quality_metrics.png'
    fig1.savefig(quality_filename, dpi=300, bbox_inches='tight')
    print(f"Quality plots saved to {quality_filename}")

    # --- Plotting Semantic Metric (mIoU) ---
    print("Generating semantic evaluation plot...")
    fig2, ax2 = plt.subplots(figsize=(8, 6))

    valid_miou_labels = [label for label in DATASETS.values() if all_metrics['mIoU'][label]]
    miou_data = [all_metrics['mIoU'][label] for label in valid_miou_labels]
    
    # Format X labels for mIoU (in case some datasets lack semantic masks)
    x_labels_miou = [f"{label}\n({image_counts_miou[label]} images)" for label in valid_miou_labels]

    if miou_data:
        ax2.boxplot(miou_data, labels=x_labels_miou, patch_artist=True)
        ax2.set_title('mIoU (higher is better)')
        ax2.set_xlabel('Subset ID')
        ax2.set_ylabel('mIoU')
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
        ax2.tick_params(axis='x', rotation=45)
    else:
        ax2.set_title('mIoU (No Data Available)')
        ax2.axis('off')

    plt.tight_layout()
    miou_filename = 'combined_datasets_miou_evaluation.png'
    fig2.savefig(miou_filename, dpi=300, bbox_inches='tight')
    print(f"Semantic plot saved to {miou_filename}")


if __name__ == "__main__":
    main()