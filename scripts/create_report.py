
import numpy as np
import scipy.io as sio
import os
import h5py
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def get_noisy_patch(data_folder, img_id, box_id):
    if 'info_file' not in get_noisy_patch.__dict__:
        get_noisy_patch.info_file = h5py.File(os.path.join(data_folder, 'info.mat'), 'r')
    
    info = get_noisy_patch.info_file['info']
    bb = info['boundingboxes']
    
    filename_original = os.path.join(data_folder, 'images_srgb', f'{img_id:04d}.mat')
    with h5py.File(filename_original, 'r') as img_original_h5:
        Inoisy_full = np.float32(np.array(img_original_h5['InoisySRGB']).T)

    ref = bb[0][img_id-1]
    boxes = np.array(info[ref]).T
    idx = [int(boxes[box_id-1,0]-1), int(boxes[box_id-1,2]), int(boxes[box_id-1,1]-1), int(boxes[box_id-1,3])]
    noisy_patch = Inoisy_full[idx[0]:idx[1], idx[2]:idx[3], :].copy()
    return np.clip(noisy_patch, 0, 1)

def evaluate_and_plot_metrics(data_folder, filter_outputs, img_id, box_id):
    print(f"\nĐang tính toán và vẽ biểu đồ cho ảnh {img_id}, vùng {box_id}...")
    
    reference_patch = get_noisy_patch(data_folder, img_id, box_id)
    filter_names = list(filter_outputs.keys())
    psnr_scores, ssim_scores = [], []

    for name, folder in filter_outputs.items():
        filepath = os.path.join(folder, f'{img_id:04d}_{box_id:02d}.mat')
        if os.path.exists(filepath):
            denoised_data = sio.loadmat(filepath)
            denoised_patch = np.clip(denoised_data['Idenoised_crop'], 0, 1)
            psnr_val = psnr(reference_patch, denoised_patch, data_range=1)
            ssim_val = ssim(reference_patch, denoised_patch, data_range=1, channel_axis=2, win_size=7)
            psnr_scores.append(psnr_val)
            ssim_scores.append(ssim_val)
        else:
            psnr_scores.append(0)
            ssim_scores.append(0)

    x = np.arange(len(filter_names))
    width = 0.35
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle(f'So sánh các bộ lọc cho Ảnh {img_id} - Vùng {box_id}\n(So với ảnh gốc bị nhiễu)', fontsize=16)

    ax1.bar(x, psnr_scores, width, label='PSNR', color='skyblue')
    ax1.set_ylabel('PSNR (dB)'); ax1.set_title('Tỷ số tín hiệu trên nhiễu đỉnh (PSNR)'); ax1.set_xticks(x); ax1.set_xticklabels(filter_names); ax1.legend()
    ax2.bar(x, ssim_scores, width, label='SSIM', color='salmon')
    ax2.set_ylabel('SSIM'); ax2.set_title('Chỉ số tương đồng cấu trúc (SSIM)'); ax2.set_xticks(x); ax2.set_xticklabels(filter_names); ax2.legend(); ax2.set_ylim(0, 1)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    chart_filename = os.path.join('F:\\', f'metrics_comparison_img{img_id}_box{box_id}.png')
    plt.savefig(chart_filename)
    print(f"Đã lưu biểu đồ so sánh vào file: {chart_filename}")
    plt.show()

def create_report_visualization(data_folder, filter_outputs, img_ids_to_show, box_ids_to_show, save_path):
    print("\nĐang tạo visualization tổng hợp cho báo cáo...")
    
    num_images, num_boxes, num_filters = len(img_ids_to_show), len(box_ids_to_show), len(filter_outputs)
    num_cols, num_rows = 1 + num_filters, num_images * num_boxes
    
    if num_rows == 1:
        fig, axes = plt.subplots(1, num_cols, figsize=(num_cols * 4, num_rows * 4))
    else:
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 4, num_rows * 4))
    
    fig.suptitle('Tổng hợp kết quả các bộ lọc Fourier', fontsize=24, y=1.0)

    col_titles = ['Ảnh Gốc Bị Nhiễu'] + list(filter_outputs.keys())
    
    axes_for_titles = axes[0] if num_rows > 1 else axes
    for ax, col_title in zip(axes_for_titles, col_titles):
        ax.set_title(col_title, fontsize=16, pad=20)

    current_row = 0
    for img_id in img_ids_to_show:
        for box_id in box_ids_to_show:
            if current_row >= num_rows: break
            
            row_axes = axes[current_row, :] if num_rows > 1 else axes

            noisy_patch = get_noisy_patch(data_folder, img_id, box_id)
            row_axes[0].imshow(noisy_patch)
            row_axes[0].axis('off')
            row_axes[0].text(-0.1, 0.5, f'Ảnh {img_id}\nVùng {box_id}', transform=row_axes[0].transAxes, ha="right", va="center", fontsize=14, rotation=90)

            for i, (name, folder) in enumerate(filter_outputs.items()):
                ax = row_axes[i + 1]
                filepath = os.path.join(folder, f'{img_id:04d}_{box_id:02d}.mat')
                if os.path.exists(filepath):
                    denoised_data = sio.loadmat(filepath)
                    denoised_crop = np.clip(denoised_data['Idenoised_crop'], 0, 1)
                    ax.imshow(denoised_crop)
                else:
                    ax.text(0.5, 0.5, 'Không tìm thấy file', ha='center', va='center')
                ax.axis('off')
            
            current_row += 1

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"Đã lưu ảnh tổng hợp báo cáo vào: {save_path}")
    plt.close(fig)

    if 'info_file' in get_noisy_patch.__dict__:
        get_noisy_patch.info_file.close()
        del get_noisy_patch.info_file

if __name__ == '__main__':
    data_folder = r'F:\dnd_2017'
    
    output_folders = {
        'Low-Pass Filter': r'F:\output_low_pass',
        'High-Pass Filter': r'F:\output_high_pass',
        'Band-Pass Filter': r'F:\output_band_pass',
        'Band-Stop Filter': r'F:\output_band_stop'
    }

    all_folders_exist = True
    for folder in output_folders.values():
        if not os.path.isdir(folder):
            print(f"LỖI: Thư mục kết quả '{folder}' không tồn tại. Vui lòng chạy file 'run_denoising.py' trước.")
            all_folders_exist = False
            break

    if all_folders_exist:
        print("Tất cả các thư mục kết quả đã tồn tại. Bắt đầu tạo báo cáo...")
        evaluate_and_plot_metrics(data_folder, output_folders, img_id=1, box_id=5)
        
        report_images = [1, 7, 25]
        report_boxes = [5, 12]
        report_save_path = r'F:\Fourier_Filters_Report_Final.png'
        
        create_report_visualization(data_folder, 
                                    output_folders, 
                                    report_images, 
                                    report_boxes, 
                                    report_save_path)