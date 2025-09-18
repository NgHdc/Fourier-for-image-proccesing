
import numpy as np
import scipy.io as sio
import os
import h5py
import cv2


def apply_fourier_filter(noisy_patch, mask):
    rows, cols, channels = noisy_patch.shape
    denoised_patch = np.zeros_like(noisy_patch, dtype=np.float32)

    for i in range(channels):
        channel_noisy = noisy_patch[:, :, i]
        f = np.fft.fft2(channel_noisy)
        fshift = np.fft.fftshift(f)
        fshift_filtered = fshift * mask
        f_ishift = np.fft.ifftshift(fshift_filtered)
        channel_denoised = np.fft.ifft2(f_ishift)
        denoised_patch[:, :, i] = np.abs(channel_denoised)
        
    return denoised_patch

def low_pass_denoiser(noisy_patch, noise_info):
    rows, cols, _ = noisy_patch.shape
    crow, ccol = int(rows / 2), int(cols / 2)
    radius = 40
    mask = np.zeros((rows, cols), np.uint8)
    cv2.circle(mask, (ccol, crow), radius, 1, thickness=-1)
    return apply_fourier_filter(noisy_patch, mask)

def high_pass_denoiser(noisy_patch, noise_info):
    rows, cols, _ = noisy_patch.shape
    crow, ccol = int(rows / 2), int(cols / 2)
    radius = 40
    mask = np.ones((rows, cols), np.uint8)
    cv2.circle(mask, (ccol, crow), radius, 0, thickness=-1)
    return apply_fourier_filter(noisy_patch, mask)

def band_pass_denoiser(noisy_patch, noise_info):
    rows, cols, _ = noisy_patch.shape
    crow, ccol = int(rows / 2), int(cols / 2)
    r_outer = 80
    r_inner = 20
    mask = np.zeros((rows, cols), np.uint8)
    cv2.circle(mask, (ccol, crow), r_outer, 1, thickness=-1)
    cv2.circle(mask, (ccol, crow), r_inner, 0, thickness=-1)
    return apply_fourier_filter(noisy_patch, mask)

def band_stop_denoiser(noisy_patch, noise_info):
    rows, cols, _ = noisy_patch.shape
    crow, ccol = int(rows / 2), int(cols / 2)
    r_outer = 80
    r_inner = 20
    mask = np.ones((rows, cols), np.uint8)
    cv2.circle(mask, (ccol, crow), r_outer, 0, thickness=-1)
    cv2.circle(mask, (ccol, crow), r_inner, 1, thickness=-1)
    return apply_fourier_filter(noisy_patch, mask)


def load_nlf(info, img_id):
    nlf = {}
    nlf_h5 = info[info["nlf"][0][img_id]]
    nlf["a"] = nlf_h5["a"][0][0]
    nlf["b"] = nlf_h5["b"][0][0]
    return nlf

def load_sigma_srgb(info, img_id, bb):
    nlf_h5 = info[info["sigma_srgb"][0][img_id]]
    sigma = nlf_h5[0,bb]
    return sigma

def denoise_srgb(denoiser, data_folder, out_folder):
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    print(f'Bắt đầu quá trình cho thư mục: {out_folder}')
    infos = h5py.File(os.path.join(data_folder, 'info.mat'), 'r')
    info = infos['info']
    bb = info['boundingboxes']
    print('Tải file info.mat thành công.\n')
    for i in range(50):
        filename = os.path.join(data_folder, 'images_srgb', '%04d.mat'%(i+1))
        img = h5py.File(filename, 'r')
        Inoisy = np.float32(np.array(img['InoisySRGB']).T)
        ref = bb[0][i]
        boxes = np.array(info[ref]).T
        for k in range(20):
            idx = [int(boxes[k,0]-1), int(boxes[k,2]), int(boxes[k,1]-1), int(boxes[k,3])]
            Inoisy_crop = Inoisy[idx[0]:idx[1], idx[2]:idx[3], :].copy()
            nlf = load_nlf(info, i)
            nlf["sigma"] = load_sigma_srgb(info, i, k)
            Idenoised_crop = denoiser(Inoisy_crop, nlf)
            save_file = os.path.join(out_folder, '%04d_%02d.mat'%(i+1, k+1))
            sio.savemat(save_file, {'Idenoised_crop': Idenoised_crop})
        print(f'--- HOÀN THÀNH ẢNH {i+1:02d}/50 ---')
    print(f'*** Quá trình cho {out_folder} hoàn tất! ***\n')

if __name__ == '__main__':
    data_folder = r'F:\dnd_2017' 

    filters_to_run = {
        'Low-Pass Filter': (low_pass_denoiser, r'F:\output_low_pass'),
        'High-Pass Filter': (high_pass_denoiser, r'F:\output_high_pass'),
        'Band-Pass Filter': (band_pass_denoiser, r'F:\output_band_pass'),
        'Band-Stop Filter': (band_stop_denoiser, r'F:\output_band_stop')
    }

    if not os.path.isdir(data_folder) or not os.path.exists(os.path.join(data_folder, 'info.mat')):
        print(f"LỖI: Không tìm thấy thư mục dữ liệu '{data_folder}' hoặc file 'info.mat' bên trong.")
    else:
        for name, (denoiser_func, out_folder) in filters_to_run.items():
            denoise_srgb(denoiser_func, data_folder, out_folder)
    
    print("!!! ĐÃ XỬ LÝ VÀ LƯU KẾT QUẢ CỦA TẤT CẢ CÁC BỘ LỌC !!!")