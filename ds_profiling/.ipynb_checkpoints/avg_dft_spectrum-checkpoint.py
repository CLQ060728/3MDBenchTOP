# author: Qian Liu
# email: qian.liu@fokus.fraunhofer.de
# date: 17-06-2024
# description: compute the average discrete fourier transform spectrum for a dataset.

import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import json
import os, math
from concurrent.futures import ProcessPoolExecutor


def calculate_average_frequency_spectra(images, rgb_or_g, full_band):
    if len(images) == 0:
        return None

    def get_avg(images, full_band):
        avg_spectra = np.zeros(images[0].shape[:2], dtype=np.float32)
        # Iterate over the images
        for raw_image in images:
            if full_band:
                image = raw_image
            else:
                # Perform high-pass filtering
                blurred_image = cv2.medianBlur(raw_image, 25)  # kernel size increases, magnitude-left decreases
                image = raw_image - blurred_image

            # Compute the Fourier transform
            freq_spectrum = np.fft.fft2(image)

            # Shift the zero-frequency component to the center
            freq_spectrum_shifted = np.fft.fftshift(freq_spectrum)

            # Accumulate the spectra
            avg_spectra += np.abs(freq_spectrum_shifted)
            avg_phase_spectra = np.angle(freq_spectrum_shifted)

        # Calculate the average spectrum
        avg_spectra /= len(images)
        avg_phase_spectra /= len(images)

            # Return the average spectrum
        return avg_spectra, avg_phase_spectra
     
    if rgb_or_g:
        images_b = images[:, :, :, 0].squeeze()
        images_g = images[:, :, :, 1].squeeze()
        images_r = images[:, :, :, 2].squeeze()
        avg_spectra_b, avg_phase_spectra_b = get_avg(images_b, full_band)
        avg_spectra_g, avg_phase_spectra_g = get_avg(images_g, full_band)
        avg_spectra_r, avg_phase_spectra_r = get_avg(images_r, full_band)
        return avg_spectra_b, avg_spectra_g, avg_spectra_r, avg_phase_spectra_b, avg_phase_spectra_g, avg_phase_spectra_r
    else:
        images_gr = np.array([])
        for raw_image in images:
            # Convert the image to grayscale
            image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
            if images_gr.shape[0] == 0:
                images_gr = image
            else:
                images_gr = np.vstack((images_gr, image))
        images_gr = images_gr.reshape((-1, *image.shape))
        avg_spectra_gr, avg_phase_spectra_gr = get_avg(images_gr, full_band)
        return avg_spectra_gr, None, None, avg_phase_spectra_gr, None, None

def visualize_average_frequency_spectra(real_images, fake_images, filename, rgb_or_g=False, full_band=False, out_path = "./"):
    # Clear the plot canvas
    plt.clf()

    # Randomly select images from real and fake images
    # selected_real_images = random.sample(real_images, k=min(len(real_images), 2000))
    # selected_fake_images = random.sample(fake_images, k=min(len(fake_images), 2000))
    if rgb_or_g:
        # Calculate the average frequency spectra for real images
        avg_spectra_real_b, avg_spectra_real_g, avg_spectra_real_r, \
        avg_phase_spectra_real_b, avg_phase_spectra_real_g, avg_phase_spectra_real_r = \
        calculate_average_frequency_spectra(real_images, rgb_or_g, full_band)
        
        # Calculate the average frequency spectra for fake images
        avg_spectra_fake_b, avg_spectra_fake_g, avg_spectra_fake_r, \
        avg_phase_spectra_fake_b, avg_phase_spectra_fake_g, avg_phase_spectra_fake_r = \
        calculate_average_frequency_spectra(fake_images, rgb_or_g, full_band)

        # Calculate the difference of the average frequency spectra
        diff_spectra_b = np.abs(avg_spectra_real_b - avg_spectra_fake_b)
        diff_spectra_g = np.abs(avg_spectra_real_g - avg_spectra_fake_g)
        diff_spectra_r = np.abs(avg_spectra_real_r - avg_spectra_fake_r)
        diff_phase_spectra_b = np.abs(avg_phase_spectra_real_b - avg_phase_spectra_fake_b)
        diff_phase_spectra_g = np.abs(avg_phase_spectra_real_g - avg_phase_spectra_fake_g)
        diff_phase_spectra_r = np.abs(avg_phase_spectra_real_r - avg_phase_spectra_fake_r)
        
        # Set up the plot
        fig, axes = plt.subplots(nrows=6, ncols=3, figsize=(18, 36))
        fig.suptitle(filename, fontsize=16)
        
        # Plot real images 
        axes[0][0].imshow(np.log(avg_spectra_real_b), cmap='viridis')
        axes[0][0].set_title('Real Images Blue Channel')
        axes[0][0].axis('off')
        axes[1][0].imshow(np.log(avg_spectra_real_g), cmap='viridis')
        axes[1][0].set_title('Real Images Green Channel')
        axes[1][0].axis('off')
        axes[2][0].imshow(np.log(avg_spectra_real_r), cmap='viridis')
        axes[2][0].set_title('Real Images Red Channel')
        axes[2][0].axis('off')
    
        # Plot fake images
        axes[0][1].imshow(np.log(avg_spectra_fake_b), cmap='viridis')
        axes[0][1].set_title('Fake Images Blue Channel')
        axes[0][1].axis('off')
        axes[1][1].imshow(np.log(avg_spectra_fake_g), cmap='viridis')
        axes[1][1].set_title('Fake Images Green Channel')
        axes[1][1].axis('off')
        axes[2][1].imshow(np.log(avg_spectra_fake_r), cmap='viridis')
        axes[2][1].set_title('Fake Images Red Channel')
        axes[2][1].axis('off')
    
        # Plot the difference
        pos_02 = axes[0][2].imshow(diff_spectra_b, cmap=plt.cm.coolwarm)
        axes[0][2].set_title('Blue Channel Difference (Real - Fake)')
        axes[0][2].axis('off')
        cbar_02 = fig.colorbar(pos_02, ax=axes[0][2], extend='both')
        cbar_02.minorticks_on()  
        pos_12 = axes[1][2].imshow(diff_spectra_g, cmap=plt.cm.coolwarm)
        axes[1][2].set_title('Green Channel Difference (Real - Fake)')
        axes[1][2].axis('off')
        cbar_12 = fig.colorbar(pos_12, ax=axes[1][2], extend='both')
        cbar_12.minorticks_on()  
        pos_22 = axes[2][2].imshow(diff_spectra_r, cmap=plt.cm.coolwarm)
        axes[2][2].set_title('Red Channel Difference (Real - Fake)')
        axes[2][2].axis('off')
        cbar_22 = fig.colorbar(pos_22, ax=axes[2][2], extend='both')
        cbar_22.minorticks_on()  
        
        # plot phase spectrums
        # Plot real images 
        axes[3][0].imshow(avg_phase_spectra_real_b, cmap='viridis')
        axes[3][0].set_title('Real Images Blue Channel - Phase')
        axes[3][0].axis('off')
        axes[4][0].imshow(avg_phase_spectra_real_g, cmap='viridis')
        axes[4][0].set_title('Real Images Green Channel - Phase')
        axes[4][0].axis('off')
        axes[5][0].imshow(avg_phase_spectra_real_r, cmap='viridis')
        axes[5][0].set_title('Real Images Red Channel - Phase')
        axes[5][0].axis('off')
    
        # Plot fake images
        axes[3][1].imshow(avg_phase_spectra_fake_b, cmap='viridis')
        axes[3][1].set_title('Fake Images Blue Channel - Phase')
        axes[3][1].axis('off')
        axes[4][1].imshow(avg_phase_spectra_fake_g, cmap='viridis')
        axes[4][1].set_title('Fake Images Green Channel - Phase')
        axes[4][1].axis('off')
        axes[5][1].imshow(avg_phase_spectra_fake_r, cmap='viridis')
        axes[5][1].set_title('Fake Images Red Channel - Phase')
        axes[5][1].axis('off')
    
        # Plot the difference
        pos_32 = axes[3][2].imshow(diff_spectra_b, cmap=plt.cm.coolwarm)
        axes[3][2].set_title('Blue Channel Phase Difference (Real - Fake)')
        axes[3][2].axis('off')
        cbar_32 = fig.colorbar(pos_32, ax=axes[3][2], extend='both')
        cbar_32.minorticks_on()  
        pos_42 = axes[4][2].imshow(diff_spectra_g, cmap=plt.cm.coolwarm)
        axes[4][2].set_title('Green Channel Phase Difference (Real - Fake)')
        axes[4][2].axis('off')
        cbar_42 = fig.colorbar(pos_42, ax=axes[4][2], extend='both')
        cbar_42.minorticks_on()  
        pos_52 = axes[5][2].imshow(diff_spectra_r, cmap=plt.cm.coolwarm)
        axes[5][2].set_title('Red Channel Phase Difference (Real - Fake)')
        axes[5][2].axis('off')
        cbar_52 = fig.colorbar(pos_52, ax=axes[5][2], extend='both')
        cbar_52.minorticks_on()  
        
    else:
        # Calculate the average frequency spectra for real images
        avg_spectra_real_gr, _, _, avg_phase_spectra_real_gr, _, _ = \
        calculate_average_frequency_spectra(real_images, rgb_or_g, full_band)

        # Calculate the average frequency spectra for fake images
        avg_spectra_fake_gr, _, _, avg_phase_spectra_fake_gr, _, _ = \
        calculate_average_frequency_spectra(fake_images, rgb_or_g, full_band)

        # Calculate the difference of the average frequency spectra
        diff_spectra_gr = np.abs(avg_spectra_real_gr - avg_spectra_fake_gr)
        diff_phase_spectra_gr = np.abs(avg_phase_spectra_real_gr - avg_phase_spectra_fake_gr)
        
        # Set up the plot
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 12))
        fig.suptitle(filename, fontsize=16)
    
        # Plot real images 
        axes[0][0].imshow(np.log(avg_spectra_real_gr), cmap='viridis')
        axes[0][0].set_title('Real Images')
        axes[0][0].axis('off')

        # Plot fake images
        axes[0][1].imshow(np.log(avg_spectra_fake_gr), cmap='viridis')
        axes[0][1].set_title('Fake Images')
        axes[0][1].axis('off')

        # Plot the difference
        pos_02 = axes[0][2].imshow(diff_spectra_gr, cmap=plt.cm.coolwarm)
        axes[0][2].set_title('Difference (Real - Fake)')
        axes[0][2].axis('off')
        cbar_02 = fig.colorbar(pos_02, ax=axes[0][2], extend='both')
        cbar_02.minorticks_on()  
        
        # plot phase spectrums
        # Plot real images 
        axes[1][0].imshow(avg_phase_spectra_real_gr, cmap='viridis')
        axes[1][0].set_title('Real Images - Phase')
        axes[1][0].axis('off')

        # Plot fake images
        axes[1][1].imshow(avg_phase_spectra_fake_gr, cmap='viridis')
        axes[1][1].set_title('Fake Images - Phase')
        axes[1][1].axis('off')

        # Plot the difference   plt.cm.Spectral
        pos_12 = axes[1][2].imshow(diff_phase_spectra_gr, cmap=plt.cm.coolwarm)
        axes[1][2].set_title('Phase Difference (Real - Fake)')
        axes[1][2].axis('off')
        cbar_12 = fig.colorbar(pos_12, ax=axes[1][2], extend='both')
        cbar_12.minorticks_on() 
    
    # Save the plot
    plt.savefig(os.path.join(out_path, 
                             f"dft_spectrums_{'colour' if rgb_or_g else 'gray'}_{'fullband' if full_band else ''}.png"))
    plt.close()

def load_images(input_path, task_amount, size=(512, 512)):
    image_arr = None
    for _, _, img_files in os.walk(input_path):
        # img_files = img_files[:200]
        print(f"img_files length: {len(img_files)}")
        num_tasks = math.ceil(len(img_files) / task_amount)
        remainder = len(img_files) % task_amount
        with ProcessPoolExecutor(num_tasks) as executor:
            input_paths = list()
            task_iterables = list()
            size_iterables = list()
            for idx in range(1, num_tasks+1, 1):
                if remainder == 0:
                    input_paths.append(input_path)
                    task_iterables.append(img_files[(idx-1)*task_amount:idx*task_amount])
                    size_iterables.append(size)
                else:
                    if idx < num_tasks:
                        input_paths.append(input_path)
                        task_iterables.append(img_files[(idx-1)*task_amount:idx*task_amount])
                        size_iterables.append(size)
                    else:
                        input_paths.append(input_path)
                        task_iterables.append(img_files[(idx-1)*task_amount:(idx-1)*task_amount+remainder])
                        size_iterables.append(size)
                    
            for result in executor.map(load_images_task, input_paths, task_iterables, size_iterables):
                if image_arr is None:
                    image_arr = result
                else:
                    image_arr = np.vstack((image_arr, result))
    
    return image_arr

def load_images_task(input_path, img_files, size):
    image_arr = np.array([])
    for img_file in img_files:
        img_path = os.path.join(input_path, img_file)
        image = cv2.imread(img_path)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, size, interpolation= cv2.INTER_LINEAR)
        if image_arr.shape[0] == 0:
            image_arr = image
        else:
            image_arr = np.vstack((image_arr, image))

    image_arr = image_arr.reshape((-1, *image.shape))
    
    return image_arr