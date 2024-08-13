# author: Qian Liu
# email: qian.liu@fokus.fraunhofer.de
# date: 17-06-2024
# description: compute the average discrete fourier transform power spectrum for a dataset.

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
import random
import json
import os, math
from control.matlab import unwrap
from concurrent.futures import ProcessPoolExecutor
from psds import power_spectrum


def calculate_average_frequency_power_spectra(images, rgb_or_g):
    if len(images) == 0:
        return None

    def get_avgs(images):
        avg_spectra = np.zeros(images[0].shape[:2], dtype=np.float32)
        avg_power_spectra = np.zeros(images[0].shape[:2], dtype=np.float32)
        # Iterate over the images
        for raw_image in images:
            # Compute the Fourier transform
            freq_spectrum = np.fft.fft2(raw_image)

            # Shift the zero-frequency component to the center
            freq_spectrum_shifted = np.fft.fftshift(freq_spectrum)

            # compute power spectrum
            freq_power_spectrum = np.abs(freq_spectrum_shifted)**2
            
            # Accumulate the spectra
            avg_spectra += np.abs(freq_spectrum_shifted)
            avg_power_spectra += freq_power_spectrum

        # Calculate the average spectrum
        avg_spectra /= len(images)
        avg_power_spectra /= len(images)

            # Return the average spectrum
        return avg_spectra, avg_power_spectra
     
    if rgb_or_g:
        images_b = images[:, :, :, 0].squeeze()
        images_g = images[:, :, :, 1].squeeze()
        images_r = images[:, :, :, 2].squeeze()
        avg_spectra_b, avg_power_spectra_b = get_avgs(images_b)
        avg_spectra_g, avg_power_spectra_g = get_avgs(images_g)
        avg_spectra_r, avg_power_spectra_r = get_avgs(images_r)
        return avg_spectra_b, avg_spectra_g, avg_spectra_r,\
               avg_power_spectra_b, avg_power_spectra_g, avg_power_spectra_r
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
        avg_spectra_gr, avg_power_spectra_gr = get_avgs(images_gr)
        return avg_spectra_gr, None, None, avg_power_spectra_gr, None, None

def get_fxy(m, n):
    a = 4.65
    fx = np.zeros((m,1))
    for k in range(m):
        fx[k] = (2 * np.pi / m) * k
    fx = np.fft.fftshift(fx)
    # for removing discontinuity in the middle, shift data by 2pi
    fx = unwrap((fx-2*np.pi).squeeze())
    fx = fx / a
    
    fy = np.zeros((n,1))
    for k in range(n):
        fy[k] = (2 * np.pi / n) * k
    fy = np.fft.fftshift(fy)
    # for removing discontinuity in the middle, shift data by 2pi
    fy = unwrap((fy-2*np.pi).squeeze())
    fy = fy / a

    return fx, fy

def visualize_average_frequency_power_spectra(images, plot_names, rgb_or_g=False, out_path = "./"):
    
    num_img_sets = len(images)
    if rgb_or_g:
        avg_spectra_bs = list()
        avg_spectra_gs = list()
        avg_spectra_rs = list()
        avg_power_spectra_bs = list()
        avg_power_spectra_gs = list()
        avg_power_spectra_rs = list()
        for idx in range(num_img_sets):
            # Calculate the average frequency spectra for image datasets
            avg_spectra_b, avg_spectra_g, avg_spectra_r,\
            avg_power_spectra_b, avg_power_spectra_g, avg_power_spectra_r = \
            calculate_average_frequency_power_spectra(images[idx], rgb_or_g)
            avg_spectra_bs.append(avg_spectra_b)
            avg_spectra_gs.append(avg_spectra_g)
            avg_spectra_rs.append(avg_spectra_r)
            avg_power_spectra_bs.append(avg_power_spectra_b)
            avg_power_spectra_gs.append(avg_power_spectra_g)
            avg_power_spectra_rs.append(avg_power_spectra_r)

        num_rows = math.ceil(num_img_sets / 4)
        remainder = num_img_sets % 4
        num_cols = remainder if num_rows == 1 and remainder != 0 else 4
        
        # fig.suptitle(filename, fontsize=16)
        
        # Plot images power spectrum
        num_freq_bin = avg_spectra_bs[0].shape[0]
        print(f"num_freq_bin: {num_freq_bin}")
        fx, fy = get_fxy(num_freq_bin, num_freq_bin)
        fx, fy = np.meshgrid(fx, fy)
        channels = ["Blue", "Green", "Red"]

        for channel in channels:
            # avg_idx = 0
            # Clear the plot canvas
            plt.clf()
            # Set up the plot
            fig = plt.figure(figsize=(num_cols*5, num_rows*5))
            if channel == "Blue":
                Z = avg_power_spectra_bs
                avg_spectra = avg_spectra_bs
            elif channel == "Green":
                Z = avg_power_spectra_gs
                avg_spectra = avg_spectra_gs
            else:
                Z = avg_power_spectra_rs
                avg_spectra = avg_spectra_rs
            
            for avg_idx in range(num_img_sets):
                ax = fig.add_subplot(num_rows, num_cols, avg_idx+1, projection='3d')
                ax.plot_surface(fx, fy, np.log(Z[avg_idx]), cmap=cm.coolwarm, linewidth=0, antialiased=False)
                ax.set_title(plot_names[avg_idx])
                ax.set_xlabel("fx")
                ax.set_ylabel("fy")
                ax.set_zlabel("Log ||A(f)||")

            outFullPath = os.path.join(out_path, f"dft_power_spectrum_3D_{channel}.png")
            fig.savefig(outFullPath, dpi=fig.dpi)
            
    else:
        avg_spectra_grs = list()
        avg_power_spectra_grs = list()
        for idx in range(num_img_sets):
            # Calculate the average frequency spectra for image datasets
            avg_spectra_gr, _, _, avg_power_spectra_gr, _, _ = \
            calculate_average_frequency_power_spectra(images[idx], rgb_or_g)
            avg_spectra_grs.append(avg_spectra_gr)
            avg_power_spectra_grs.append(avg_power_spectra_gr)

        num_rows = math.ceil(num_img_sets / 4)
        remainder = num_img_sets % 4
        num_cols = remainder if num_rows == 1 and remainder != 0 else 4

        # Plot images power spectrum
        num_freq_bin = avg_spectra_grs[0].shape[0]
        print(f"num_freq_bin: {num_freq_bin}")
        fx, fy = get_fxy(num_freq_bin, num_freq_bin)
        fx, fy = np.meshgrid(fx, fy)
        Z = avg_power_spectra_grs
        # Clear the plot canvas
        plt.clf()
        # Set up the plot
        fig = plt.figure(figsize=(num_cols*5, num_rows*5))
        
        for avg_idx in range(num_img_sets):
            ax = fig.add_subplot(num_rows, num_cols, avg_idx+1, projection='3d')
                                 
            ax.plot_surface(fx, fy, np.log(Z[avg_idx]), cmap=cm.coolwarm, linewidth=0, antialiased=False)
            ax.set_title(plot_names[avg_idx])
            ax.set_xlabel("fx")
            ax.set_ylabel("fy")
            ax.set_zlabel("Log ||A(f)||")

        outFullPath = os.path.join(out_path, "dft_power_spectrum_3D_Gray.png")
        fig.savefig(outFullPath, dpi=fig.dpi)


def visualise_average_images_psd(images, plot_names, fig_size, rgb_or_g=False, out_path = "./"):
    num_img_sets = len(images)
    num_rows = math.ceil(num_img_sets / 4)
    remainder = num_img_sets % 4
    num_cols = remainder if num_rows == 1 and remainder != 0 else 4
    
    if rgb_or_g:
        channels = ["Blue", "Green", "Red"]
        for channel in channels:
            plt.clf()
            # Set up the plot
            fig = plt.figure(figsize=fig_size)
            for idx in range(num_img_sets):
                mean_images = np.mean(images[idx], axis=0)
                mean_images_b = mean_images[:, :, 0]
                mean_images_g = mean_images[:, :, 1]
                mean_images_r = mean_images[:, :, 2]
                if channel == "Blue":
                    freq, power = power_spectrum(mean_images_b)
                elif channel == "Green":
                    freq, power = power_spectrum(mean_images_g)
                else:
                    freq, power = power_spectrum(mean_images_r)
                
                ax_freq = fig.add_subplot(num_rows, 4, idx+1)
                ax_freq.plot(np.log(freq), np.log(power))
                ax_freq.set_title(plot_names[idx])
                ax_freq.set_xlabel("Log Frequency")
                ax_freq.set_ylabel("Log Power Magnitude")
                ax_freq.grid(visible=True)

            outFullPath_2D = os.path.join(out_path, f"dft_power_frequency_2D_{channel}.png")
            fig.savefig(outFullPath_2D, dpi=fig.dpi)
    else:
        plt.clf()
        # Set up the plot
        fig = plt.figure(figsize=fig_size)
        for idx in range(num_img_sets):
            images_gr = np.array([])
            for raw_image in images[idx]:
                # Convert the image to grayscale
                gray_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
                if images_gr.shape[0] == 0:
                    images_gr = gray_image
                else:
                    images_gr = np.vstack((images_gr, gray_image))
            images_gr = images_gr.reshape((-1, *gray_image.shape))
            mean_images = np.mean(images_gr, axis=0)
            freq, power = power_spectrum(mean_images)
            
            ax_freq = fig.add_subplot(num_rows, 4, idx+1)
            ax_freq.plot(np.log(freq), np.log(power))
            ax_freq.set_title(plot_names[idx])
            ax_freq.set_xlabel("Log Frequency")
            ax_freq.set_ylabel("Log Power Magnitude")
            ax_freq.grid(visible=True)

        outFullPath_2D = os.path.join(out_path, f"dft_power_frequency_2D_Gray.png")
        fig.savefig(outFullPath_2D, dpi=fig.dpi)
            

def load_images(input_path, task_amount, size=(512, 512)):
    image_arr = None
    for _, _, img_files in os.walk(input_path):
        # img_files = img_files[:200]
        print(f"img_files length: {len(img_files)}")
        num_tasks = math.ceil(len(img_files) / task_amount)
        print(f"num_tasks: {num_tasks}; img_files length: {len(img_files)}; task_amount: {task_amount}")
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