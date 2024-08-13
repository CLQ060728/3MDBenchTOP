# author: Qian Liu
# email: qian.liu@fokus.fraunhofer.de
# date: 17-06-2024
# description: compute the average discrete fourier transform power spectrum for a dataset.

import numpy as np
import cv2
import os, math
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
from skimage.feature import graycomatrix, graycoprops


def compute_img_rgb_histogram(images, rgb_or_g, output_path="./"):
    if rgb_or_g:
        for image in images:
            hist_b = cv2.calcHist([image[:,:,0]],[0],None,[256],[0,256])
            hist_b += hist_b
            hist_g = cv2.calcHist([image[:,:,1]],[0],None,[256],[0,256])
            hist_g += hist_g
            hist_r = cv2.calcHist([image[:,:,2]],[0],None,[256],[0,256])
            hist_r += hist_r
        
        x = range(256)
        channels = ["blue", "green", "red"]
        for channel in channels:
            if channel == "blue":
                hist = hist_b
            elif channel == "green":
                hist = hist_g
            else:
                hist = hist_r
            
            plt.clf()
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.bar(x, hist.squeeze(), width=2, color=channel)

            outFullPath = os.path.join(output_path, f"average_histogram_RGB_{channel}.png")
            fig.savefig(outFullPath, dpi=fig.dpi)
    else:
        for image in images:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            hist_gr = cv2.calcHist([gray_image],[0],None,[256],[0,256])
            hist_gr += hist_gr
        
        x = range(256)
        plt.clf()
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.bar(x, hist_gr.squeeze(), width=2, color="gray")

        outFullPath = os.path.join(output_path, f"average_histogram_RGB_gray.png")
        fig.savefig(outFullPath, dpi=fig.dpi)


def compute_img_hsv_histogram(images, output_path="./"):
    hsv_name = ["hue", "saturation", "value"]
    for name in hsv_name:
        if name == "hue":
            channel = 0
            max_boundary = 180
        elif name == "saturation":
            channel = 1
            max_boundary = 256
        else:
            channel = 2
            max_boundary = 256
            
        for image in images:
            image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([image_hsv[:,:,channel]],[0],None,[max_boundary],[0,max_boundary])
            hist += hist
        
        plt.clf()
        x = range(max_boundary)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.bar(x, hist.squeeze(), width=2)

        outFullPath = os.path.join(output_path, f"average_histogram_HSV_{name}.png")
        fig.savefig(outFullPath, dpi=fig.dpi)


# ----------------- calculate greycomatrix() & greycoprops() for angle 0, 45, 90, 135 ----------------------------------
def calc_glcm_props_all_agls(img, props, dists, agls, lvl=256, sym=True, norm=True):
    
    glcm = graycomatrix(img, 
                        distances=dists, 
                        angles=agls, 
                        levels=lvl,
                        symmetric=sym, 
                        normed=norm)
    
    glcm_props = np.array([])
    distances = len(dists)
    angles = len(agls)
    
    for name in props:
        if glcm_props.shape[0] == 0:
            glcm_props = graycoprops(glcm, name)
        else:
            glcm_prop = graycoprops(glcm, name)
            glcm_props = np.vstack((glcm_props, glcm_prop))
    glcm_props = glcm_props.reshape((-1, distances, angles))
    
    return glcm_props


def compute_glcm_textures(input_path, rgb_or_g, task_amount, output_path="./"):
    def plot_hist(glcm_props_xs, num_rows, num_cols, distances_names, angles_names, properties, channel, output_path):    
        bin_sequence = [round(0.01*num, ndigits=2) for num in range(0, 101, 5)]
        bin_labels = np.array([])
        for bin_idx in range(1, len(bin_sequence), 1):
            bin_labels = np.append(bin_labels, f"({bin_sequence[bin_idx - 1]}, {bin_sequence[bin_idx]}]")
        
        for prop_idx in range(len(glcm_props_xs)):
            plt.clf()
            top_fig = plt.figure(constrained_layout=True, figsize=(30, 35))
            sub_figs = top_fig.subfigures(nrows=num_rows, ncols=1)
            for row, sub_fig in enumerate(sub_figs):
                sub_fig.suptitle(distances_names[row], x=0.5, y=1.01, fontsize=20)
                axs = sub_fig.subplots(nrows=1, ncols=num_cols)
                for col, ax in enumerate(axs):
                    bin_counts = np.zeros(len(bin_labels))
                    for bin_idx in range(1, len(bin_labels), 1):
                        condition = np.array([(glcm_props_xs[prop_idx, row, :, col] > bin_sequence[bin_idx - 1]) & 
                                    (glcm_props_xs[prop_idx, row, :, col] <= bin_sequence[bin_idx])])
                        bin_counts[bin_idx-1] += condition.sum(where=True)
                    
                    ax.bar(bin_labels, bin_counts, width=2)
                    ax.set_title(angles_names[col], fontsize=18)
                    ax.set_xticks(ax.get_xticks(), bin_labels, rotation=90, fontsize=14)
                    # ax.set_xticklabels(bin_labels, rotation=45, ha='right')
                    
            outFullPath = os.path.join(output_path, f"GLCM_Prop_Histogram_{properties[prop_idx]}_{channel}.png")
            top_fig.savefig(outFullPath, dpi=top_fig.dpi)
    
    distances = [1,2,3,4,5]
    distances_names = [f"Distance of {dist} Pixel(s)" for dist in distances]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    angles_names = ["0", "π/4", "π/2", "3π/4"]
    properties = ["correlation", "homogeneity", "ASM"]
    img_paths = []
    for _, _, img_files in os.walk(input_path):
        for img_file in img_files:
            img_paths.append(os.path.join(input_path, img_file))
    # img_paths = img_paths[:250]
    num_tasks = math.ceil(len(img_paths) / task_amount)
    print(f"num_tasks: {num_tasks}; img_paths length: {len(img_paths)}; task_amount: {task_amount}")
    remainder = len(img_paths) % task_amount
    
    with ProcessPoolExecutor(num_tasks) as executor:
        if rgb_or_g:
            task_futures = []
            for task_idx in range(1, num_tasks+1, 1):
                if remainder == 0:
                    task_futures.append(executor.submit(compute_rgb_glcm_props,
                                                        img_paths[((task_idx-1) * task_amount) : (task_idx * task_amount)],
                                                        properties, distances, angles, task_idx))
                else:
                    if task_idx < num_tasks:
                        task_futures.append(executor.submit(compute_rgb_glcm_props,
                                                            img_paths[((task_idx-1) * task_amount) : (task_idx * task_amount)],
                                                            properties, distances, angles, task_idx))
                    else:
                        task_futures.append(executor.submit(compute_rgb_glcm_props,
                                                            img_paths[((task_idx-1) * task_amount) : ((task_idx-1) * task_amount + remainder)],
                                                            properties, distances, angles, task_idx))

            glcm_props_bs = glcm_props_gs = glcm_props_rs = None
            for task_future in as_completed(task_futures):
                if glcm_props_bs is None and glcm_props_gs is None and glcm_props_rs is None:
                    glcm_props_bs, glcm_props_gs, glcm_props_rs = task_future.result()
                else:
                    glcm_props_b, glcm_props_g, glcm_props_r = task_future.result()
                    glcm_props_bs = np.concatenate((glcm_props_bs, glcm_props_b), axis=2)
                    glcm_props_gs = np.concatenate((glcm_props_gs, glcm_props_g), axis=2)
                    glcm_props_rs = np.concatenate((glcm_props_rs, glcm_props_r), axis=2)
            
            num_rows = len(distances)
            num_cols = len(angles)
            plot_hist(glcm_props_bs, num_rows, num_cols, distances_names, angles_names, properties, "Blue", output_path)
            plot_hist(glcm_props_gs, num_rows, num_cols, distances_names, angles_names, properties, "Green", output_path)
            plot_hist(glcm_props_rs, num_rows, num_cols, distances_names, angles_names, properties, "Red", output_path)
            save_glcm_stats(glcm_props_bs, properties, distances_names, angles_names, "Blue", output_path)
            save_glcm_stats(glcm_props_gs, properties, distances_names, angles_names, "Green", output_path)
            save_glcm_stats(glcm_props_rs, properties, distances_names, angles_names, "Red", output_path)
        else:
            task_futures = []
            for task_idx in range(1, num_tasks+1, 1):
                if remainder == 0:
                    task_futures.append(executor.submit(compute_gray_glcm_props,
                                                        img_paths[((task_idx-1) * task_amount) : (task_idx * task_amount)],
                                                        properties, distances, angles, task_idx))
                else:
                    if task_idx < num_tasks:
                        task_futures.append(executor.submit(compute_gray_glcm_props,
                                                            img_paths[((task_idx-1) * task_amount) : (task_idx * task_amount)],
                                                            properties, distances, angles, task_idx))
                    else:
                        task_futures.append(executor.submit(compute_gray_glcm_props,
                                                            img_paths[((task_idx-1) * task_amount) : ((task_idx-1) * task_amount + remainder)],
                                                            properties, distances, angles, task_idx))

            glcm_props_grs = None
            for task_future in as_completed(task_futures):
                if glcm_props_grs is None:
                    glcm_props_grs = task_future.result()
                else:
                    glcm_props_gr = task_future.result()
                    glcm_props_grs = np.concatenate((glcm_props_grs, glcm_props_gr), axis=2)
                
            num_rows = len(distances)
            num_cols = len(angles)
            plot_hist(glcm_props_grs, num_rows, num_cols, distances_names, angles_names, properties, "Gray", output_path)
            save_glcm_stats(glcm_props_grs, properties, distances_names, angles_names, "Gray", output_path)


def compute_rgb_glcm_props(img_paths, properties, distances, angles, task_id):
    glcm_props_bs = np.array([])
    glcm_props_gs = np.array([])
    glcm_props_rs = np.array([])
    for img_path in img_paths:
        image = cv2.imread(img_path)
        glcm_props_b = calc_glcm_props_all_agls(image[:,:,0], properties, distances, angles)
        glcm_props_b = glcm_props_b.reshape((glcm_props_b.shape[0], glcm_props_b.shape[1], 1, glcm_props_b.shape[2]))
        glcm_props_g = calc_glcm_props_all_agls(image[:,:,1], properties, distances, angles)
        glcm_props_g = glcm_props_g.reshape((glcm_props_g.shape[0], glcm_props_g.shape[1], 1, glcm_props_g.shape[2]))
        glcm_props_r = calc_glcm_props_all_agls(image[:,:,2], properties, distances, angles)
        glcm_props_r = glcm_props_r.reshape((glcm_props_r.shape[0], glcm_props_r.shape[1], 1, glcm_props_r.shape[2]))
        
        if glcm_props_bs.shape[0] == 0:
            glcm_props_bs = glcm_props_b
        else:
            glcm_props_bs = np.concatenate((glcm_props_bs, glcm_props_b), axis=2)
        if glcm_props_gs.shape[0] == 0:
            glcm_props_gs = glcm_props_g
        else:
            glcm_props_gs = np.concatenate((glcm_props_gs, glcm_props_g), axis=2)
        if glcm_props_rs.shape[0] == 0:
            glcm_props_rs = glcm_props_r
        else:
            glcm_props_rs = np.concatenate((glcm_props_rs, glcm_props_r), axis=2)
    
    return glcm_props_bs, glcm_props_gs, glcm_props_rs


def compute_gray_glcm_props(img_paths, properties, distances, angles, task_id):
    glcm_props_grs = np.array([])
    for img_path in img_paths:
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        glcm_props_gr = calc_glcm_props_all_agls(image, properties, distances, angles)
        glcm_props_gr = glcm_props_gr.reshape((glcm_props_gr.shape[0], glcm_props_gr.shape[1], 1, glcm_props_gr.shape[2]))
        if glcm_props_grs.shape[0] == 0:
            glcm_props_grs = glcm_props_gr
        else:
            glcm_props_grs = np.concatenate((glcm_props_grs, glcm_props_gr), axis=2)
    
    return glcm_props_grs


def save_glcm_stats(glcm_props_xs, properties, distances_names, angles_names, channel, output_path):
    header_str = f"GLCM array with shape (properties, pixel distances, # images, angles) - "\
                 f"({glcm_props_xs.shape}) for {channel}-band\n"\
                 f"Properties: {properties}\n {distances_names}\n Angles: {angles_names}\n"
    for pro_idx in range(len(properties)):
        for dis_idx in range(len(distances_names)):
            for angle_idx in range(len(angles_names)):
                header_str += f"Property: {properties[pro_idx]} - {distances_names[dis_idx]} - Angle: {angles_names[angle_idx]}\n"
                min = glcm_props_xs[pro_idx, dis_idx, :, angle_idx].min()
                min = round(min, ndigits=8)
                max = glcm_props_xs[pro_idx, dis_idx, :, angle_idx].max()
                max = round(max, ndigits=8)
                mean = glcm_props_xs[pro_idx, dis_idx, :, angle_idx].mean()
                mean = round(mean, ndigits=8)
                median = np.median(glcm_props_xs[pro_idx, dis_idx, :, angle_idx])
                median = round(median, ndigits=8)
                std = glcm_props_xs[pro_idx, dis_idx, :, angle_idx].std()
                std = round(std, ndigits=8)
                header_str += f"MIN: {min}; MAX: {max}; MEAN: {mean}; MEDIAN: {median}; STD: {std}\n"
    
    output_full_path = os.path.join(output_path, f"GLCM-{channel}.txt")
    with open(output_full_path, "a") as file_writer:
        file_writer.write(header_str)


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