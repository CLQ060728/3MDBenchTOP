from __future__ import division
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import convolve2d
import os, math
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed


# FUNCTIONS
def RGB2YIQ(img_bgr):
    """
    Conversion from BGR color space to YIQ color space
    :param img_bgr: input BGR image
    :return: image YIQ
    """
    BGR = img_bgr.copy().astype(float)
    R = BGR[:, :, 0]
    G = BGR[:, :, 1]
    B = BGR[:, :, 2]

    Y = (0.299 * R) + (0.587 * G) + (0.114 * B)
    I = (0.59590059 * R) + (-0.27455667 * G) + (-0.32134392 * B)
    Q = (0.21153661 * R) + (-0.52273617 * G) + (0.31119955 * B)

    YIQ = np.round(np.dstack((Y, I + 128, Q + 128))).astype(np.uint8)
    return YIQ


def delta(img, center, x, y):
    """
    Compares current position with the center of the lbp region
    :param img: image in input
    :param center: center of lbp region
    :param x: current x coordinate
    :param y: current y coordinate
    :return: 1 if current position is greater or equal than the center position, otherwise 0
    """
    new_value = 0
    try:
        # If local neighbourhood pixel value is greater than or equal to center pixel values then set it to 1
        if img[x][y] >= center:
            new_value = 1
    except:
        # Exception is required when neighbourhood value of a center pixel value is null (values present at boundaries).
        pass

    return new_value


def lbp_standard(img, x, y):
    """
    Compute the lbp standard of an image
    :param img: input image
    :param x: current x coordinate
    :param y: current y coordinate
    :return: image in lbp form
    """
    center = img[x][y]

    #Values array, it is the 8bit LBP string
    val_ar = []
    #top_left
    val_ar.append(delta(img, center, x - 1, y - 1))
    #top_center
    val_ar.append(delta(img, center, x - 1, y))
    #top_right
    val_ar.append(delta(img, center, x - 1, y + 1))
    #right
    val_ar.append(delta(img, center, x, y + 1))
    #bottom_right
    val_ar.append(delta(img, center, x + 1, y + 1))
    #bottom_center
    val_ar.append(delta(img, center, x + 1, y))
    #bottom_left
    val_ar.append(delta(img, center, x + 1, y - 1))
    #left
    val_ar.append(delta(img, center, x, y - 1))

    #Conversion into decimal notation
    power_val = [1, 2, 4, 8, 16, 32, 64, 128]
    val = 0

    for i in range(len(val_ar)):
        val += val_ar[i] * power_val[i]

    final_lbp = np.array(val)
    print(final_lbp)

    return val


def lbp_uniform(img, x, y):
    """
    Compute the lbp uniform of an image (this is more efficient with faces)
    :param img: input image
    :param x: current x coordinate
    :param y: current y coordinate
    :return: image in lbp form
    """
    value = 0 #result
    #Center of the image and number of point (or pixel) that we will check
    center = img[x][y]
    p = 8

    #Distribution of pixels and respective names, '--' is the center
    # P0 P1 P2
    # P7 -- P3
    # P6 P5 P4

    #Uniformity check, we check the difference on LBP value between each pixel
    #Check between pixel P7(Left) and P0(TopLeft)
    ue = abs(delta(img, center, x, y - 1) - delta(img, center, x - 1, y - 1))
    #Check between pixel P1(Top) e P0(TopLeft)
    us1 = abs(delta(img, center, x - 1, y) - delta(img, center, x - 1, y - 1))
    #Check between pixel P2(TopRight) and P1(Top)
    us2 = abs(delta(img, center, x - 1, y + 1) - delta(img, center, x - 1, y))
    #Check between pixel P3(Right) and P2(TopRight)
    us3 = abs(delta(img, center, x, y + 1) - delta(img, center, x - 1, y + 1))
    #Check between pixel P4(BottomRight) and P3(Right)
    us4 = abs(delta(img, center, x + 1, y + 1) - delta(img, center, x, y + 1))
    #Check between pixel P5(Bottom) and P4(BottomRight)
    us5 = abs(delta(img, center, x + 1, y) - delta(img, center, x + 1, y + 1))
    #Check between pixel P6(BottomLeft) and P5(Bottom)
    us6 = abs(delta(img, center, x + 1, y - 1) - delta(img, center, x + 1, y))
    #Check between pixel P7(Left) and P6(BottomLeft)
    us7 = abs(delta(img, center, x, y - 1) - delta(img, center, x + 1, y - 1))

    transitions_number = ue + us1 + us2 + us3 + us4 + us5 + us6 + us7

    #Verification of LBP, if we have 2 or less transitions on the LBP binary string, then it is uniform
    if transitions_number <= 2:
        val_ar = [delta(img, center, x - 1, y - 1), delta(img, center, x - 1, y),  # top_left - top
                  delta(img, center, x - 1, y + 1), delta(img, center, x, y + 1),  # top_right - right
                  delta(img, center, x + 1, y + 1), delta(img, center, x + 1, y),  # bottom_right - bottom
                  delta(img, center, x + 1, y - 1), delta(img, center, x, y - 1)]  # bottom_left - left

        #Compute the correct decimal LBP value for that pixel
        power_val = [1, 2, 4, 8, 16, 32, 64, 128]
        for i in range(len(val_ar)):
            value += val_ar[i] * power_val[i]

    #else if we have more than two transitions then the lbp is not uniform (we assign a fixed decimal value)
    else:
        value = p*(p-1)+2

    return value


def split_region(img, region_factor):
    """
    Splits the input image into a certain number of regions
    :param img: input image
    :param region_factor: length and width of how to factorize the image
    :return:
    """
    #Divide the image in equal region given the wanted number of regions
    height = int(img.shape[0] / region_factor)
    width = int(img.shape[1] / region_factor)

    #Create a region list and fill it with the image's regions (with height and width computed previously)
    region = []
    i = 0
    for x in range(region_factor):
        for y in range(region_factor):
            temp_frame = img[0 + (x * height):height + (x * height), 0 + (y * width):width + (y * width)]
            region.append(temp_frame)
            i += 1

    return region


def lbp_region_image_compute(region_list):
    """
    Compute the lbp of every region obtained
    :param region_list: list of regions obtained with split_region method
    :return: list of lbp regions
    """
    #Create a list of empty regions all with the same height and width as the input regions
    lbp_region = []
    height, width = region_list[0].shape

    #Initialize new regions, then compute the LBP for each region and add the LBP to the list lbp_region
    for v in region_list:
        temp_region = np.zeros((height, width), np.uint8)

        for x in range(height):
            for y in range(width):
                temp_region[x, y] = lbp_uniform(v, x, y)

        lbp_region.append(temp_region)

    return lbp_region


def histogram(img, v_min, v_max, scale):
    """
    Creates histogram of a given image
    :param img: input image
    :param v_min: lower bound of admitted values
    :param v_max: upper bound of admitted values
    :param scale: Scale can be used to reduce the resolution and limit the computation phase during test
    :return: histogram
    """
    #Initialize a histogram
    hist = []

    #Initialize the histogram with the correct len
    for x in range(v_min, v_max):
        hist.append(0)

    #For each pixel in height x and width y, add it to the right column of the histogram
    #Scale can be used to reduce the resolution and limit the computation phase during test
    range_x = int(img.shape[0]*scale)
    range_y = int(img.shape[1]*scale)
    total_pixel = range_x * range_y

    for x in range(range_x):
        for y in range(range_y):
            error = 1
            for v in range(v_min, v_max):
                if img[x][y] == v:
                    hist[v] = hist[v] + 1
                    error = 0
            #Check if the pixel under analysis has been assigned to no column of the histogram
            if error == 1:
                print("WARNING: pixel (", x, y, ") NOT ADDED TO ANY COLUMN")

    #Histogram element number check
    histogram_elements = 0
    for x in range(v_min, v_max):
        histogram_elements = histogram_elements + hist[x]
    # print("Histogram total element:", histogram_elements, "/", total_pixel,
          # ", Missing elements:", total_pixel - histogram_elements)
    # if histogram_elements != total_pixel:
        # print("MISSING ELEMENTS, CHECK IF v_min and v_max are correct!")

    np_hist = np.array(hist)

    return np_hist


def histogram_region(lbp_region, v_min, v_max, scale):
    """
    Compute the histogram of all the regions
    :param lbp_region: regions list obtained with lbp_region_image_compute
    :param v_min: lower bound of admitted values
    :param v_max: upper bound of admitted values
    :param scale: Scale can be used to reduce the resolution and limit the computation phase during test
    :return: histogram list
    """
    #Create a list of histogram, then compute the histogram for each region and save the result in the list
    hist = []
    for x in lbp_region:
        temp_hist = histogram(x, v_min, v_max, scale)
        hist.append(temp_hist)

    np_hist = np.array(hist)

    return np_hist


def histogram_full_image_print(hist, name):
    """
    Prints the final histogram
    :param hist: input histogram to print
    :param name: name of the histogram
    :return: printed histogram
    """
    plt.figure()
    plt.bar(np.arange(len(hist)), hist, width=10, color='b')
    plt.title(name)
    plt.show()
    return


def histogram_concatenate(hist_list, chart=False, name="NAME"):
    """
    Concatenate all the histograms of the different regions coming from histogram_region
    :param hist_list: input list of histograms
    :param chart: tells you whether you want to plot the final histogram or not (default value FALSE)
    :param name: name of a single histogram plot
    :return: final concatenated histogram
    """
    final_hist = []
    single_hist_len = len(hist_list[0])
    final_hist_len = len(hist_list) * single_hist_len

    #Initialize the final histogram with the correct len
    for x in range(final_hist_len):
        final_hist.append(0)

    #Populate the final histogram concatenating all the histogram from the input list in one single histogram
    for x in range(len(hist_list)):
        temp_hist = hist_list[x]
        for y in range(single_hist_len):
            final_hist[y + x*single_hist_len] = temp_hist[y]

    final_np_hist = np.array(final_hist)

    #Prints the histogram
    if chart:
        histogram_full_image_print(final_np_hist, 'LBP Histogram for channel: %(name)s' % {"name": name})

    # print("LBP descriptor for channel", name, "successfully computed. It is a fusion of:", len(hist_list),
          # "histograms, with length:", single_hist_len, ",total length of:", final_hist_len)

    return final_np_hist


def CoALBP(image, chart=False, name="NAME", lbp_r=1, co_r=2):
    """
    Compute the Co-occurrence of Adjacent Local Binary Patterns of an image
    :param image: input image
    :param chart: tells you whether you want to plot the final histogram or not (default value FALSE)
    :param name: name of a single CoALBP plot
    :param lbp_r: radius for adjacent local binary patterns
    :param co_r: radius for co-occurence of the patterns
    :return: CoALBP descriptor with length 1024 * number of channels
    """
    height, width, channels = image.shape

    #albp and co-occurrence per channel in image
    histogram = np.empty(0, dtype=int)
    for i in range(image.shape[2]):
        C = image[lbp_r:height - lbp_r, lbp_r:width - lbp_r, i].astype(float)
        X = np.zeros((4, height - 2 * lbp_r, width - 2 * lbp_r))
        # adjacent local binary patterns
        X[0, :, :] = image[lbp_r:height - lbp_r, lbp_r + lbp_r:width - lbp_r + lbp_r, i] - C
        X[1, :, :] = image[lbp_r - lbp_r:height - lbp_r - lbp_r, lbp_r:width - lbp_r, i] - C
        X[2, :, :] = image[lbp_r:height - lbp_r, lbp_r - lbp_r:width - lbp_r - lbp_r, i] - C
        X[3, :, :] = image[lbp_r + lbp_r:height - lbp_r + lbp_r, lbp_r:width - lbp_r, i] - C
        X = (X > 0).reshape(4, -1)

        # co-occurrence of the patterns
        A = np.dot(np.array([1, 2, 4, 8]), X)
        A = A.reshape(height - 2 * lbp_r, width - 2 * lbp_r) + 1
        hh, ww = A.shape
        D = (A[co_r:hh - co_r, co_r:ww - co_r] - 1) * 16 - 1
        Y1 = A[co_r:hh - co_r, co_r + co_r:ww - co_r + co_r] + D
        Y2 = A[co_r - co_r:hh - co_r - co_r, co_r + co_r:ww - co_r + co_r] + D
        Y3 = A[co_r - co_r:hh - co_r - co_r, co_r:ww - co_r] + D
        Y4 = A[co_r - co_r:hh - co_r - co_r, co_r - co_r:ww - co_r - co_r] + D
        Y1 = np.bincount(Y1.ravel(), minlength=256)
        Y2 = np.bincount(Y2.ravel(), minlength=256)
        Y3 = np.bincount(Y3.ravel(), minlength=256)
        Y4 = np.bincount(Y4.ravel(), minlength=256)
        pattern = np.concatenate((Y1, Y2, Y3, Y4))
        histogram = np.concatenate((histogram, pattern))

    CoALBPdesc = histogram

    # Prints the histogram of the descriptor
    if chart:
        plt.figure()
        plt.bar(np.arange(len(CoALBPdesc)), CoALBPdesc, width=10)
        plt.title('CoALBP histogram for channel: %(name)s, with lbp_r: %(lbpr)d and co_r: %(cor)d' % {"lbpr": lbp_r, "cor": co_r, "name": name})
        plt.show()        
    # print("CoALBP descriptor for channel", name, "successfully computed with lbp_r:", lbp_r, ", and co_r:", co_r, "total length of:", len(CoALBPdesc))

    return CoALBPdesc


def lpq(image, chart=False, name="NAME", winSize=3, freqestim=1, mode='nh'):
    """
    Compute the Local Phase Quantization of an image
    :param image: input image
    :param chart: tells you whether you want to plot the final histogram or not (default value FALSE)
    :param name: name of a single CoALBP plot
    :param winSize:
    :param freqestim:
    :param mode:
    :return: descriptor with length 256
    """

    rho = 0.90

    STFTalpha = 1/winSize   # alpha in STFT approaches (for Gaussian derivative alpha=1)
    sigmaS = (winSize-1)/4  # Sigma for STFT Gaussian window (applied if freqestim==2)
    sigmaA = 8/(winSize-1)  # Sigma for Gaussian derivative quadrature filters (applied if freqestim==3)

    convmode = 'valid' # Compute descriptor responses only on part that have full neigborhood. Use 'same' if all pixels are included (extrapolates np.image with zeros).

    image = np.float64(image)  # Convert np.image to double
    r = (winSize-1)/2  # Get radius from window size
    x = np.arange(-r, r+1)[np.newaxis]  # Form spatial coordinates in window

    w0 = w1 = w2 = 0

    if freqestim == 1:  # STFT uniform window
        #  Basic STFT filters
        w0 = np.ones_like(x)
        w1 = np.exp(-2*np.pi*x*STFTalpha*1j)
        w2 = np.conj(w1)
        # print("Vector w0:", x)

    ## Run filters to compute the frequency response in the four points. Store np.real and np.imaginary parts separately
    # Run first filter
    filterResp1 = convolve2d(convolve2d(image, w0.T, convmode), w1, convmode)
    filterResp2 = convolve2d(convolve2d(image, w1.T, convmode), w0, convmode)
    filterResp3 = convolve2d(convolve2d(image, w1.T, convmode), w1, convmode)
    filterResp4 = convolve2d(convolve2d(image, w1.T, convmode), w2, convmode)

    # Initilize frequency domain matrix for four frequency coordinates (np.real and np.imaginary parts for each frequency).
    freqResp = np.dstack([filterResp1.real, filterResp1.imag,
                        filterResp2.real, filterResp2.imag,
                        filterResp3.real, filterResp3.imag,
                        filterResp4.real, filterResp4.imag])

    ## Perform quantization and compute LPQ codewords
    inds = np.arange(freqResp.shape[2])[np.newaxis,np.newaxis,:]
    LPQdesc = ((freqResp>0)*(2**inds)).sum(2)

    ## Switch format to uint8 if LPQ code np.image is required as output
    if mode == 'im':
        LPQdesc = np.uint8(LPQdesc)

    ## Histogram if needed
    if mode == 'nh' or mode == 'h':
        LPQdesc = np.histogram(LPQdesc.flatten(), range(256))[0]

    ## Normalize histogram if needed
    if mode == 'nh':
        LPQdesc = LPQdesc/LPQdesc.sum()

    if chart:
        plt.figure()
        plt.bar(np.arange(len(LPQdesc)), LPQdesc, width=5)
        plt.title('LPQ histogram for channel: %(name)s' % {"name": name})
        plt.show()        

    LPQdesc = np.concatenate((LPQdesc, np.zeros(1)))
    # print("LPQ descriptor for channel", name, "successfully computed, total length of:", len(LPQdesc))

    return LPQdesc


def channel_descriptor(channel, channel_3D, channel_name="NAME", n_split=3, v_min=0, v_max=256, scale=1.0, LBP_multiplier=1, CoALBP_multiplier=1, LPQ_multiplier=1, single_charts=False, final_charts=False):
    """
    Given a channel we carry out the full analysis composed of LBP, CoALBP (run 3 times with different parameters) and LPQ
    :param channel: input channel (bi-dimensional image)
    :param channel_3D: input 3D channel (three dimensions matrix with only one channel, used for CoALBP)
    :param channel_name: name of the channel we are working with
    :param n_split: length and width of how to factorize the image
    :param v_min: lower bound of admitted values
    :param v_max: upper bound of admitted values
    :param scale: Scale can be used to reduce the resolution and limit the computation phase during test
    :param LBP_multiplier: scale value used in order to see the columns of the histogram
    :param CoALBP_multiplier: scale value used in order to see the columns of the histogram
    :param LPQ_multiplier: scale value used in order to see the columns of the histogram
    :param single_charts: tells you whether you want to plot all the single histograms or not (default value FALSE)
    :param final_charts: plot the final descriptor for the current channel (default value FALSE)
    :return: the complete descriptor for the channel
    """
    print("---------- STARTING COMPUTATION OF FINAL DESCRIPTOR FOR CHANNEL", channel_name, "----------")
    #Compute LBP descriptor (Split channel, compute single split LBP, compute single histograms, concatenate, print)
    channel_splits = split_region(channel, n_split)
    channel_splits_lbp = lbp_region_image_compute(channel_splits)
    channel_splits_lbp_hist = histogram_region(channel_splits_lbp, v_min, v_max, scale)
    channel_lbp_descriptor = histogram_concatenate(channel_splits_lbp_hist, single_charts, channel_name) * LBP_multiplier
    
    #Compute CoALBP descriptors (each one automatically printed)
    channel_CoALBP_descriptor_1 = CoALBP(channel_3D, single_charts, channel_name, lbp_r=1, co_r=2) * CoALBP_multiplier
    channel_CoALBP_descriptor_2 = CoALBP(channel_3D, single_charts, channel_name, lbp_r=2, co_r=4) * CoALBP_multiplier
    channel_CoALBP_descriptor_3 = CoALBP(channel_3D, single_charts, channel_name, lbp_r=4, co_r=8) * CoALBP_multiplier

    #Compute LPQ descriptor (automatically printed)
    channel_LPQ_descriptor = lpq(channel, single_charts, channel_name) * LPQ_multiplier

    #Compute FINAL DESCRIPTOR (and print)
    channel_final_descriptor = np.concatenate((channel_lbp_descriptor, channel_CoALBP_descriptor_1,
                                               channel_CoALBP_descriptor_2, channel_CoALBP_descriptor_3,
                                               channel_LPQ_descriptor))

    print("FINAL DESCRIPTOR for channel", channel_name, "successfully computed, total length of:", len(channel_final_descriptor))
    if final_charts:
        histogram_full_image_print(channel_final_descriptor, 'FINAL DESCRIPTOR for channel: %(name)s' % {"name": channel_name})

    return channel_final_descriptor


def channel_descriptor_concatenate(desc1, desc2, desc3=[], desc4=[], desc5=[], desc6=[], n_descriptor=2, final_chart=False):
    """
    Concatenate all the channel descriptors (from 2 to 6 channels)
    :param desc1-6: descriptor for the given channel (from 1 to 6)
    :param n_descriptor: number of descriptor to concatenate
    :param final_chart: plot the final concatenated descriptor (default value FALSE)
    :return: final image descriptor on the given channel
    """
    final_descriptor = []
    if n_descriptor < 2:
        print("ERROR, minimum descriptor number = 2")
        return
    print("---------- STARTING COMPUTATION OF FINAL DESCRIPTOR ----------")
    if n_descriptor == 2:
        final_descriptor = np.concatenate((desc1, desc2))
    elif n_descriptor == 3:
        final_descriptor = np.concatenate((desc1, desc2, desc3))
    elif n_descriptor == 4:
        final_descriptor = np.concatenate((desc1, desc2, desc3, desc4))
    elif n_descriptor == 5:
        final_descriptor = np.concatenate((desc1, desc2, desc3, desc4, desc5))
    elif n_descriptor == 6:
        final_descriptor = np.concatenate((desc1, desc2, desc3, desc4, desc5, desc6))

    print("FINAL DESCRIPTOR successfully computed, total length of:", len(final_descriptor))
    if final_chart:
        histogram_full_image_print(final_descriptor, 'FINAL DESCRIPTOR')

    return final_descriptor


#####################################################################################################################################
def compute_chi_square_distance(hist1, hist2):
    d_chi_square = (hist1 - hist2) ** 2
    hist_sum = hist1 + hist2
    d_chi_square_division = np.divide(d_chi_square, hist_sum, where=(hist_sum != 0))
    d_chi_square = d_chi_square_division.sum()
    d_chi_square = round(d_chi_square, 6)

    return d_chi_square


def histogram_full_image_comparison(hist_real, hist_fake, descriptor_name, real_generator_name, fake_generator_name, 
                                    channel, output_path):
    fig, ax = plt.subplots(1, 2, figsize=(18, 8))
    ax[0].bar(np.arange(len(hist_real)), hist_real, width=12, color="green", alpha=0.5)
    ax[1].bar(np.arange(len(hist_fake)), hist_fake, width=12, color="red", alpha=0.6)
    fig.suptitle(f"{descriptor_name}\n dχ2 = {compute_chi_square_distance(hist_real, hist_fake)}", fontsize=18)
    ax[0].set_title(f"{real_generator_name}", fontsize=16)
    ax[1].set_title(f"{fake_generator_name}", fontsize=16)
    outFullPath = os.path.join(output_path, f"{descriptor_name}_{channel}.png")
    fig.savefig(outFullPath, dpi=fig.dpi)


def get_LBP_Descriptor(channel, n_split, v_min, v_max, scale, single_charts, channel_name, LBP_multiplier):
    #Compute LBP descriptor (Split channel, compute single split LBP, compute single histograms, concatenate, print)
    channel_splits = split_region(channel, n_split)
    channel_splits_lbp = lbp_region_image_compute(channel_splits)
    channel_splits_lbp_hist = histogram_region(channel_splits_lbp, v_min, v_max, scale)
    channel_lbp_descriptor = histogram_concatenate(channel_splits_lbp_hist, single_charts, channel_name) * LBP_multiplier

    return channel_lbp_descriptor


def get_CoALBP_Descriptor(channel_3D, single_charts, channel_name, CoALBP_multiplier):
    #Compute CoALBP descriptors (each one automatically printed)
    channel_CoALBP_descriptor_1 = CoALBP(channel_3D, single_charts, channel_name, lbp_r=1, co_r=2) * CoALBP_multiplier
    channel_CoALBP_descriptor_2 = CoALBP(channel_3D, single_charts, channel_name, lbp_r=2, co_r=4) * CoALBP_multiplier
    channel_CoALBP_descriptor_3 = CoALBP(channel_3D, single_charts, channel_name, lbp_r=4, co_r=8) * CoALBP_multiplier

    return channel_CoALBP_descriptor_1, channel_CoALBP_descriptor_2, channel_CoALBP_descriptor_3


def get_LPQ_Descriptor(channel, single_charts, channel_name, LPQ_multiplier):
    #Compute LPQ descriptor (automatically printed)
    channel_LPQ_descriptor = lpq(channel, single_charts, channel_name) * LPQ_multiplier

    return channel_LPQ_descriptor


def plot_channel_images(img_real, img_fake, fig_title, file_name, fake_generator_name):
        fig = plt.figure(figsize(10, 10))
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(img_real)
        ax1.set_title("Real", fontsize=14)
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.imshow(img_fake)
        ax2.set_title("Fake", fontsize=14)
        fig.suptitle(fig_title, fontsize=16)
        fig.savefig(file_name, dpi=fig.dpi)


def get_channel_images_RF_plots(real_path, fake_path, fake_generator_name, plot_channels=True):
    yiq = 0
    ycrcb = 1
    hsv = 1

    # Read image in BGR
    img_bgr_real = cv2.imread(real_path, 1)
    # img_bgr_real = cv2.resize(img_bgr_real, size, interpolation= cv2.INTER_LINEAR)
    print("BGR image_real created, dimension:", img_bgr_real.shape)
    img_bgr_fake = cv2.imread(fake_path, 1)
    print("BGR image_fake created, dimension:", img_bgr_fake.shape)
    
    # Convert image to RGB
    img_rgb_real = cv2.cvtColor(img_bgr_real, cv2.COLOR_BGR2RGB)
    img_rgb_fake = cv2.cvtColor(img_bgr_fake, cv2.COLOR_BGR2RGB)
    if plot_channels:
        plot_channel_images(img_rgb_real, img_rgb_fake, "RGB IMAGEs", "./img_rgb_rf.png", fake_generator_name)
    print("RGB images created")

    if yiq == 1:
        # Convert image to YIQ
        img_YIQ_real = RGB2YIQ(img_bgr_real)
        img_YIQ_fake = RGB2YIQ(img_bgr_fake)
        if plot_channels:
            plot_channel_images(img_YIQ_real, img_YIQ_fake, "YIQ IMAGEs", "./img_YIQ_rf.png", fake_generator_name)
        print("YIQ images created")

        # Extract single channels Y, I, Q from YIQ
        channel_Y_fromYIQ_real = img_YIQ_real[:, :, 0]
        channel_Y_fromYIQ_3D_real = img_YIQ_real[:, :, 0:1]
        channel_Y_fromYIQ_fake = img_YIQ_fake[:, :, 0]
        channel_Y_fromYIQ_3D_fake = img_YIQ_fake[:, :, 0:1]
        if plot_channels:
            plot_channel_images(channel_Y_fromYIQ_real, channel_Y_fromYIQ_fake, "Channel Y of YIQ", 
                                "./img_YIQ_rf_Y.png", fake_generator_name)
        print("Channel Y extracted")

        channel_I_real = img_YIQ_real[:, :, 1]
        channel_I_3D_real = img_YIQ_real[:, :, 1:2]
        channel_I_fake = img_YIQ_fake[:, :, 1]
        channel_I_3D_fake = img_YIQ_fake[:, :, 1:2]
        if plot_channels:
            plot_channel_images(channel_I_real, channel_I_fake, "Channel I of YIQ", 
                                "./img_YIQ_rf_I.png", fake_generator_name)
        print("Channel I extracted")

        channel_Q_real = img_YIQ_real[:, :, 2]
        channel_Q_3D_real = img_YIQ_real[:, :, 2:3]
        channel_Q_fake = img_YIQ_fake[:, :, 2]
        channel_Q_3D_fake = img_YIQ_fake[:, :, 2:3]
        if plot_channels:
            plot_channel_images(channel_Q_real, channel_Q_fake, "Channel Q of YIQ", 
                                "./img_YIQ_rf_Q.png", fake_generator_name)
        print("Channel Q extracted")

    if ycrcb == 1:
        # Convert image to YCrCb
        img_YCrCb_real = cv2.cvtColor(img_bgr_real, cv2.COLOR_BGR2YCrCb)
        img_YCrCb_fake = cv2.cvtColor(img_bgr_fake, cv2.COLOR_BGR2YCrCb)
        if plot_channels:
            plot_channel_images(img_YCrCb_real, img_YCrCb_fake, "YCrCb IMAGEs", 
                                "./img_YCrCb_rf.png", fake_generator_name)
        print("YCrCb images created")

        # Extract single channels Y, Cr, Cb from YCrCb
        channel_Y_real = img_YCrCb_real[:, :, 0]
        channel_Y_3D_real = img_YCrCb_real[:, :, 0:1]
        channel_Y_fake = img_YCrCb_fake[:, :, 0]
        channel_Y_3D_fake = img_YCrCb_fake[:, :, 0:1]
        if plot_channels:
            plot_channel_images(channel_Y_real, channel_Y_fake, "Channel Y of YCrCb", 
                                "./img_YCrCb_rf_Y.png", fake_generator_name)
        print("Channel Y extracted")

        channel_Cr_real = img_YCrCb_real[:, :, 1]
        channel_Cr_3D_real = img_YCrCb_real[:, :, 1:2]
        channel_Cr_fake = img_YCrCb_fake[:, :, 1]
        channel_Cr_3D_fake = img_YCrCb_fake[:, :, 1:2]
        if plot_channels:
            plot_channel_images(channel_Cr_real, channel_Cr_fake, "Channel Cr of YCrCb", 
                                "./img_YCrCb_rf_Cr.png", fake_generator_name)
        print("Channel Cr extracted")

        channel_Cb_real = img_YCrCb_real[:, :, 2]
        channel_Cb_3D_real = img_YCrCb_real[:, :, 2:3]
        channel_Cb_fake = img_YCrCb_fake[:, :, 2]
        channel_Cb_3D_fake = img_YCrCb_fake[:, :, 2:3]
        if plot_channels:
            plot_channel_images(channel_Cb_real, channel_Cb_fake, "Channel Cb of YCrCb", 
                                "./img_YCrCb_rf_Cb.png", fake_generator_name)
        print("Channel Cb extracted")

    if hsv == 1:
        # Convert image to HSV
        img_HSV_real = cv2.cvtColor(img_bgr_real, cv2.COLOR_BGR2HSV)
        img_HSV_fake = cv2.cvtColor(img_bgr_fake, cv2.COLOR_BGR2HSV)
        if plot_channels:
            plot_channel_images(img_HSV_real, img_HSV_fake, "HSV IMAGEs", "./img_HSV_rf.png", fake_generator_name)
        print("HSV images created")

        # Extract single channels Y, Cr, Cb from YCrCb
        channel_H_real = img_HSV_real[:, :, 0]
        channel_H_3D_real = img_HSV_real[:, :, 0:1]
        channel_H_fake = img_HSV_fake[:, :, 0]
        channel_H_3D_fake = img_HSV_fake[:, :, 0:1]
        if plot_channels:
            plot_channel_images(channel_H_real, channel_H_fake, "Channel H of HSV", 
                                "./img_HSV_rf_H.png", fake_generator_name)
        print("Channel H extracted")

        channel_S_real = img_HSV_real[:, :, 1]
        channel_S_3D_real = img_HSV_real[:, :, 1:2]
        channel_S_fake = img_HSV_fake[:, :, 1]
        channel_S_3D_fake = img_HSV_fake[:, :, 1:2]
        if plot_channels:
            plot_channel_images(channel_S_real, channel_S_fake, "Channel S of HSV", 
                                "./img_HSV_rf_S.png", fake_generator_name)
        print("Channel S extracted")

        channel_V_real = img_HSV_real[:, :, 2]
        channel_V_3D_real = img_HSV_real[:, :, 2:3]
        channel_V_fake = img_HSV_fake[:, :, 2]
        channel_V_3D_fake = img_HSV_fake[:, :, 2:3]
        if plot_channels:
            plot_channel_images(channel_V_real, channel_V_fake, "Channel V of HSV", 
                                "./img_HSV_rf_V.png", fake_generator_name)
        print("Channel V extracted")


def get_image_channels(img_path):
    yiq = 0
    ycrcb = 1
    hsv = 1
    channel_dict = dict()
    
    # Read image in BGR
    img_bgr = cv2.imread(img_path, 1)
    # img_bgr_real = cv2.resize(img_bgr_real, size, interpolation= cv2.INTER_LINEAR)
    # print("BGR image created, dimension:", img_bgr.shape)
    
    # Convert image to RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # print("RGB images created, dimension:", img_rgb.shape)
    
    if yiq == 1:
        # Convert image to YIQ
        img_YIQ = RGB2YIQ(img_bgr)
        # print("YIQ images created")

        # Extract single channels Y, I, Q from YIQ
        channel_Y_fromYIQ = img_YIQ[:, :, 0]
        channel_Y_fromYIQ_3D = img_YIQ[:, :, 0:1]
        channel_dict["YIQ_Y"] = channel_Y_fromYIQ
        channel_dict["YIQ_Y_3D"] = channel_Y_fromYIQ_3D
        # print("Channel Y extracted")

        channel_I = img_YIQ[:, :, 1]
        channel_I_3D = img_YIQ[:, :, 1:2]
        channel_dict["YIQ_I"] = channel_I
        channel_dict["YIQ_I_3D"] = channel_I_3D
        # print("Channel I extracted")

        channel_Q = img_YIQ[:, :, 2]
        channel_Q_3D = img_YIQ[:, :, 2:3]
        channel_dict["YIQ_Q"] = channel_Q
        channel_dict["YIQ_Q_3D"] = channel_Q_3D
        # print("Channel Q extracted")

    if ycrcb == 1:
        # Convert image to YCrCb
        img_YCrCb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
        # print("YCrCb images created")

        # Extract single channels Y, Cr, Cb from YCrCb
        channel_Y = img_YCrCb[:, :, 0]
        channel_Y_3D = img_YCrCb[:, :, 0:1]
        channel_dict["YCrCb_Y"] = channel_Y
        channel_dict["YCrCb_Y_3D"] = channel_Y_3D
        # print("Channel Y extracted")

        channel_Cr = img_YCrCb[:, :, 1]
        channel_Cr_3D = img_YCrCb[:, :, 1:2]
        channel_dict["YCrCb_Cr"] = channel_Cr
        channel_dict["YCrCb_Cr_3D"] = channel_Cr_3D
        # print("Channel Cr extracted")

        channel_Cb = img_YCrCb[:, :, 2]
        channel_Cb_3D = img_YCrCb[:, :, 2:3]
        channel_dict["YCrCb_Cb"] = channel_Cb
        channel_dict["YCrCb_Cb_3D"] = channel_Cb_3D
        # print("Channel Cb extracted")

    if hsv == 1:
        # Convert image to HSV
        img_HSV = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        # print("HSV images created")

        # Extract single channels Y, Cr, Cb from YCrCb
        channel_H = img_HSV[:, :, 0]
        channel_H_3D = img_HSV[:, :, 0:1]
        channel_dict["HSV_H"] = channel_H
        channel_dict["HSV_H_3D"] = channel_H_3D
        # print("Channel H extracted")

        channel_S = img_HSV[:, :, 1]
        channel_S_3D = img_HSV[:, :, 1:2]
        channel_dict["HSV_S"] = channel_S
        channel_dict["HSV_S_3D"] = channel_S_3D
        # print("Channel S extracted")

        channel_V = img_HSV[:, :, 2]
        channel_V_3D = img_HSV[:, :, 2:3]
        channel_dict["HSV_V"] = channel_V
        channel_dict["HSV_V_3D"] = channel_V_3D
        # print("Channel V extracted")
    
    return channel_dict


def get_channels_descriptors(img_path):
    channels_dict = get_image_channels(img_path)
    channels_descriptors = dict()
    
    n_split=3 
    v_min=0
    v_max=256
    scale=1.0
    LBP_multiplier=1
    CoALBP_multiplier=1
    LPQ_multiplier=1
    single_charts=False
    
    channel_name = "Y"
    Y_LBP = get_LBP_Descriptor(channels_dict["YCrCb_Y"], n_split, v_min, v_max, scale, single_charts, channel_name, LBP_multiplier)
    Y_CoALBP_1, Y_CoALBP_2, Y_CoALBP_3 = get_CoALBP_Descriptor(channels_dict["YCrCb_Y_3D"], single_charts, 
                                                               channel_name, CoALBP_multiplier)
    Y_LPQ = get_LPQ_Descriptor(channels_dict["YCrCb_Y"], single_charts, channel_name, LPQ_multiplier)
    channels_descriptors["Y_LBP"] = Y_LBP
    channels_descriptors["Y_CoALBPs"] = np.array([Y_CoALBP_1, Y_CoALBP_2, Y_CoALBP_3])
    channels_descriptors["Y_LPQ"] = Y_LPQ
    
    channel_name = "Cr"
    Cr_LBP = get_LBP_Descriptor(channels_dict["YCrCb_Cr"], n_split, v_min, v_max, scale, single_charts, channel_name, LBP_multiplier)
    Cr_CoALBP_1, Cr_CoALBP_2, Cr_CoALBP_3 = get_CoALBP_Descriptor(channels_dict["YCrCb_Cr_3D"], single_charts, 
                                                               channel_name, CoALBP_multiplier)
    Cr_LPQ = get_LPQ_Descriptor(channels_dict["YCrCb_Cr"], single_charts, channel_name, LPQ_multiplier)
    channels_descriptors["Cr_LBP"] = Cr_LBP
    channels_descriptors["Cr_CoALBPs"] = np.array([Cr_CoALBP_1, Cr_CoALBP_2, Cr_CoALBP_3])
    channels_descriptors["Cr_LPQ"] = Cr_LPQ
    
    channel_name = "Cb"
    Cb_LBP = get_LBP_Descriptor(channels_dict["YCrCb_Cb"], n_split, v_min, v_max, scale, single_charts, channel_name, LBP_multiplier)
    Cb_CoALBP_1, Cb_CoALBP_2, Cb_CoALBP_3 = get_CoALBP_Descriptor(channels_dict["YCrCb_Cb_3D"], single_charts, 
                                                               channel_name, CoALBP_multiplier)
    Cb_LPQ = get_LPQ_Descriptor(channels_dict["YCrCb_Cb"], single_charts, channel_name, LPQ_multiplier)
    channels_descriptors["Cb_LBP"] = Cb_LBP
    channels_descriptors["Cb_CoALBPs"] = np.array([Cb_CoALBP_1, Cb_CoALBP_2, Cb_CoALBP_3])
    channels_descriptors["Cb_LPQ"] = Cb_LPQ

    channel_name = "H"
    H_LBP = get_LBP_Descriptor(channels_dict["HSV_H"], n_split, v_min, v_max, scale, single_charts, channel_name, LBP_multiplier)
    H_CoALBP_1, H_CoALBP_2, H_CoALBP_3 = get_CoALBP_Descriptor(channels_dict["HSV_H_3D"], single_charts, 
                                                               channel_name, CoALBP_multiplier)
    H_LPQ = get_LPQ_Descriptor(channels_dict["HSV_H"], single_charts, channel_name, LPQ_multiplier)
    channels_descriptors["H_LBP"] = H_LBP
    channels_descriptors["H_CoALBPs"] = np.array([H_CoALBP_1, H_CoALBP_2, H_CoALBP_3])
    channels_descriptors["H_LPQ"] = H_LPQ

    channel_name = "S"
    S_LBP = get_LBP_Descriptor(channels_dict["HSV_S"], n_split, v_min, v_max, scale, single_charts, channel_name, LBP_multiplier)
    S_CoALBP_1, S_CoALBP_2, S_CoALBP_3 = get_CoALBP_Descriptor(channels_dict["HSV_S_3D"], single_charts, 
                                                               channel_name, CoALBP_multiplier)
    S_LPQ = get_LPQ_Descriptor(channels_dict["HSV_S"], single_charts, channel_name, LPQ_multiplier)
    channels_descriptors["S_LBP"] = S_LBP
    channels_descriptors["S_CoALBPs"] = np.array([S_CoALBP_1, S_CoALBP_2, S_CoALBP_3])
    channels_descriptors["S_LPQ"] = S_LPQ

    channel_name = "V"
    V_LBP = get_LBP_Descriptor(channels_dict["HSV_V"], n_split, v_min, v_max, scale, single_charts, channel_name, LBP_multiplier)
    V_CoALBP_1, V_CoALBP_2, V_CoALBP_3 = get_CoALBP_Descriptor(channels_dict["HSV_V_3D"], single_charts, 
                                                               channel_name, CoALBP_multiplier)
    V_LPQ = get_LPQ_Descriptor(channels_dict["HSV_V"], single_charts, channel_name, LPQ_multiplier)
    channels_descriptors["V_LBP"] = V_LBP
    channels_descriptors["V_CoALBPs"] = np.array([V_CoALBP_1, V_CoALBP_2, V_CoALBP_3])
    channels_descriptors["V_LPQ"] = V_LPQ

    return channels_descriptors
    

def get_ds_channels_descriptors_task(img_paths, task_id):
    channels_descriptors_ds_dict = None
    for img_path in img_paths:
        if channels_descriptors_ds_dict is None:
            channels_descriptors_ds_dict = get_channels_descriptors(img_path)
        else:
            channels_descriptors_dict = get_channels_descriptors(img_path)
            for descr_key, descr_value in channels_descriptors_ds_dict.items():
                channels_descriptors_ds_dict[descr_key] += channels_descriptors_dict[descr_key]

    return channels_descriptors_ds_dict


def get_ds_channels_descriptors(input_path, task_amount):
    img_paths = []
    for _, _, img_files in os.walk(input_path):
        for img_file in img_files:
            img_paths.append(os.path.join(input_path, img_file))
    # img_paths = img_paths[:3]
    print(f"img_paths length: {len(img_paths)}")
    num_tasks = math.ceil(len(img_paths) / task_amount)
    print(f"num_tasks: {num_tasks}; img_paths length: {len(img_paths)}; task_amount: {task_amount}")
    remainder = len(img_paths) % task_amount
    
    with ProcessPoolExecutor(num_tasks) as executor:
        task_futures = []
        for task_idx in range(1, num_tasks+1, 1):
            if remainder == 0:
                task_futures.append(executor.submit(get_ds_channels_descriptors_task, 
                                                    img_paths[((task_idx-1) * task_amount) : (task_idx * task_amount)],
                                                    task_idx))
            else:
                if task_idx < num_tasks:
                    task_futures.append(executor.submit(get_ds_channels_descriptors_task, 
                                        img_paths[((task_idx-1) * task_amount) : (task_idx * task_amount)], task_idx))
                else:
                    task_futures.append(executor.submit(get_ds_channels_descriptors_task, 
                                        img_paths[((task_idx-1) * task_amount) : ((task_idx-1) * task_amount + remainder)],
                                                       task_idx))

        channels_descriptors_dataset_dict = None
        for task_future in as_completed(task_futures):
            if channels_descriptors_dataset_dict is None:
                channels_descriptors_dataset_dict = task_future.result()
            else:
                channels_descriptors_ds_dict = task_future.result()
                for descr_key, descr_value in channels_descriptors_dataset_dict.items():
                    channels_descriptors_dataset_dict[descr_key] += channels_descriptors_ds_dict[descr_key]

    return channels_descriptors_dataset_dict
    

#####################################################################################################################################


def final_function(path, plot_channels, plot_all_descriptors, plot_final_descriptor):
    """
    Function that merges all the previous methods. First computes the different color spaces and then processes the final
    descriptor
    :param path: path of the image
    :return: final histogram
    """

    yiq = 0
    ycrcb = 1
    hsv = 1

    if plot_final_descriptor:
        plot_final_descriptors = True

    # Read image in BGR
    img_bgr = cv2.imread(path, 1)
    print("BGR image created, dimension:", img_bgr.shape)

    # Convert image to RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    if plot_channels:
        plt.figure()
        plt.imshow(img_rgb)
        plt.title('RGB IMAGE')
        plt.show()
    print("RGB image created")

    if yiq == 1:
        # Convert image to YIQ
        img_YIQ = RGB2YIQ(img_bgr)
        if plot_channels:
            plt.figure()
            plt.imshow(img_YIQ)
            plt.title('YIQ image created')
            plt.show()
        print("YIQ image created")

        # Extract single channels Y, I, Q from YIQ
        channel_Y_fromYIQ = img_YIQ[:, :, 0]
        channel_Y_fromYIQ_3D = img_YIQ[:, :, 0:1]
        if plot_channels:
            plt.figure()
            plt.imshow(channel_Y_fromYIQ)
            plt.title('Channel Y of YIQ')
            plt.show()
        print("Channel Y extracted")

        channel_I = img_YIQ[:, :, 1]
        channel_I_3D = img_YIQ[:, :, 1:2]
        if plot_channels:
            plt.figure()
            plt.imshow(channel_I)
            plt.title('Channel I of YIQ')
            plt.show()
        print("Channel I extracted")

        channel_Q = img_YIQ[:, :, 2]
        channel_Q_3D = img_YIQ[:, :, 2:3]
        if plot_channels:
            plt.figure()
            plt.imshow(channel_Q)
            plt.title('Channel Q of YIQ')
            plt.show()
        print("Channel Q extracted")

    if ycrcb == 1:
        # Convert image to YCrCb
        img_YCrCb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
        if plot_channels:
            plt.figure()
            plt.imshow(img_YCrCb)
            plt.title('YCrCb IMAGE')
            plt.show()
        print("YCrCb image created")

        # Extract single channels Y, Cr, Cb from YCrCb
        channel_Y = img_YCrCb[:, :, 0]
        channel_Y_3D = img_YCrCb[:, :, 0:1]
        if plot_channels:
            plt.figure()
            plt.imshow(channel_Y)
            plt.title('Channel Y of YCrCb')
            plt.show()
        print("Channel Y extracted")

        channel_Cr = img_YCrCb[:, :, 1]
        channel_Cr_3D = img_YCrCb[:, :, 1:2]
        if plot_channels:
            plt.figure()
            plt.imshow(channel_Cr)
            plt.title('Channel Cr of YCrCb')
            plt.show()
        print("Channel Cr extracted")

        channel_Cb = img_YCrCb[:, :, 2]
        channel_Cb_3D = img_YCrCb[:, :, 2:3]
        if plot_channels:
            plt.figure()
            plt.imshow(channel_Cb)
            plt.title('Channel Cb of YCrCb')
            plt.show()
        print("Channel Cb extracted")

    if hsv == 1:
        # Convert image to HSV
        img_HSV = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        if plot_channels:
            plt.figure()
            plt.imshow(img_HSV)
            plt.title('HSV IMAGE')
            plt.show()
        print("HSV image created")

        # Extract single channels Y, Cr, Cb from YCrCb
        channel_H = img_HSV[:, :, 0]
        channel_H_3D = img_HSV[:, :, 0:1]
        if plot_channels:
            plt.figure()
            plt.imshow(channel_H)
            plt.title('Channel H of HSV')
            plt.show()
        print("Channel H extracted")

        channel_S = img_HSV[:, :, 1]
        channel_S_3D = img_HSV[:, :, 1:2]
        if plot_channels:
            plt.figure()
            plt.imshow(channel_S)
            plt.title('Channel S of HSV')
            plt.show()
        print("Channel S extracted")

        channel_V = img_HSV[:, :, 2]
        channel_V_3D = img_HSV[:, :, 2:3]
        if plot_channels:
            plt.figure()
            plt.imshow(channel_V)
            plt.title('Channel V of HSV')
            plt.show()
        print("Channel V extracted")

    # DESCRIPTORS COMPUTATION AND CONCATENATION

    Y_descriptor = channel_descriptor(channel_Y, channel_Y_3D, "Y", 3, 0, 256, final_charts=plot_all_descriptors)
    Cr_descriptor = channel_descriptor(channel_Cr, channel_Cr_3D, "Cr", 3, 0, 256, final_charts=plot_all_descriptors)
    Cb_descriptor = channel_descriptor(channel_Cb, channel_Cb_3D, "Cb", 3, 0, 256, final_charts=plot_all_descriptors)
    H_descriptor = channel_descriptor(channel_H, channel_H_3D, "H", 3, 0, 256, final_charts=plot_all_descriptors)
    S_descriptor = channel_descriptor(channel_S, channel_S_3D, "S", 3, 0, 256, final_charts=plot_all_descriptors)
    V_descriptor = channel_descriptor(channel_V, channel_V_3D, "V", 3, 0, 256, final_charts=plot_all_descriptors)
    final_result = channel_descriptor_concatenate(Y_descriptor, Cr_descriptor, Cb_descriptor, H_descriptor, S_descriptor,
                                                  V_descriptor, 6, final_chart=plot_final_descriptor)

    return final_result