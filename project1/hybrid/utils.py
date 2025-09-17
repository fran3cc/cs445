import time
import os

import cv2
import argparse
import numpy as np
from scipy import signal
from math import ceil, floor
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def align_images(input_img_1, input_img_2, pts_img_1, pts_img_2,
                 save_images=False):
    
    # Load images
    im1 = cv2.imread(input_img_1)
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)

    im2 = cv2.imread(input_img_2)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)

    # get image sizes
    h1, w1, b1 = im1.shape
    h2, w2, b2 = im2.shape

    # Get center coordinate of the line segment
    center_im1 = np.mean(pts_img_1, axis=0)
    center_im2 = np.mean(pts_img_2, axis=0)

    plt.close('all')

    # translate first so that center of ref points is center of image
    tx = np.around((w1 / 2 - center_im1[0]) * 2).astype(int)

    if tx > 0:
        im1 = np.r_['1', np.zeros((im1.shape[0], tx, 3)), im1]

    else:
        im1 = np.r_['1', im1, np.zeros((im1.shape[0], -tx, 3))]

    ty = np.round((h1 / 2 - center_im1[1]) * 2).astype(int)

    if ty > 0:
        im1 = np.r_['0', np.zeros((ty, im1.shape[1], 3)), im1]

    else:
        im1 = np.r_['0', im1, np.zeros((-ty, im1.shape[1], 3))]

    tx = np.around((w2 / 2 - center_im2[0]) * 2).astype(int)

    if tx > 0:
        im2 = np.r_['1', np.zeros((im2.shape[0], tx, 3)), im2]

    else:
        im2 = np.r_['1', im2, np.zeros((im2.shape[0], -tx, 3))]

    ty = np.round((h2 / 2 - center_im2[1]) * 2).astype(int)

    if ty > 0:
        im2 = np.r_['0', np.zeros((ty, im2.shape[1], 3)), im2]

    else:
        im2 = np.r_['0', im2, np.zeros((-ty, im2.shape[1], 3))]

    # downscale larger image so that lengths between ref points are the same
    len1 = np.linalg.norm(pts_img_1[0]-pts_img_1[1])
    len2 = np.linalg.norm(pts_img_2[0]-pts_img_2[1])
    dscale = len2 / len1

    if dscale < 1:
        width = int(im1.shape[1] * dscale)
        height = int(im1.shape[0] * dscale)
        dim = (width, height)
        im1 = cv2.resize(im1, dim, interpolation=cv2.INTER_LINEAR)

    else:
        width = int(im2.shape[1] * 1 / dscale)
        height = int(im2.shape[0] * 1 / dscale)
        dim = (width, height)
        im2 = cv2.resize(im2, dim, interpolation=cv2.INTER_LINEAR)

    # rotate im1 so that angle between points is the same
    theta1 = np.arctan2(-(pts_img_1[:, 1][1]-pts_img_1[:, 1][0]),
                        pts_img_1[:, 0][1]-pts_img_1[:, 0][0])
    theta2 = np.arctan2(-(pts_img_2[:, 1][1]-pts_img_2[:, 1][0]),
                        pts_img_2[:, 0][1]-pts_img_2[:, 0][0])
    dtheta = theta2-theta1
    rows, cols = im1.shape[:2]
    M = cv2.getRotationMatrix2D((cols/2, rows/2), dtheta*180/np.pi, 1)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((rows * sin) + (cols * cos))
    nH = int((rows * cos) + (cols * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cols/2
    M[1, 2] += (nH / 2) - rows/2

    im1 = cv2.warpAffine(im1, M, (nW, nH))

    # Crop images (on both sides of border) to be same height and width
    h1, w1, b1 = im1.shape
    h2, w2, b2 = im2.shape

    minw = min(w1, w2)
    brd = (max(w1, w2)-minw)/2
    if minw == w1:  # crop w2
        im2 = im2[:, ceil(brd):-floor(brd), :]
        tx = tx-ceil(brd)
    else:
        im1 = im1[:, ceil(brd):-floor(brd), :]
        tx = tx+ceil(brd)

    minh = min(h1, h2)
    brd = (max(h1, h2)-minh)/2
    if minh == h1:  # crop w2
        im2 = im2[ceil(brd):-floor(brd), :, :]
        ty = ty-ceil(brd)
    else:
        im1 = im1[ceil(brd):-floor(brd), :, :]
        ty = ty+ceil(brd)

    im1 = cv2.cvtColor(im1.astype(np.uint8), cv2.COLOR_RGB2BGR)
    im2 = cv2.cvtColor(im2.astype(np.uint8), cv2.COLOR_RGB2BGR)

    if save_images:
        output_img_1 = 'aligned_{}'.format(os.path.basename(input_img_1))
        output_img_2 = 'aligned_{}'.format(os.path.basename(input_img_2))
        cv2.imwrite(output_img_1, im1)
        cv2.imwrite(output_img_2, im2)

    return im1, im2


def prompt_eye_selection(image):
    fig = plt.figure()
    plt.imshow(image, cmap='gray')
    fig.set_label('Click on two points for alignment')
    plt.axis('off')
    xs = []
    ys = []
    clicked = np.zeros((2, 2), dtype=np.float32)

    # Define a callback function that will update the textarea
    def onmousedown(event):
        x = event.xdata
        y = event.ydata
        xs.append(x)
        ys.append(y)

        plt.plot(xs, ys, 'r-+')

    def onmouseup(event):
        if(len(xs) >= 2):
            plt.close(fig)

    def onclose(event):
        clicked[:, 0] = xs
        clicked[:, 1] = ys
    # Create an hard reference to the callback not to be cleared by the garbage
    # collector
    fig.canvas.mpl_connect('button_press_event', onmousedown)
    fig.canvas.mpl_connect('button_release_event', onmouseup)
    fig.canvas.mpl_connect('close_event', onclose)

    return clicked

def crop_image(image, points):
    points = points.astype(int)
    ys = points[:,1]
    xs = points[:,0]
    if len(image.shape)==2:
        image = image[int(ys[0]):int(ys[1]), int(xs[0]):int(xs[1])]
    else:
        image = image[int(ys[0]):int(ys[1]), int(xs[0]):int(xs[1]),:]

    return image

def interactive_crop(image):
    
    fig = plt.figure()
    plt.imshow(image, cmap='gray')
    fig.set_label('Click upper-left and lower-right corner to crop')
    plt.axis('off')
    xs = []
    ys = []
    clicked = np.zeros((2, 2), dtype=np.float32)
    cropped_image = np.zeros_like(image)
    return_object = {
        'cropped_image': None,
        'crop_bound': None
    }

    # Define a callback function that will update the textarea
    def onmousedown(event):
        x = event.xdata
        y = event.ydata
        xs.append(x)
        ys.append(y)
        if(len(xs) >= 2):
            clicked[:, 0] = xs
            clicked[:, 1] = ys
            cropped = crop_image(image, clicked)
            return_object['crop_bound'] = clicked
            return_object['cropped_image'] = cropped
            plt.clf()
            plt.imshow(cropped, cmap='gray')
            plt.axis('off')
        else:
            plt.plot(xs, ys, 'r+')
                
            
    def onmouseup(event):
        if(len(xs) >= 2):
            plt.close(fig)

    # Create an hard reference to the callback not to be cleared by the garbage
    # collector
    fig.canvas.mpl_connect('button_press_event', onmousedown)
    fig.canvas.mpl_connect('button_release_event', onmouseup)
    return return_object

def gaussian_kernel(sigma, kernel_half_size):
    '''
    Inputs:
        sigma = standard deviation for the gaussian kernel
        kernel_half_size = recommended to be at least 3*sigma
    
    Output:
        Returns a 2D Gaussian kernel matrix
    '''
    window_size = kernel_half_size*2+1
    gaussian_kernel_1d = signal.gaussian(window_size, std=sigma).reshape(window_size, 1)
    gaussian_kernel_2d = np.outer(gaussian_kernel_1d, gaussian_kernel_1d)
    gaussian_kernel_2d /= np.sum(gaussian_kernel_2d) # make sure it sums to one

    return gaussian_kernel_2d


def plot(array, filename=None):
    # plots gray images
    plt.imshow(array, cmap='gray') 
    plt.axis('off')
    if filename:
        array=np.clip(array,0,1)
        array=(array*255).astype(np.uint8)
        cv2.imwrite(filename, array)
        
        
def plot_spectrum(magnitude_spectrum):
    # A logarithmic colormap
    plt.imshow(magnitude_spectrum, norm=LogNorm(vmin=1/5)) #,vmax=10e1
    plt.colorbar()


def align_images_manual(im1, im2, pts1, pts2):
    """
    Align two images using manually specified eye coordinates with improved size handling.
    
    Args:
        im1: First image (RGB)
        im2: Second image (RGB)
        pts1: Eye coordinates for first image [left_eye_x, left_eye_y, right_eye_x, right_eye_y]
        pts2: Eye coordinates for second image [left_eye_x, left_eye_y, right_eye_x, right_eye_y]
    
    Returns:
        Aligned versions of both images with same dimensions
    """
    # Convert coordinate format to numpy arrays
    pts_img_1 = np.array([[pts1[0], pts1[1]], [pts1[2], pts1[3]]], dtype=np.float32)
    pts_img_2 = np.array([[pts2[0], pts2[1]], [pts2[2], pts2[3]]], dtype=np.float32)
    
    # Get center coordinate of the line segment
    center_im1 = np.mean(pts_img_1, axis=0)
    center_im2 = np.mean(pts_img_2, axis=0)
    
    # Get image sizes
    h1, w1 = im1.shape[:2]
    h2, w2 = im2.shape[:2]
    
    # Calculate eye distances for scaling
    len1 = np.linalg.norm(pts_img_1[0] - pts_img_1[1])
    len2 = np.linalg.norm(pts_img_2[0] - pts_img_2[1])
    
    # Determine target size - use the larger of the two images as reference
    target_eye_distance = max(len1, len2)
    scale1 = target_eye_distance / len1 if len1 > 0 else 1.0
    scale2 = target_eye_distance / len2 if len2 > 0 else 1.0
    
    # Scale images to have same eye distance
    new_w1 = int(w1 * scale1)
    new_h1 = int(h1 * scale1)
    new_w2 = int(w2 * scale2)
    new_h2 = int(h2 * scale2)
    
    im1_scaled = cv2.resize(im1, (new_w1, new_h1), interpolation=cv2.INTER_LINEAR)
    im2_scaled = cv2.resize(im2, (new_w2, new_h2), interpolation=cv2.INTER_LINEAR)
    
    # Update eye coordinates after scaling
    pts_img_1_scaled = pts_img_1 * scale1
    pts_img_2_scaled = pts_img_2 * scale2
    
    # Calculate new centers
    center_im1_scaled = np.mean(pts_img_1_scaled, axis=0)
    center_im2_scaled = np.mean(pts_img_2_scaled, axis=0)
    
    # Determine final canvas size - use the larger dimensions
    final_w = max(new_w1, new_w2) + 200  # Add padding for translation
    final_h = max(new_h1, new_h2) + 200
    
    # Create canvases and place images centered
    canvas1 = np.zeros((final_h, final_w, 3), dtype=im1.dtype)
    canvas2 = np.zeros((final_h, final_w, 3), dtype=im2.dtype)
    
    # Calculate placement positions to center the eye centers
    target_center_x = final_w // 2
    target_center_y = final_h // 2
    
    # Calculate translation needed to center the eye centers
    tx1 = int(target_center_x - center_im1_scaled[0])
    ty1 = int(target_center_y - center_im1_scaled[1])
    tx2 = int(target_center_x - center_im2_scaled[0])
    ty2 = int(target_center_y - center_im2_scaled[1])
    
    # Apply translation using cv2.warpAffine
    M1 = np.float32([[1, 0, tx1], [0, 1, ty1]])
    M2 = np.float32([[1, 0, tx2], [0, 1, ty2]])
    
    im1_final = cv2.warpAffine(im1_scaled, M1, (final_w, final_h))
    im2_final = cv2.warpAffine(im2_scaled, M2, (final_w, final_h))
    
    # Crop to remove excess padding while keeping images centered
    # Find the bounding box that contains both images
    gray1 = cv2.cvtColor(im1_final, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(im2_final, cv2.COLOR_RGB2GRAY)
    
    # Find non-zero regions
    coords1 = np.column_stack(np.where(gray1 > 0))
    coords2 = np.column_stack(np.where(gray2 > 0))
    
    if len(coords1) > 0 and len(coords2) > 0:
        # Get bounding box that includes both images
        min_y = min(coords1[:, 0].min(), coords2[:, 0].min())
        max_y = max(coords1[:, 0].max(), coords2[:, 0].max())
        min_x = min(coords1[:, 1].min(), coords2[:, 1].min())
        max_x = max(coords1[:, 1].max(), coords2[:, 1].max())
        
        # Add some padding
        padding = 50
        min_y = max(0, min_y - padding)
        max_y = min(final_h, max_y + padding)
        min_x = max(0, min_x - padding)
        max_x = min(final_w, max_x + padding)
        
        # Crop both images to the same region
        im1_final = im1_final[min_y:max_y, min_x:max_x]
        im2_final = im2_final[min_y:max_y, min_x:max_x]
    
    # Ensure both images have exactly the same dimensions
    h_final, w_final = im1_final.shape[:2]
    im2_final = cv2.resize(im2_final, (w_final, h_final), interpolation=cv2.INTER_LINEAR)
    
    return im1_final, im2_final
