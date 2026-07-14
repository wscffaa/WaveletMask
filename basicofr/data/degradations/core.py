"""
核心退化函数

从 degradation.py 提取的活跃退化函数。
历史版本（v1-v3、debug）已归档到 _legacy/degradation_legacy.py。

主要函数：
- standard_degradation_pipeline: degradation_video_list_4 的规范命名版
- degradation_video_list_4: 向后兼容别名
- degradation_video_list_4_one_channel: RRTN 灰度版
- degradation_video_list_5: 分级退化版
"""

from .blend_modes import addition, multiply, subtract
import cv2
from PIL import Image, ImageEnhance
import random
from io import BytesIO
import os
import numpy as np
from scipy import ndimage
import scipy.stats as ss

from .texture import texture_generator, moving_line_texture_generator

_TEXTURE_CACHE = {}

def _cached_texture_files(path, only_001=False):
    """缓存纹理文件列表，避免每次退化都重复 os.walk。"""
    key = (os.path.abspath(path), only_001)
    if key in _TEXTURE_CACHE:
        return _TEXTURE_CACHE[key]

    files = []
    for dirpath, _, filenames in os.walk(path):
        if only_001 and not dirpath.endswith("001"):
            continue
        for name in filenames:
            files.append(os.path.join(dirpath, name))
    files.sort()
    _TEXTURE_CACHE[key] = files
    return files


def texture_blending(image, texture, folder_name, verbose=False): ## Input: PIL | Return: PIL

    ## Keep the same dimension  TODO: More augmentation
    w,h=image.size
    # texture=texture.resize((h,w))
    # texture=texture_augmentation(texture,h,w)
    texture = texture_generator(texture, folder_name, h, w)
    ##


    image = np.array(image.convert('RGBA')).astype(float)
    texture = np.array(texture.convert('RGBA')).astype(float)

    mode = random.randint(0, 2)
    if folder_name == "011":
        mode = 0
    if verbose:
        distortion_type = ['addition', 'subtract', 'multiply']
        print(distortion_type[mode])

    if mode==0:
        scratched_image=addition(image,texture,opacity=random.uniform(0.6, 1.0))
    elif mode==1:
        scratched_image=subtract(image,texture,opacity=random.uniform(0.6, 1.0))
    elif mode==2:
        scratched_image=multiply(image,texture,opacity=random.uniform(0.6, 1.0))

    scratched_image = np.uint8(scratched_image)[:, :, :3]
    x=Image.fromarray(scratched_image).convert("L")

    return x


def moving_line_texture_blending(image, texture, last_texture, folder_name, mode, verbose=False): ## Input: PIL | Return: PIL

    ## Keep the same dimension  TODO: More augmentation
    w,h=image.size
    # texture=texture.resize((h,w))
    # texture=texture_augmentation(texture,h,w)
    texture_P = moving_line_texture_generator(texture, last_texture, h, w)
    ##


    image = np.array(image.convert('RGBA')).astype(float)
    texture = np.array(texture_P.convert('RGBA')).astype(float)

    if verbose:
        distortion_type = ['addition', 'subtract', 'multiply']
        print(distortion_type[mode])

    if mode==0:
        scratched_image=addition(image,texture,opacity=random.uniform(0.6, 1.0))
    elif mode==1:
        scratched_image=subtract(image,texture,opacity=random.uniform(0.6, 1.0))
    elif mode==2:
        scratched_image=multiply(image,texture,opacity=random.uniform(0.6, 1.0))

    scratched_image = np.uint8(scratched_image)[:, :, :3]
    x=Image.fromarray(scratched_image).convert("L")

    return x, texture_P


def pil_to_np(img_PIL):
    '''Converts image in PIL format to np.array.
    From W x H x C [0...255] to C x W x H [0..1]
    '''
    ar = np.array(img_PIL)

    if len(ar.shape) == 3:
        ar = ar.transpose(2, 0, 1)
    else:
        ar = ar[None, ...]

    return ar.astype(np.float32) / 255.


def np_to_pil(img_np):
    '''Converts image in np.array format to PIL image.
    From C x W x H [0..1] to  W x H x C [0...255]
    '''
    ar = np.clip(img_np * 255, 0, 255).astype(np.uint8)

    if img_np.shape[0] == 1:
        ar = ar[0]
    else:
        ar = ar.transpose(1, 2, 0)

    return Image.fromarray(ar)


def apply_color_jitter(image, brightness_range, contrast_range, saturation_range, always_apply=False, p=0.5):
    """Lightweight replacement for albumentations ColorJitter."""
    if not always_apply and random.random() > p:
        return image

    img = image.convert("RGB")

    if brightness_range[0] != brightness_range[1]:
        factor = random.uniform(brightness_range[0], brightness_range[1])
        img = ImageEnhance.Brightness(img).enhance(factor)

    if contrast_range[0] != contrast_range[1]:
        factor = random.uniform(contrast_range[0], contrast_range[1])
        img = ImageEnhance.Contrast(img).enhance(factor)

    if saturation_range[0] != saturation_range[1]:
        factor = random.uniform(saturation_range[0], saturation_range[1])
        img = ImageEnhance.Color(img).enhance(factor)

    return img


def jpeg_artifact_v2(image, quality, verbose=False):
    image = np.array(image)
    image[image >= 255] = 255
    image[image < 0] = 0
    image = image.astype(np.uint8)
    image = Image.fromarray(image)

    quality_variance = random.randint(-15, 15)
    new_quality = np.clip(quality+quality_variance,40,100)

    with BytesIO() as f:
        image.save(f, format='JPEG', quality=int(new_quality))
        f.seek(0)
        image_jpeg = Image.open(f).convert('L')

    if verbose:
        print('JPEG quality =', new_quality)

    return image_jpeg


def random_scaling(img,x,y):

    down_method = ['bicubic','bilinear','lanczos']
    selected_method = random.sample(down_method,1)[0]
    if selected_method == 'bicubic':
        img=img.resize((x,y),Image.BICUBIC)
    if selected_method == 'bilinear':
        img=img.resize((x,y),Image.BILINEAR)
    if selected_method == 'lanczos':
        img=img.resize((x,y),Image.LANCZOS)
    
    return img


def downsampling_artifact_v3_fixed(img,params,verbose=False):
    w,h=img.size

    rnum = params['rnum']
    if rnum > 0.8:  # up
        sf1 = params['up_scale']
    elif rnum < 0.7:  # down
        sf1 = params['down_scale']
    else:
        sf1 = 1.0

    new_w = int(sf1 * w)
    new_h = int(sf1 * h)

    img = random_scaling(img,new_w,new_h)
    # img = random_scaling(img,w,h)

    if verbose:
        print('down-sampling size =(%d,%d)'%(new_w,new_h))

    return img


def blur_artifact_v2(img, kernel_size, std, verbose=False):


    x=np.array(img)
    # kernel_size_candidate=[(3,3),(5,5),(7,7)]
    # kernel_size=random.sample(kernel_size_candidate,1)[0]
    # std=random.uniform(1.,5.)

    std_variance=random.uniform(-1.,1.)
    new_std = np.clip(std + std_variance, 1., 5.)

    #print("The gaussian kernel size: (%d,%d) std: %.2f"%(kernel_size[0],kernel_size[1],std))
    blur=cv2.GaussianBlur(x,kernel_size,new_std)

    if verbose:
        print("Blur kernel =",kernel_size)

    return Image.fromarray(blur.astype(np.uint8))


def gm_blur_kernel(mean, cov, size=15):
    center = size / 2.0 + 0.5
    k = np.zeros([size, size])
    for y in range(size):
        for x in range(size):
            cy = y - center + 1
            cx = x - center + 1
            k[y, x] = ss.multivariate_normal.pdf([cx, cy], mean=mean, cov=cov)

    k = k / np.sum(k)
    return k


def anisotropic_Gaussian(ksize=15, theta=np.pi, l1=6, l2=6):
    """ generate an anisotropic Gaussian kernel
    Args:
        ksize : e.g., 15, kernel size
        theta : [0,  pi], rotation angle range
        l1    : [0.1,50], scaling of eigenvalues
        l2    : [0.1,l1], scaling of eigenvalues
        If l1 = l2, will get an isotropic Gaussian kernel.
    Returns:
        k     : kernel
    """
    v = np.dot(np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]), np.array([1., 0.]))
    V = np.array([[v[0], v[1]], [v[1], -v[0]]])
    D = np.array([[l1, 0], [0, l2]])
    Sigma = np.dot(np.dot(V, D), np.linalg.inv(V))
    k = gm_blur_kernel(mean=[0, 0], cov=Sigma, size=ksize)

    return k


def fspecial_gaussian(hsize, sigma):
    hsize = [hsize, hsize]
    siz = [(hsize[0]-1.0)/2.0, (hsize[1]-1.0)/2.0]
    std = sigma
    [x, y] = np.meshgrid(np.arange(-siz[1], siz[1]+1), np.arange(-siz[0], siz[0]+1))
    arg = -(x*x + y*y)/(2*std*std)
    h = np.exp(arg)
    h[h < np.finfo(float).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h = h/sumh
    return h


def fspecial(filter_type, *args, **kwargs):
    '''
    python code from:
    https://github.com/ronaldosena/imagens-medicas-2/blob/40171a6c259edec7827a6693a93955de2bd39e76/Aulas/aula_2_-_uniform_filter/matlab_fspecial.py
    '''
    if filter_type == 'gaussian':
        return fspecial_gaussian(*args, **kwargs)


def add_blur_fixed(img,params):

    wd2 = 4.0 + 4
    wd = 2.0 + 0.2*4

    if params['type_value'] < 0.5:
        l1 = wd2 * float(np.clip(params['l1_value']+(random.random()-0.5)/10.,1e-8,1-1e-8))
        l2 = wd2 * float(np.clip(params['l2_value']+(random.random()-0.5)/10.,1e-8,1-1e-8))
        k = anisotropic_Gaussian(ksize=2*params['shape_value']+3, theta=float(np.clip(params['angle_value']+(random.random()-0.5)/5.,1e-8,1-1e-8))*np.pi, l1=l1, l2=l2)
    else:
        k = fspecial('gaussian', 2*params['shape_value']+3, wd * float(np.clip(params['l1_value']+(random.random()-0.5)/10.,1e-8,1-1e-8)))
    img = ndimage.filters.convolve(img, np.expand_dims(k, axis=2), mode='mirror')

    return img


def gaussian_noise_artifact_v2(image,std,verbose=False):

    ## Give PIL, return the noisy PIL

    img_pil=pil_to_np(image)

    mean=0
    # std=random.uniform(std_l/255.,std_r/255.)
    std_variance=random.uniform(-0.5, 0.5)
    new_std = np.clip(std + std_variance/255. , 5.0/255., 10.0/255.)

    gauss=np.random.normal(loc=mean,scale=new_std,size=img_pil.shape)
    noisy=img_pil+gauss
    noisy=np.clip(noisy,0,1).astype(np.float32)

    if verbose:
        print("Gaussian noise std =", new_std)

    return np_to_pil(noisy)


def speckle_noise_artifact_v2(image,std,verbose=False):

    ## Give PIL, return the noisy PIL

    img_pil=pil_to_np(image)

    mean=0
    # std=random.uniform(std_l/255.,std_r/255.)
    std_variance=random.uniform(-0.5, 0.5)
    new_std = np.clip(std + std_variance/255. , 5.0/255., 10.0/255.)

    gauss=np.random.normal(loc=mean,scale=new_std,size=img_pil.shape)
    noisy=img_pil+gauss*img_pil
    noisy=np.clip(noisy,0,1).astype(np.float32)

    if verbose:
        print("Speckle noise std =", new_std)

    return np_to_pil(noisy)


def degradation_v3(image, distortion_sequence, distortion_degree, distortion_probability, verbose=False):

    P=1.0
    x,y = image.size
    distortion_types = ['blur', 'downsample', 'noise', 'jpeg']
    for i,distortion_index in enumerate(distortion_sequence):
        distortion = distortion_types[i]
        if distortion == 'blur' and distortion_probability[i] < P:
            temp_cv2 = transfer_2(image)
            # image = blur_artifact_v2(image,distortion_degree['blur_kernel'],distortion_degree['blur_std'], verbose)
            temp_cv2 = add_blur_fixed(temp_cv2,distortion_degree)
            # temp_cv2 = add_blur(temp_cv2)
            image = transfer_1(temp_cv2)
        elif distortion == 'noise' and distortion_probability[i] < P:
            noise_type = random.choice([1, 2])
            if noise_type==1:
                image = gaussian_noise_artifact_v2(image, distortion_degree['noise_std'], verbose)
            elif noise_type==2:
                image = speckle_noise_artifact_v2(image, distortion_degree['noise_std'], verbose)
        elif distortion == 'jpeg' and distortion_probability[i] < P: 
            image = jpeg_artifact_v2(image, distortion_degree['jpeg_quality'] , verbose)
        elif distortion == 'downsample':
            image = downsampling_artifact_v3_fixed(image,distortion_degree,verbose=False)

    image = random_scaling(image,x,y)
    return image


def color_jitter(image, verbose=False):

    jittered = apply_color_jitter(
        image,
        brightness_range=(0.8, 1.2),
        contrast_range=(0.9, 1.0),
        saturation_range=(1.0, 1.0),
        always_apply=True,
        p=0.5,
    )

    return jittered.convert('L')


def transfer_1(img_np):
    """BGR float32 -> PIL 灰度，尽量减少多余拷贝。"""
    ar = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
    if ar.ndim == 2:
        return Image.fromarray(ar, mode='L')
    rgb = ar[..., ::-1]
    return Image.fromarray(rgb, mode='RGB').convert("L")


def transfer_2(img_pil):
    """PIL（RGB/灰度）-> BGR float32。"""
    arr = np.asarray(img_pil, dtype=np.uint8)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.shape[-1] == 4:
        arr = arr[:, :, :3]
    return arr[..., ::-1].astype(np.float32) / 255.


def add_sharpening(img, weight=0.5, radius=50, threshold=10):
    """USM sharpening. borrowed from real-ESRGAN
    Input image: I; Blurry image: B.
    1. K = I + weight * (I - B)
    2. Mask = 1 if abs(I - B) > threshold, else: 0
    3. Blur mask:
    4. Out = Mask * K + (1 - Mask) * I
    Args:
        img (Numpy array): Input image, HWC, BGR; float32, [0, 1].
        weight (float): Sharp weight. Default: 1.
        radius (float): Kernel size of Gaussian blur. Default: 50.
        threshold (int):
    """
    if radius % 2 == 0:
        radius += 1
    blur = cv2.GaussianBlur(img, (radius, radius), 0)
    residual = img - blur
    mask = np.abs(residual) > threshold / 255.0  # 阈值归一化到[0,1]，residual为[0,1]范围的浮点
    mask = mask.astype('float32')
    soft_mask = cv2.GaussianBlur(mask, (radius, radius), 0)

    K = img + weight * residual
    K = np.clip(K, 0, 1)
    return soft_mask * K + (1 - soft_mask) * img


def standard_degradation_pipeline(video_list,texture_url='./noise_data'): ## Add moving lines

    texture_templates=getfilelist(texture_url)
    # print(texture_templates)
    degraded=[]
    gt_L=[]

    ## For each video, fix the core degradation pattern, then disturb among frames
    distortion_types = ['blur', 'noise', 'jpeg', 'downsample']
    distortion_sequence = np.random.permutation(len(distortion_types))
    distortion_degree = {'type_value': random.random(), 'l1_value': random.random(), 'l2_value': random.random(),'angle_value': random.random(), 'shape_value': random.randint(2,11), 'noise_std': random.uniform(5.0/255.,10.0/255.), 'jpeg_quality': random.randint(40, 100), 'rnum': np.random.rand(), 'up_scale': random.uniform(1, 2), 'down_scale': random.uniform(0.5/4, 1)}
    distortion_probability = [random.uniform(0, 1) for i in range(4)]
    ##

    moving_line_flag = False
    if random.uniform(0, 1)<0.2:
        moving_line_flag = True  
        texture_templates=getfilelist_001(texture_url)
        selected_texture_url=random.sample(texture_templates,1)[0]
        texture_pil=Image.open(selected_texture_url).convert("L") ##  Initial line texture
        mode = random.randint(0, 2) ## pre-define the blending mode
        # print("Yes, Moving Line is training")
        last_texture=None

    for x in video_list:
        
        frame_pil = transfer_1(x)
        frame_cv2 = transfer_2(frame_pil.convert("RGB"))
        GT_cv2 = add_sharpening(frame_cv2)

        # frame_pil=Image.fromarray(x.astype('uint8')).convert("L")
        if not moving_line_flag: 
            selected_texture_url=random.sample(texture_templates,1)[0]
            folder_name = selected_texture_url.split('/')[-2]
            texture_pil=Image.open(selected_texture_url).convert("L")
            x1=texture_blending(frame_pil,texture_pil,folder_name)
        else:
            x1, last_texture=moving_line_texture_blending(frame_pil, texture_pil, last_texture,'001', mode)


        x2=degradation_v3(x1, distortion_sequence, distortion_degree, distortion_probability)
        x3=color_jitter(x2)

        x3=x3.convert("RGB")
        degraded.append(transfer_2(x3))

        gt_L.append(GT_cv2)

    return degraded,gt_L


def _sample_texture_from_bank(texture_pool):
    """从磁盘路径或缓存列表中取纹理，返回 (folder, PIL.Image)。"""
    texture_choice = random.choice(texture_pool)
    if isinstance(texture_choice, str):
        folder_name = os.path.basename(os.path.dirname(texture_choice))
        with Image.open(texture_choice) as img:
            texture = img.convert("L")
    else:
        folder_name, texture_img = texture_choice
        texture = texture_img.copy()
    return folder_name, texture


def degradation_video_list_5(video_list,
                             degree=1,
                             texture_url='./noise_data',
                             texture_bank=None,
                             texture_bank_001=None):
    """按退化强度分级（0/1/2）生成合成 LQ-GT 对，可复用缓存纹理减少 IO。"""

    texture_templates = texture_bank if texture_bank else getfilelist(texture_url)
    if not texture_templates:
        raise ValueError(f'纹理模板目录为空，请检查路径: {texture_url}')

    texture_templates_001 = texture_bank_001 if texture_bank_001 else getfilelist_001(texture_url)
    degraded = []
    gt_L = []

    distortion_types = ['blur', 'noise', 'jpeg', 'downsample']
    distortion_sequence = np.random.permutation(len(distortion_types))
    distortion_degree1 = {
        'type_value': random.random(),
        'l1_value': random.random(),
        'l2_value': random.random(),
        'angle_value': random.random(),
        'shape_value': random.randint(2, 5),
        'noise_std': random.uniform(5.0 / 255., 6.0 / 255.),
        'jpeg_quality': random.randint(80, 100),
        'rnum': np.random.rand(),
        'up_scale': random.uniform(1, 1.5),
        'down_scale': random.uniform(0.5, 1)
    }
    distortion_degree2 = {
        'type_value': random.random(),
        'l1_value': random.random(),
        'l2_value': random.random(),
        'angle_value': random.random(),
        'shape_value': random.randint(5, 8),
        'noise_std': random.uniform(6.0 / 255., 8.0 / 255.),
        'jpeg_quality': random.randint(60, 80),
        'rnum': np.random.rand(),
        'up_scale': random.uniform(1, 2),
        'down_scale': random.uniform(0.5 / 2, 1)
    }
    distortion_degree3 = {
        'type_value': random.random(),
        'l1_value': random.random(),
        'l2_value': random.random(),
        'angle_value': random.random(),
        'shape_value': random.randint(8, 11),
        'noise_std': random.uniform(8.0 / 255., 10.0 / 255.),
        'jpeg_quality': random.randint(40, 60),
        'rnum': np.random.rand(),
        'up_scale': random.uniform(1, 2),
        'down_scale': random.uniform(0.5 / 4, 1)
    }
    distortion_degree_candidates = [distortion_degree1, distortion_degree2, distortion_degree3]
    if degree < 0 or degree >= len(distortion_degree_candidates):
        raise ValueError(f'degree 需在 0~2 之间，收到 {degree}')
    distortion_degree = distortion_degree_candidates[degree]
    distortion_probability = [1, 1, 1, 1]

    moving_line_flag = random.uniform(0, 1) < 0.2 and bool(texture_templates_001)
    if moving_line_flag:
        _, texture_pil = _sample_texture_from_bank(texture_templates_001)
        mode = random.randint(0, 2)
        last_texture = None
    else:
        texture_pil = None
        last_texture = None
        mode = 0

    for frame in video_list:

        frame_pil = transfer_1(frame)
        frame_cv2 = transfer_2(frame_pil.convert("RGB"))
        gt_L.append(frame_cv2)

        if not moving_line_flag:
            folder_name, texture_pil = _sample_texture_from_bank(texture_templates)
            blended = texture_blending(frame_pil, texture_pil, folder_name)
        else:
            blended, last_texture = moving_line_texture_blending(
                frame_pil, texture_pil, last_texture, '001', mode)

        degraded_frame = degradation_v3(
            blended, distortion_sequence, distortion_degree, distortion_probability)
        degraded_frame = color_jitter(degraded_frame)

        degraded.append(transfer_2(degraded_frame.convert("RGB")))

    return degraded, gt_L


def degradation_video_list_4_one_channel(video_list, texture_url='./noise_data'):
    """灰度版退化：保持旧 RRTN 的单通道流程。"""

    texture_templates = getfilelist(texture_url)
    degraded = []
    gt_l = []

    distortion_types = ['blur', 'noise', 'jpeg', 'downsample']
    distortion_sequence = np.random.permutation(len(distortion_types))
    distortion_degree = {
        'type_value': random.random(),
        'l1_value': random.random(),
        'l2_value': random.random(),
        'angle_value': random.random(),
        'shape_value': random.randint(2, 11),
        'noise_std': random.uniform(5.0 / 255., 10.0 / 255.),
        'jpeg_quality': random.randint(40, 100),
        'rnum': np.random.rand(),
        'up_scale': random.uniform(1, 2),
        'down_scale': random.uniform(0.5 / 4, 1)
    }
    distortion_probability = [random.uniform(0, 1) for _ in range(4)]

    moving_line_flag = False
    if random.uniform(0, 1) < 0.2:
        moving_line_flag = True
        texture_templates = getfilelist_001(texture_url)
        selected_texture_url = random.sample(texture_templates, 1)[0]
        texture_pil = Image.open(selected_texture_url).convert('L')
        mode = random.randint(0, 2)
        last_texture = None
    else:
        texture_pil = None
        mode = 0
        last_texture = None

    for frame in video_list:
        frame_pil = transfer_1(frame)
        frame_np = np.array(frame_pil) / 255.
        gt_l.append(np.expand_dims(np.clip(add_sharpening(frame_np), 0, 1), -1))

        if not moving_line_flag:
            selected_texture_url = random.sample(texture_templates, 1)[0]
            folder_name = selected_texture_url.split('/')[-2]
            texture_pil = Image.open(selected_texture_url).convert('L')
            blended = texture_blending(frame_pil, texture_pil, folder_name)
        else:
            blended, last_texture = moving_line_texture_blending(
                frame_pil, texture_pil, last_texture, '001', mode)

        degraded_frame = degradation_v3(
            blended, distortion_sequence, distortion_degree, distortion_probability)
        degraded.append(
            np.expand_dims(
                np.clip(np.array(color_jitter(degraded_frame)) / 255., 0, 1), -1))

    return degraded, gt_l


def getfilelist_001(path):
    """仅返回末级目录名为 001 的纹理列表（带缓存）。"""
    return list(_cached_texture_files(path, only_001=True))


degradation_video_list_4 = standard_degradation_pipeline


def getfilelist(path):
    """返回纹理列表（带缓存）。"""
    return list(_cached_texture_files(path, only_001=False))


__all__ = [
    '_cached_texture_files',
    'texture_blending',
    'moving_line_texture_blending',
    'pil_to_np',
    'np_to_pil',
    'apply_color_jitter',
    'jpeg_artifact_v2',
    'random_scaling',
    'downsampling_artifact_v3_fixed',
    'blur_artifact_v2',
    'gm_blur_kernel',
    'anisotropic_Gaussian',
    'fspecial_gaussian',
    'fspecial',
    'add_blur_fixed',
    'gaussian_noise_artifact_v2',
    'speckle_noise_artifact_v2',
    'degradation_v3',
    'color_jitter',
    'transfer_1',
    'transfer_2',
    'add_sharpening',
    'standard_degradation_pipeline',
    'degradation_video_list_4',
    '_sample_texture_from_bank',
    'degradation_video_list_5',
    'degradation_video_list_4_one_channel',
    'getfilelist_001',
    'getfilelist',
]
