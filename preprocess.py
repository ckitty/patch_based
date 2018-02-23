import numpy as np
import cv2
from tqdm import tqdm


def gamma_transform(img, gamma):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)


def random_gamma_transform(img, gamma_vari):
    log_gamma_vari = np.log(gamma_vari)
    alpha = np.random.uniform(-log_gamma_vari, log_gamma_vari)
    gamma = np.exp(alpha)
    return gamma_transform(img, gamma)


def rotate(xb, yb, angle):
    M_rotate = cv2.getRotationMatrix2D((65 / 2, 65 / 2), angle, 1)
    xb = cv2.warpAffine(xb, M_rotate, (65, 65))
    yb = cv2.warpAffine(yb, M_rotate, (65, 65))
    return xb, yb


def blur(img):
    img = cv2.blur(img, (3, 3))
    return img


def add_noise(img):
    for i in range(50):
        temp_x = np.random.randint(0, img.shape[0])
        temp_y = np.random.randint(0, img.shape[1])
        img[temp_x][temp_y] = 255
    return img


def data_augment(xb):

    if np.random.random() < 0.25:
        xb = cv2.flip(xb, 1)


    if np.random.random() < 0.25:
        xb = random_gamma_transform(xb, 1.0)

    if np.random.random() < 0.25:
        xb = blur(xb)

    if np.random.random() < 0.2:
        xb = add_noise(xb)

    return xb


def createtrain(num=1000, time=3):
    image_sets = ['data/train/1.png', 'data/train/2.png']
    label_sets = ['data/train/1_class.png', 'data/train/2_class.png']
    print('creating train...')
    g_count = 0
    for i in range(2):
        each = num//2
        src_img = cv2.imread(image_sets[i])
        label_img = cv2.imread(label_sets[i], cv2.IMREAD_GRAYSCALE)
        label_patch = label_img[32:label_img.shape[0]-32,32:label_img.shape[1]-32]
        for class_ in tqdm(range(5)):
            index0, index1 = np.where(label_patch==class_)
            random_indexes = np.random.randint(0,index0.shape[0],size=[each //5]).tolist()
            index0 = index0[random_indexes]
            index1 = index1[random_indexes]
            for piex in range(len(index0)):
                for times in range(time):
                    sub_image = src_img[index0[piex]:index0[piex]+65,index1[piex]:index1[piex]+65]
                    sub_image = data_augment(sub_image)
                    cv2.imwrite(('dataset/train/images/%06d-%d.png' % (g_count,class_)), sub_image)
                    g_count += 1


def createtest(num = 1000):
    g_count = 0
    image_dir = 'data/test/test3.png'
    label_dir = 'data/test/test3_labels_8bits.png'
    print('creating test...')
    src_img = cv2.imread(image_dir)
    label_img = cv2.imread(label_dir, cv2.IMREAD_GRAYSCALE)
    label_patch = label_img[32:label_img.shape[0] - 32, 32:label_img.shape[1] - 32]
    for class_ in tqdm(range(5)):
        index0, index1 = np.where(label_patch == class_)
        random_indexes = np.random.randint(0, index0.shape[0], size=[num //5]).tolist()
        index0 = index0[random_indexes]
        index1 = index1[random_indexes]
        for piex in range(len(index0)):
            sub_image = src_img[index0[piex]:index0[piex] + 65, index1[piex]:index1[piex] + 65]
            cv2.imwrite(('dataset/test/images/%06d-%d.png' % (g_count, class_)), sub_image)
            g_count += 1

if __name__ == '__main__':
    createtest()
    createtrain()