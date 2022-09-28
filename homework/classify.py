# 图像识别
import numpy as np
import cv2
import os
import pandas as pd

# fmt:off
path = "D:\\DigitalImageProcessing\\img\\classify\\train\\"
kind_list = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# fmt:on
def get_image_similarity(image1, image2):
    """基于欧氏距离的图片相似度"""
    image_similarity = np.sum(np.abs(image1 - image2))
    return image_similarity


def get_image_similarity_by_hist(image1, image2):
    """基于直方图的图片相似度"""
    hist1 = cv2.calcHist([image1], [0], None, [256], [0, 255])
    hist2 = cv2.calcHist([image2], [0], None, [256], [0, 255])
    image_similarity = np.sum(np.abs(hist1 - hist2))
    return image_similarity


def detect_image_kind(test_image):
    """寻找相似的图片"""
    similarity_list = np.zeros((10000, 3))  # 第一列为灰度相似度，第二列为直方图相似度，第三列为图片类型
    n = 0
    for kind in kind_list:
        filename_list = os.listdir(path + kind)
        for filename in filename_list:
            file_path = path + kind + "\\" + filename
            train_image = cv2.imread(file_path, 0)
            image_similarity1 = get_image_similarity(test_image, train_image)
            image_similarity2 = get_image_similarity_by_hist(test_image, train_image)
            similarity_list[n, 0] = image_similarity1
            similarity_list[n, 1] = image_similarity2
            similarity_list[n, 2] = kind_list.index(kind)
            n = n + 1
    gray_similarity_list = np.array(sorted(similarity_list, key=lambda x: x[0]))  # 升序排序
    hist_similarity_list = np.array(sorted(similarity_list, key=lambda x: x[1]))  # 升序排序

    # 比较前10相似度
    detect_kinds = []
    for i in range(10):
        normalized1 = gray_similarity_list[i, 0] / max(
            gray_similarity_list[:, 0]
        )  # 归一化灰度相似度
        normalized2 = hist_similarity_list[i, 1] / max(
            hist_similarity_list[:, 1]
        )  # 归一化直方图相似度
        if normalized1 < normalized2:
            kind = kind_list[int(gray_similarity_list[i][2])]
        else:
            kind = kind_list[int(hist_similarity_list[i][2])]

        # kind = list(kind_id.keys())[
        #         list(kind_id.values()).index(hist_similarity_list[i][2])
        #     ]  # 由值找键
        detect_kinds.append(kind)
    count_res = pd.value_counts(detect_kinds)
    detect_kind = list(count_res.keys())[0]
    return detect_kind


if __name__ == "__main__":
    correct_num = 0
    error_num = 0
    N = 200  # 检测n张照片
    for kind_index in range(N):
        test_kind = kind_list[np.random.randint(10)]  # 随机类别
        test_path = "D:\\DigitalImageProcessing\\img\\classify\\val\\" + test_kind
        filename = os.listdir(test_path)[np.random.randint(450)]  # 随机图片
        image = cv2.imread(test_path + "\\" + filename, 0)
        detect_kind = detect_image_kind(image)
        if test_kind == detect_kind:
            correct_num = correct_num + 1
        else:
            error_num = error_num + 1
        print(
            "测试"
            + str(correct_num + error_num)
            + "  测试类型：["
            + test_kind
            + "]  测试图片"
            + filename
            + "  测试结果："
            + detect_kind
        )
    correct_rate = correct_num / (correct_num + error_num)
    print("\n共测试" + str(N) + "张图片，识别正确率：" + str(correct_rate * 100) + "%\n")
