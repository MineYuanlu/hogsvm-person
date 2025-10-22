import joblib
import cv2
from PIL import Image
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import tqdm
import os

winSize = (64, 128)  # HOG窗口大小
hog = cv2.HOGDescriptor(
    _winSize=winSize,  # HOG窗口大小
    _blockSize=(16, 16),  # 块大小，为2*2个细胞
    _blockStride=(8, 8),  # 块步长，为块大小的一半
    _cellSize=(8, 8),  # 细胞大小
    _nbins=9,  # 梯度方向的数量
)


def load_image_hog(img_path: str):
    global hog
    image = np.array(Image.open(img_path).convert("L"))
    image = cv2.resize(image, winSize)
    hog_feature = hog.compute(image)
    return hog_feature


def load_dataset(pos_dir: "str|list[str]", neg_dir: "str|list[str]"):
    """
    加载数据集

    :param pos_dir: 正样本目录
    :param neg_dir: 负样本目录
    :return: 特征矩阵(X)和标签列表(y)
    """
    features = []
    labels = []

    def get_files(dirs: "str|list[str]"):
        return [
            os.path.join(dir, fn)
            for dir in (dirs if isinstance(dirs, list) else [dirs])
            for fn in os.listdir(dir)
        ]

    for pos_file in tqdm.tqdm(get_files(pos_dir), desc="pos"):
        features.append(load_image_hog(pos_file))
        labels.append(1)
    for neg_file in tqdm.tqdm(get_files(neg_dir), desc="neg"):
        features.append(load_image_hog(neg_file))
        labels.append(0)

    return np.array(features), np.array(labels)


def detect_pedestrian(
    image: np.ndarray,
    model: LinearSVC,
    scale=1.05,
    window_step=8,
    svm_threshold=0.5,
    nms_threshold=0.3,
    min_size=(64, 128),
    max_layers=200,
):
    """
    HOG+SVM 行人检测函数（多尺度金字塔+滑动窗口+NMS）

    :param image: 图像
    :param model: 模型
    :param scale: 金字塔缩放因子
    :param window_step: 滑动窗口步长 (像素)
    :param svm_threshold: SVM决策函数阈值
    :param nms_threshold: NMS重叠阈值
    :param min_size: 最小图像尺寸停止金字塔
    :param max_layers: 金字塔最大层数
    """

    boxes: "list[tuple[float,float,float,float]]" = []
    confidences: "list[float]" = []

    original_h, original_w = image.shape[:2]

    # 创建图像金字塔
    pyramid_images: "list[tuple[np.ndarray, float]]" = []

    current_scale = 0.5
    layers = 0
    while layers < max_layers:
        resized_h = int(original_h / current_scale)
        resized_w = int(original_w / current_scale)
        if resized_h < min_size[1] or resized_w < min_size[0]:
            break
        resized = cv2.resize(image, (resized_w, resized_h))
        pyramid_images.append((resized, current_scale))
        current_scale *= scale
        layers += 1

    # 遍历金字塔
    confidences: "list[float]" = []
    for pyramid, pyr_scale in tqdm.tqdm(pyramid_images):
        h, w = pyramid.shape[:2]
        for y in range(0, h - winSize[1] + 1, window_step):
            for x in range(0, w - winSize[0] + 1, window_step):
                window = pyramid[y : y + winSize[1], x : x + winSize[0]]
                if window.shape[0] != winSize[1] or window.shape[1] != winSize[0]:
                    continue
                # 灰度化+归一化
                gray = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY)
                # gray = gray.astype(np.float32) / 255.0
                hog_feat = hog.compute(gray).reshape(1, -1)
                confidence = model.decision_function(hog_feat)[0]
                confidences.append(confidence)
                if confidence > svm_threshold:
                    # 还原到原图坐标
                    x1 = int(x * pyr_scale)
                    y1 = int(y * pyr_scale)
                    x2 = int((x + winSize[0]) * pyr_scale)
                    y2 = int((y + winSize[1]) * pyr_scale)
                    boxes.append((x1, y1, x2, y2))
                    confidences.append(float(confidence))

    # 绘制Plot: confidences
    import matplotlib.pyplot as plt

    plt.plot(confidences)
    plt.show()

    # 非极大值抑制
    if boxes:
        indices = cv2.dnn.NMSBoxes(
            bboxes=boxes,
            scores=confidences,
            score_threshold=svm_threshold,
            nms_threshold=nms_threshold,
        )
        indices = indices.flatten()
        return [boxes[i] for i in indices]
    else:
        return []


def train(
    dataset_dir: str,
    train_dir="train_64x128_H96",
    test_dir="test_64x128_H96",
    # train_dir="Train",
    # test_dir="Test",
    save="hog_svm_model.pkl",
):
    """
    进行训练，并保存模型

    :param dataset_dir: INRIAPerson数据集目录
    :param train_dir: 训练集目录
    :param test_dir: 测试集目录
    :param save: 模型保存路径
    """
    train_dir = os.path.join(dataset_dir, train_dir)
    test_dir = os.path.join(dataset_dir, test_dir)

    # 加载训练集
    print("加载数据集")
    x_train, y_train = load_dataset(
        [os.path.join(train_dir, "pos"), os.path.join(test_dir, "pos")],
        [
            os.path.join(train_dir, "neg"),
            #   os.path.join(test_dir, "neg")
        ],
    )

    # 训练SVM分类器
    print("训练中")
    model = LinearSVC()
    model.fit(x_train, y_train)

    # 保存模型
    print("保存模型")
    joblib.dump(model, save)

    # 加载测试集
    print("加载测试集")
    x_test, y_test = load_dataset(
        os.path.join(test_dir, "pos"), os.path.join(test_dir, "neg")
    )

    # 在测试集上评估
    y_pred = model.predict(x_test)
    print(classification_report(y_test, y_pred))
    return model


def load_model(path="hog_svm_model.pkl") -> LinearSVC:
    """
    加载训练好的模型

    :param path: 模型保存路径
    :return: 训练好的SVM分类器
    """
    return joblib.load(path)


def test_show(model: LinearSVC, img: str, limit=3):
    """
    测试并显示检测结果

    :param model: 训练好的SVM分类器
    :param img: 测试图像路径 (或测试图像文件夹)
    :param limit: 如果img是文件夹, 最大显示图像数
    """
    # 测试检测函数
    if os.path.isdir(img):
        # 随机选取limit张图片进行测试
        img_paths = sorted(os.listdir(img), key=lambda x: np.random.random())
        test_images: "list[np.ndarray]" = []
        for img_path in img_paths:
            test_image = cv2.imread(os.path.join(img, img_path))
            if test_image is not None:
                test_images.append(test_image)
                if len(test_images) >= limit:
                    break
        assert len(test_images) > 0, "测试图像不存在"

    else:
        test_image = cv2.imread(img)
        assert test_image is not None, "测试图像不存在"
        test_images = [test_image]

    for idx, test_image in enumerate(test_images):
        boxes = detect_pedestrian(test_image, model)

        # 绘制检测框
        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(test_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow(f"Detection {idx}", test_image)
        cv2.waitKey(1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":  # 主函数

    def main():
        ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ASSETS_DIR = os.path.join(ROOT_DIR, "assets")
        dataset_dir = os.path.join(ASSETS_DIR, "INRIAPerson")
        save_path = os.path.join(ASSETS_DIR, "hog_svm_model.pkl")

        # 训练模型 / 加载模型
        # model = train(dataset_dir, save=save_path)
        model = load_model(save_path)

        # 测试并显示检测结果
        # test_show(model, img=os.path.join(dataset_dir, "Test/pos"), limit=1)
        test_show(model, os.path.join(dataset_dir, "Test/pos/crop_000023.png"))

    main()
