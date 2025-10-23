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


def load_image(img_path: str, L=False):
    """
    加载图片

    :param img_path: 图片路径
    :param L: 是否转为灰度图
    :return: 图片矩阵
    """
    image = Image.open(img_path)
    if L:
        image = image.convert("L")
    image = np.array(image)
    return image


def load_image_hog(img_path: str):
    """
    加载图像的HOG特征

    :param img_path: 图片路径
    :return: HOG特征
    """
    image = cv2.resize(load_image(img_path, True), winSize)
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


def train(
    dataset_dir: str,
    train_dir="train_64x128_H96",
    test_dir="test_64x128_H96",
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
        os.path.join(train_dir, "pos"), os.path.join(train_dir, "neg")
    )

    # 训练SVM分类器
    print("训练中")
    model = LinearSVC(
        dual=False,  # 当样本数 < 特征数时（HOG情况），使用 dual=False 效果更好
        tol=1e-9,  # 容忍度, 较小值获得更精确的边界
        C=0.05,  # 正则化强度: 越大越少正则化，模型越贴近训练数据
        verbose=1,  # 显示训练过程
        max_iter=10000,  # 最大迭代次数
    )
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


def detect(image: np.ndarray, model: LinearSVC):
    global hog
    # svm_detector = np.append(model.coef_, -model.intercept_).astype(np.float32)
    svm_detector = np.append(model.coef_.ravel(), -model.intercept_[0]).astype(
        np.float32
    )
    hog.setSVMDetector(svm_detector)

    # svm: cv2.ml.SVM = model
    # sv = svm.getSupportVectors()  # shape = (nSV, feat_len)
    # rho, alpha, svidx = svm.getDecisionFunction(0)
    # w = np.dot(alpha, sv).ravel()
    # svm_detector = np.append(w, rho).astype(np.float32)
    # hog.setSVMDetector(svm_detector)

    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    (rects, weights) = hog.detectMultiScale(
        image,
        winStride=(8, 8),
        padding=(8, 8),
        scale=1.05,
        hitThreshold=0,
        groupThreshold=2,
    )
    boxes = [(int(x), int(y), int(x + w), int(y + h)) for (x, y, w, h) in rects]
    print(boxes)
    return boxes


def test_show(model: LinearSVC, img: str, limit=3):
    """
    测试并显示检测结果

    :param model: 训练好的SVM分类器
    :param img: 测试图像路径 (或测试图像文件夹)
    :param limit: 如果img是文件夹, 最大显示图像数
    """
    if os.path.isdir(img):
        # 随机选取limit张图片进行测试
        img_paths = os.listdir(img)  # 所有图片路径
        img_paths = np.random.choice(img_paths, limit, replace=False)
        test_images: "list[np.ndarray]" = []
        for img_path in img_paths:
            test_image = load_image(os.path.join(img, img_path))
            if test_image is not None:
                test_images.append(test_image)
        assert len(test_images) > 0, "测试图像不存在"
    else:
        test_image = load_image(img)
        assert test_image is not None, "测试图像不存在"
        test_images = [test_image]

    for idx, test_image in enumerate(test_images):
        boxes = detect(test_image, model)

        img = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        # 绘制检测框
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255 * (i / len(boxes)), 0), 2)

        cv2.imshow(f"Detection {idx}", img)
        cv2.waitKey(1)
    while cv2.waitKey(100) < 0:
        pass
    cv2.destroyAllWindows()


if __name__ == "__main__":  # 主函数

    def main():
        ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ASSETS_DIR = os.path.join(ROOT_DIR, "assets")
        dataset_dir = os.path.join(ASSETS_DIR, "INRIAPerson")
        save_path = os.path.join(ASSETS_DIR, "hog_svm_model.pkl")

        # 训练模型 / 加载模型
        model = train(dataset_dir, save=save_path)
        # model = load_model(save_path)

        # 测试并显示检测结果
        # test_show(model, img=os.path.join(dataset_dir, "Test/pos"), limit=5)  # 多张
        test_show(model, os.path.join(dataset_dir, "Test/pos/person_014.png"))  # 指定

    main()
