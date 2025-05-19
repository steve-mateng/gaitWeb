import onnxruntime as ort
import numpy as np
import cv2
import os
from pathlib import Path

import faiss

from gait_recognition.models import GaitFeature


class GaitSetInference:
    def __init__(self, model_path):
        """
        初始化ONNX推理器
        :param model_path: ONNX模型路径
        """
        # 获取当前文件目录
        current_dir = Path(__file__).parent
        # 构建完整模型路径
        self.model_path = os.path.join(current_dir, 'models', 'gaitset_model.onnx')

        # 初始化推理会话
        self.ort_session = ort.InferenceSession(self.model_path)

        # 验证模型输入输出
        self._validate_io()

    def _validate_io(self):
        """验证模型输入输出格式"""
        # 获取输入信息
        self.input_name = self.ort_session.get_inputs()[0].name
        input_shape = self.ort_session.get_inputs()[0].shape

        # 检查输入维度是否符合预期 [N, C, S, H, W]
        if len(input_shape) != 5 or input_shape[1] != 1:
            raise ValueError(f"Invalid input shape {input_shape}, expected [N, 1, S, H, W]")

        # 获取输出信息
        output = self.ort_session.get_outputs()[0]
        output_shape = output.shape
        print(f"Model loaded: Input shape {input_shape}, Output shape {output_shape}")

    def preprocess_image(self, image_path):
        """
        图像预处理：读取并调整图像尺寸到模型要求
        :param image_path: 图像路径
        :return: 处理后的图像数组
        """
        # 读取图像（灰度图）
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # 调整尺寸到64x44（H x W），保持宽高比
        resized_img = cv2.resize(img, (44, 64), interpolation=cv2.INTER_AREA)

        # 归一化到[0,1]范围
        normalized_img = resized_img.astype(np.float32) / 255.0

        return normalized_img

    def prepare_input_sequence(self, image_paths):
        """
        准备输入序列：将多个轮廓图像组成一个序列
        :param image_paths: 图像路径列表
        :return: 处理后的输入张量
        """
        # 预处理所有图像
        processed_images = [self.preprocess_image(path) for path in image_paths]

        # 转换为numpy数组 [S, H, W]
        sequence = np.stack(processed_images, axis=0)  # TODO fix: 输入维度是否符合预期 [N, C, S, H, W], S：序列长度sequences为30吗？
        # 添加通道维度 [S, H, W] -> [C=1, S, H, W]
        sequence = np.expand_dims(sequence, axis=0)

        # 修改：调整维度顺序为 [C, S, H, W] 以符合模型要求
        # 或者直接调整最终张量的维度顺序
        # sequence = np.transpose(sequence, (1, 0, 2, 3))  # 如果需要 [C, S, H, W]  TODO 似乎多余，上一步“添加通道维度”axis=0即可

        # 添加batch维度 [N=1, C=1, S, H, W]
        batch_input = np.expand_dims(sequence, axis=0)

        return batch_input.astype(np.float32)

    def recognize(self, image_paths):
        """
        执行步态识别
        :param image_paths: 图像路径列表
        :return: 识别结果（嵌入向量）
        """
        # 准备输入数据
        input_data = self.prepare_input_sequence(image_paths)

        # 执行推理
        outputs = self.ort_session.run(None, {self.input_name: input_data})

        # 返回第一个输出（嵌入向量）
        return outputs[0]


class FeatureMatcher:
    def __init__(self, dimension=256, similarity_threshold=0.85):
        """初始化FAISS索引并加载已知特征

        Args:
            dimension: 特征向量维度
            similarity_threshold: 相似度阈值
        """
        self.dimension = dimension
        self.similarity_threshold = similarity_threshold

        # 初始化FAISS索引
        self.index = faiss.IndexFlatL2(dimension)  # 使用L2距离

        # 加载已知特征
        self._load_known_features()

    def _load_known_features(self):
        """从数据库加载所有已知特征到FAISS索引"""
        features = GaitFeature.objects.all()

        if not features:
            # 如果没有注册用户，创建一个虚拟向量
            dummy_vector = np.zeros((1, self.dimension), dtype=np.float32)
            self.index.add(dummy_vector)
            return

        # 转换为numpy数组并添加到索引
        vectors = np.vstack([f.get_feature().reshape(1, -1) for f in features])
        self.index.add(vectors)

        # 保存特征与用户ID的映射
        self.id_map = {i: f.user_id for i, f in enumerate(features)}
        self.id_map[-1] = None  # 默认未知用户

    def match(self, query_vector):
        """执行特征匹配查询

        Args:
            query_vector: 查询特征向量

        Returns:
            tuple: (user_id, similarity_score)
        """
        # 确保查询向量是正确的形状
        if query_vector.shape != (1, self.dimension):  # TODO dimension需要手动设置吗？如何知道dimension是多少？
            query_vector = query_vector.reshape(1, -1)

        # 执行FAISS搜索
        distances, indices = self.index.search(query_vector, 1)

        # 计算相似度分数（余弦相似度）
        distance = distances[0][0]
        index = indices[0][0]

        # L2距离转相似度
        similarity = 1 / (1 + distance)

        # 如果相似度低于阈值，视为未知用户
        if similarity < self.similarity_threshold:
            return None, similarity

        return self.id_map[index], similarity
