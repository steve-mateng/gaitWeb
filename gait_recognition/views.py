import traceback

import numpy as np

from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework import status
from .inference_utils import GaitSetInference, FeatureMatcher
import tempfile
import os

from rest_framework.permissions import IsAuthenticated
from .models import GaitFeature

# 全局加载模型和匹配器
model = GaitSetInference("gaitset_model.onnx")
matcher = FeatureMatcher()  # 确保数据库已迁移后再启用


@api_view(['POST'])
def recognize_gait(request):
    """
    步态识别API端点
    接收上传的轮廓图像序列，返回识别结果
    """
    # 检查是否有文件上传
    if 'images' not in request.FILES:
        return Response({'error': 'No images provided'}, status=status.HTTP_400_BAD_REQUEST)

    try:
        # 获取上传的文件列表
        image_files = request.FILES.getlist('images')  # TODO attention

        # 创建临时目录保存上传的文件
        with tempfile.TemporaryDirectory() as temp_dir:
            # 保存所有上传的图像文件
            image_paths = []
            for image_file in image_files:
                # 保存到临时目录，并直接记录本地路径
                file_path = os.path.join(temp_dir, image_file.name)
                with open(file_path, 'wb+') as destination:
                    for chunk in image_file.chunks():
                        destination.write(chunk)
                image_paths.append(file_path)  # 使用本地路径，不经过 default_storage

            # 执行识别，获取嵌入向量
            # 获取 embedding 并做平均
            embedding = model.recognize(image_paths)
            avg_embedding = np.mean(embedding, axis=2).squeeze()  # shape: (256,)
            print("Avg embedding shape:", avg_embedding.shape)  # TODO debug

            # 执行特征匹配
            user_id, similarity = matcher.match(avg_embedding)

            # 返回结果
            result = {
                'status': 'success',
                'recognized': True if user_id else False,
                'similarity': float(similarity),
                'embedding': embedding.tolist()  # 保留嵌入向量供调试使用
            }

            if user_id:
                result['user_id'] = user_id
            else:
                result['message'] = 'Unknown person'

            return Response(result)

    except Exception as e:
        # 捕获并打印异常 # TODO 供Debug
        print("捕获到异常：", e)
        traceback.print_exc()
        # 返回错误
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['POST'])
@permission_classes([IsAuthenticated])  # TODO attention 如何起作用的？
def register_gait(request):
    """
    注册新用户的步态特征
    需要用户认证
    """
    if 'images' not in request.FILES:
        return Response({'error': 'No images provided'}, status=status.HTTP_400_BAD_REQUEST)

    try:
        # 获取上传的文件列表
        image_files = request.FILES.getlist('images')

        # 创建临时目录保存上传的文件
        with tempfile.TemporaryDirectory() as temp_dir:
            # 保存所有上传的图像文件
            image_paths = []
            for image_file in image_files:
                file_path = os.path.join(temp_dir, image_file.name)
                with open(file_path, 'wb+') as destination:
                    for chunk in image_file.chunks():
                        destination.write(chunk)
                image_paths.append(file_path)

            # 执行识别
            # 获取 embedding 并做平均
            embedding = model.recognize(image_paths)
            avg_embedding = np.mean(embedding, axis=2).squeeze()  # shape: (256,)
            print("Avg embedding shape:", avg_embedding.shape)  # TODO debug

            # 存储特征
            feature = GaitFeature(user=request.user)
            feature.set_feature(avg_embedding)
            feature.save()

            # 重新加载匹配器中的特征
            matcher._load_known_features()

            # 返回成功响应
            return Response({
                'status': 'success',
                'message': 'Gait feature registered successfully',
                'embedding_dim': embedding.shape
            })

    except Exception as e:
        # 捕获并打印异常
        print("捕获到异常：", e)
        traceback.print_exc()
        # 返回错误
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

