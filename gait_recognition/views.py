import traceback

from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .inference_utils import GaitSetInference
import tempfile
import os

# 全局加载模型
model = GaitSetInference("gaitset_model.onnx")

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

            # 执行识别
            result = model.recognize(image_paths)

            # 返回结果（这里简化为返回嵌入向量） # TODO 应该返回ID。处理“嵌入向量”，转变为ID
            return Response({
                'status': 'success',
                'embedding': result.tolist()  # 将numpy数组转换为JSON可序列化的列表
            })

    except Exception as e:
        # 捕获并打印异常 # TODO 供Debug
        print("捕获到异常：", e)
        traceback.print_exc()
        # 返回错误
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
