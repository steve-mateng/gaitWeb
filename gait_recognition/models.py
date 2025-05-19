from django.db import models
from django.conf import settings
import numpy as np


class GaitFeature(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    feature_vector = models.BinaryField()
    created_at = models.DateTimeField(auto_now_add=True)

    def set_feature(self, vector):
        """将numpy数组转换为二进制存储"""
        self.feature_vector = vector.tobytes()

    def get_feature(self):
        """将二进制数据转换回numpy数组"""
        return np.frombuffer(self.feature_vector, dtype=np.float32)
