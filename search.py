import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import logging
import numpy as np
import faiss
import torch
import boto3
from PIL import Image
from io import BytesIO
from typing import List
from transformers import ViTModel, ViTImageProcessor

class ImageSearchEngine:
    def __init__(self, s3_config):
        self.s3_config = s3_config
        self.s3 = self._init_s3_client()
        
        # Инициализация модели
        self.model_name = "google/vit-base-patch16-224"
        self.feature_extractor = ViTImageProcessor.from_pretrained(self.model_name)
        self.model = ViTModel.from_pretrained(
            self.model_name,
            add_pooling_layer=False
        )
        self.model.eval()
        
        # Кэш индексов
        self.index_cache = {}

    def _init_s3_client(self):
        return boto3.client(
            service_name='s3',
            endpoint_url=self.s3_config.endpoint_url,
            aws_access_key_id=self.s3_config.aws_access_key_id,
            aws_secret_access_key=self.s3_config.aws_secret_access_key,
            region_name='us-east-1'
        )

    def search_similar(self, image_bytes: bytes, top_k: int = 5) -> List[dict]:
        if not self.index_cache:
            self._build_index()
            
        index_data = self.index_cache["global_index"]
        
        # Поиск по индексу
        query_features = self._extract_features_from_bytes(image_bytes)
        distances, indices = index_data['index'].search(query_features, top_k)
        
        return [{
            "product_id": index_data['product_ids'][i],
            "similarity_score": 1 / (1 + distances[0][j]),
            "image_url": f"{self.s3_config.endpoint_url}/{self.s3_config.bucket_name}/{index_data['product_ids'][i]}"
        } for j, i in enumerate(indices[0]) if i != -1]

    def _build_index(self):
        try:
            # Получаем список всех объектов в бакете
            response = self.s3.list_objects_v2(
                Bucket=self.s3_config.bucket_name
            )
            
            if 'Contents' not in response:
                raise ValueError("No images found in bucket")
                
            features = []
            product_ids = []
            
            for obj in response['Contents']:
                key = obj['Key']
                if key.lower().endswith(('.png', '.jpg', '.jpeg')):
                    try:
                        img_bytes = self._download_image(key)
                        features.append(self._extract_features_from_bytes(img_bytes))
                        product_id = os.path.splitext(key)[0]
                        product_ids.append(product_id)
                    except Exception as e:
                        logging.error(f"Error processing {key}: {e}")
            
            if not features:
                raise ValueError("No valid images found in bucket")
                
            # Создаем и кэшируем индекс
            features_array = np.vstack(features)
            index = faiss.IndexFlatL2(features_array.shape[1])
            index.add(features_array)
            
            self.index_cache["global_index"] = {
                'index': index,
                'product_ids': product_ids
            }
            
        except Exception as e:
            logging.error(f"Index build failed: {e}")
            raise

    def _download_image(self, key: str) -> bytes:
        try:
            print(f"[DEBUG] Trying to access: {self.s3_config.endpoint_url}/{self.s3_config.bucket_name}/{key}")
            
            # Проверка существования объекта
            self.s3.head_object(
                Bucket=self.s3_config.bucket_name,
                Key=key
            )
            
            # Получение объекта
            response = self.s3.get_object(
                Bucket=self.s3_config.bucket_name,
                Key=key
            )
            return response['Body'].read()
            
        except Exception as e:
            logging.error(f"Critical error: {str(e)}")
            raise

    def _extract_features_from_bytes(self, image_bytes: bytes) -> np.ndarray:
        try:
            with Image.open(BytesIO(image_bytes)) as img:
                # Добавляем проверку формата
                if img.format not in ('JPEG', 'PNG'):
                    raise ValueError(f"Unsupported image format: {img.format}")
                    
                return self._extract_features(img)
                
        except Exception as e:
            logging.error(f"Image processing error: {str(e)}")
            raise

    def _extract_features(self, image: Image.Image) -> np.ndarray:
        if image.mode != 'RGB':
            image = image.convert('RGB')

        inputs = self.feature_extractor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :].numpy()