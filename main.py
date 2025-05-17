from fastapi import FastAPI, UploadFile, HTTPException, Form
from typing import List
from app.search import ImageSearchEngine
from app.models import SearchResult, S3Config
import logging
import json

app = FastAPI(
    title="Image Search API",
    description="API для поиска товаров по изображению",
    version="0.1.0"
)
 
@app.post("/search/{shop_id}", response_model=List[SearchResult])
async def search_in_shop(
    shop_id: str,
    file: UploadFile,
    s3_config: str = Form(...),
    top_k: int = 5
):
    s3_config_obj = S3Config(**json.loads(s3_config))
    print(s3_config)
    """
    Поиск похожих товаров в указанном магазине
    
    Параметры:
    - shop_id: ID магазина
    - file: Изображение для поиска
    - top_k: Количество возвращаемых результатов
    """
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Файл должен быть изображением")
        
        contents = await file.read()
        
        # Инициализируем движок с конфигом S3 для каждого запроса
        search_engine = ImageSearchEngine(s3_config_obj)
        
        results = search_engine.search_similar(
            image_bytes=contents,
            top_k=top_k
        )
        
        return results
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logging.error(f"Ошибка при поиске в магазине {shop_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Ошибка при обработке запроса")

@app.get("/health")
async def health_check():
    return {"status": "OK"}