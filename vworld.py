import requests
import io
from PIL import Image

class VWorld:
    def __init__(self, apiKey) -> None:
        self.baseURL = 'https://api.vworld.kr/req'
        self.apiKey = apiKey
        self.cache = dict()
    
    def get_image(self, x, y) -> Image:
        coords_str = f'{x},{y}'

        if coords_str in self.cache:
            response = self.cache[coords_str]
        else:
            response = requests.get(f'{self.baseURL}/image', {
            'request': 'getmap',
            'service': 'image',
            'key': self.apiKey,
            'zoom': 18,
            'size': '1024,1024',
            'crs': 'EPSG:4326',
            'basemap': 'PHOTO',
            'center': coords_str,
        })

        image = Image.open(io.BytesIO(response.content))
        return image