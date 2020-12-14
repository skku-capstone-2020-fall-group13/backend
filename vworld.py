import requests
from collections import defaultdict

class VWorld:
    def __init__(self, apiKey) -> None:
        self.baseURL = 'https://api.vworld.kr/req'
        self.apiKey = apiKey
        self.cache = defaultdict(str)
    
    def get_image(self, coords):
        coords_str = f'{coords.x},{coords.y}'

        response = self.cache[coords_str] \
            or requests.get(f'{self.baseURL}/image', {
                'request': 'getmap',
                'service': 'image',
                'key': self.apiKey,
                'zoom': 18,
                'size': '1024,1024',
                'crs': 'EPSG:4326',
                'basemap': 'PHOTO',
                'center': coords_str,
            })

        return response.content