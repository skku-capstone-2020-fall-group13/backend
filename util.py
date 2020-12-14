import base64
import io
import PIL.Image as Image

def convert_base64(image: Image) -> str:
    image_buffer = io.BytesIO()
    image.save(image_buffer, format="png")
    return f'data:image/png;base64,{base64.b64encode(image_buffer.getvalue())}'