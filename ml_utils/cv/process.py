import base64
from io import BytesIO
from PIL import Image
import numpy as np
import os
import logging
logger = logging.getLogger(__name__)


IMAGE_PROCESS_API_EXAMPLE = """
from ml_utils.cv import b64string2numpy, imsave
from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()

class ImageRequest(BaseModel):
    data:str

@app.post("/image")
def handle_image_to_file(imreq:ImageRequest):
    img = b64string2numpy(imreq.data)
    imsave(img, "testsaveimage.png")
    return {"Status":"success"}
"""


def generate_image_router(path = ".", filename = "image_api.py"):
    with open(os.path.join(path, filename), 'w') as f:
        f.write(IMAGE_PROCESS_API_EXAMPLE)
    


def file2b64(path:str, encode:bool = True):
    """Convert a image file to base64 string, encoded or not

    Args:
        path (_type_): string
        encode (bool, optional): _description_. Defaults to True.

    Raises:
        e: _description_

    Returns:
        _type_: _description_
    """
    with open(path, "rb") as img_file:
        try:
            logger.info("Converting image file to base64 string...")
            
            buffer = base64.b64encode(img_file.read())
            return buffer if encode else buffer.decode()
        except Exception as e:
            logger.error("Failed to convert image to base64 format")
            raise e 
        
def b64string2numpy(b64string):
    """Convert a base64 encoded string to numpy array
    Args:
        b64string (_type_): string of encoded bytes
        format (str, optional): string of byte

    Returns:
        _type_: np.array
    """
    
    if isinstance(b64string, str):
        b64string = b64string.encode()
    
    buff = BytesIO(base64.b64decode(b64string))
    image = Image.open(buff)
    return np.array(image)


def imread(path:str):
    """Read image file into numpy array"""
    return np.array(Image.open(path))
    

def imsave(np_image, path):
    """Save numpy array to image file"""
    logger.info(f"Saving image to path {path}")
    Image.fromarray(np_image).save(path)