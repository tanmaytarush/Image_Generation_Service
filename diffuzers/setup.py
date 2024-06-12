from setuptools import setup, find_packages

# Read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="Image_Generation_Service",
    version="0.1.0",
    author="Tanmay Dikshit",
    author_email="tanmay.dikshit@pidilite.vc",
    description="A service for generating images using machine learning stable-diffusion model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tanmaytarush/Image_Generation_Service",
    packages=find_packages(),
    install_requires=[
        "accelerate",
        "altair",
        "basicsr",
        "diffusers",
        "facexlib",
        "fairscale",
        "fastapi",
        "gfpgan",
        "huggingface_hub",
        "loguru",
        "opencv-python",
        "protobuf",
        "pyngrok",
        "python-multipart",
        "realesrgan",
        "streamlit",
        "streamlit-drawable-canvas",
        "st-clickable-images",
        "timm",
        "transformers",
        "uvicorn",
        "starlette",
        "diffuzers",
        "Pillow",  
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
