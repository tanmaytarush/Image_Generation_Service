# Drishti
## Web UI and deployable FAST-API

## Features available in the app:

- text to image
- image to image
- instruct pix2piX
- interrogator captioning

## Features available in the api:

- text to image
- image to image
- instruct pix2piX
- interrogator captioning


## Installation

To install bleeding edge version of diffuzers, clone the repo and install it using pip.

```bash
git clone https://github.com/eclipse-tech/Img_Gen_Service
cd Img_Gen_Service
pip install -r requirements.txt
```

Installation using pip:
    
```bash 
pip install diffuzers
```

## Usage

### Web App
To run the web app, run the following command:

```bash
streamlit run Home.py -- --device cuda, mps
```
For Lightning CUDA GPU based systems

```bash
streamlit run diffuzers/Home.py --server.address 0.0.0.0 --server.port 8800 -- --device cuda:0
```

### API

To run the api, run the following command:


```bash
diffuzers api
```

Starting the API requires the following environment variables:

```
export X2IMG_MODEL=stabilityai/stable-diffusion-2-1
export DEVICE=cuda
```

If you want to use inpainting:

```
export INPAINTING_MODEL=stabilityai/stable-diffusion-2-inpainting
```

To use long prompt weighting, use:

```
export PIPELINE=lpw_stable_diffusion
```

### FOR CLI OPTIONS REFER THE DOCUMENT FILE
