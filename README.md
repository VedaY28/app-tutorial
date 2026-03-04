# HazardMon: SAGE Container Deployment

## Inside The Container:
- main.py (Yolov11 Inference Script)
  - --file: Specifies the input path for the video or image file
  - --device: Selects hardware (use cpu or 0 for GPU)
  - --batch_size: Determines the number of frames processed per inference pass
    
- requirements.txt
  - ultralytics
  - numpy
  - pywaggle[all]: The SAGE edge messaging and sensor interface
  - opencv-python-headless: Optimized image processing for headless server environments
    
- Dockerfile
  - Image Base: NVIDIA PyTorch 24.06 (nvcr.io/nvidia/pytorch:24.06-py3)
    - (Works on x86_64 Architecture)
  - System Dependencies: Installs libgl1, ffmpeg, and mosquitto-clients
  
- Yolov11 Fire Model (firedetect-11s.pt)
  
- Sage.yaml

## SAGE Container Setup
(Note: Ensure the container deployment scripts are maintained in a separate repository)

### 1. Preparation
```
git clone "REPO_URL"
cd "REPO"
```
Download the video file to the project root
#### Boreal-Dataset (High-Def Videos)
##### Use browser to access download links from this url: https://etsin.fairdata.fi/dataset/1dce1023-493a-4d63-a906-f2a44f831898/data

###### FOR NOW: Download Boreal-Forest-Fire-Subset-B/Evo-videos/evo-16
###### (Note: Only allows for 1 video for now)

### 2. Build and Run The Container
```
sudo pluginctl build .
```
##### After it successfully builts the container, grab the last line of output: image registry address 
##### (Should look something like: "10.31.81.1:5000/local/app-tutorial")

#### Format
```
sudo pluginctl run --name <repo_name> <image_registry_address> -- --file <path_to_video> --device <cpu/0> --batch_size <int>
```
GPU
```
sudo pluginctl run --name app-tutorial 10.31.81.1:5000/local/app-tutorial -- --file /app/evo_16.mp4 --device 0 --batch_size 10
```
CPU
```
sudo pluginctl run --name app-tutorial 10.31.81.1:5000/local/app-tutorial -- --file /app/evo_16.mp4 --device cpu
```
### Viewing Output
In the Sage Node, query the Beehive Data Repository:
##### Make sure to change the task name
```
curl -s 'Content-Type: application/json' https://data.sagecontinuum.org/api/v1/query -d '
{
    "start": "-5m",
    "filter": {
        "task": "<repo_name>"
    }
}'
```

