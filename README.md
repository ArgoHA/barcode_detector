## Idea
This repo is created to train a YOLOv8 model to detect barcodes. Data was gathered from [this repo](https://github.com/abbyy/barcode_detection_benchmark).

### Main tasks:
- Preprocess data (fix images format, convert labels to YOLO format, multilabel split)
- Train the model (use pretrained weights)

### To run this repo:
- Download the dataset - [ZVZ-real.zip](https://drive.google.com/drive/folders/1a_SSHyfQMuq2Q77OHp87igx0u6f3Ga2t)
- Install requirements.txt:
```
pip install -r requirements.txt
```
- In config.yaml change `data_root_path` to your folder (change other configes if needed)
- Run main.py:
```
python main.py
```

### Results:
Baseline was trained for 50 epoches with **mAP: 0.84**

[Google colab notebook](https://colab.research.google.com/drive/1Yi5qOqUP38-My3ONcEE_P-xNaMH0w8w9?usp=sharing) with training process

![val_pred](/images/val_batch0_pred.jpg)
![start_training](/images/start.png)
![start_training](/images/end.png)
