
#  Recognizing Movement from Video Segments

##  Overview
This project aims to recognize human actions from short video segments using computer vision techniques. By splitting videos into individual frames and applying the YOLOv8 object detection model, we identify and classify specific human activities. The system ultimately produces labeled actions per frame.

---

##  Recognized Actions
The trained YOLOv8 model can detect the following movements:
- 🥊 Boxing  
- 👋 Hand Waving  
- 👏 Hand Clapping

---

## 📁 Dataset Structure

The dataset follows the YOLO format and includes:

YOLO Dataset.zip
├── dataset/
│ ├── images/
│ │ ├── train/
│ │ └── val/
│ └── labels/
│ ├── train/
│ └── val/

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/<username>/Recognizing-Movement-from-Video-Segments.git
   cd Recognizing-Movement-from-Video-Segments
````

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate    # Windows
   ```
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Training

Open **YOLOv8\_All\_Models.ipynb** and set:

* Paths to your segmented frame dataset
* Model variant (`yolov8n`, `yolov8s`, `yolov8m`, etc.)
* Hyperparameters (batch size, epochs, learning rate)

Run the notebook to:

1. Load and preprocess frames
2. Train and validate the YOLOv8 model
3. Track metrics (loss, mAP, precision, recall)

Model checkpoints and logs are saved under `results/<model_name>/`.

## Evaluation & Inference

* **Evaluate**: Calculate mAP\@0.5 and mAP\@0.5:0.95 on validation frames.
* **Inference**: Example code:

  ```python
  from ultralytics import YOLO
  model = YOLO('results/yolov8n/best.pt')
  results = model.predict(source='dataset/images/val/', conf=0.25)
  results.save(save_dir='results/inference/')
  ```

## Results

* **Training Logs**: TensorBoard logs in `results/<model>/`.
* **Model Weights**: `best.pt`, `last.pt` files.
* **Inference Outputs**: Annotated frames saved in `results/inference/`.

## Dependencies

* Python 3.8+
* `ultralytics`, `torch`, `torchvision`, `matplotlib`, `numpy`, `pandas`

Install with:

```bash
pip install -r requirements.txt
```

## Author

Rimas Albahli - Lama Alotibie - Atheer alasiri Maymona Alotaibi - Elaf Alshehri - Afnan Alsuliman  Fatima bahwairith 

## License

Licensed under the MIT License. See [LICENSE](LICENSE) for details.

```
```

```
```
