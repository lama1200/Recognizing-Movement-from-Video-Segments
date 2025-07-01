
```markdown
# Recognizing Movement from Video Segments

## Project Overview
This project teaches a computer to watch short video clips and recognize simple movements. It splits videos into frames, uses a YOLOv8 model to detect people, and then labels each person’s action—like running, waving, or clapping. In the end, you get a system that reads video and tells you what actions happen, frame by frame.

## Recognized Actions
This model is trained to identify 3 core human activities:

- Boxing
- Hand waving
- Hand clapping

## Dataset
- **YOLO Dataset.zip**: Contains video frames and YOLO-format annotation files (`.txt` bounding boxes with activity labels).
  - `dataset/images/train/` and `dataset/images/val/` for frames
  - `dataset/labels/train/` and `dataset/labels/val/` for labels
- The dataset is split into 80% training and 20% validation sets.

````

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
