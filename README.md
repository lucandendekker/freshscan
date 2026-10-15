# FreshScan

**A computer vision prototype that classifies potato images as fresh or rotten.**

Built by Lucan den Dekker for the first-year Deep Learning block of Applied Data Science & AI at Breda University of Applied Sciences. The project explores an accessible alternative to manual quality sorting for small potato growers.

## Try it yourself

**[Try FreshScan in my portfolio](https://lucandendekker.nl/portfolio/freshscan-image-classifier/)** — upload a potato image in the embedded dashboard and see the model's prediction.

You can also [open the demo directly](https://lucan5533-freshscan.hf.space/).

## What is included

- The original interactive dashboard with image upload, classification scores and sample images.
- A Flask API using the saved MobileNetV2-based classifier.
- The trained model: inference does not require downloading ImageNet weights or training again.
- Seven research notebooks: data preparation, baselines, four CNN iterations and error analysis.

The dashboard also contains a **simulated sorting line**. Its animated throughput and sorting counters are not measurements from a real conveyor belt. Uploaded-image predictions use the actual trained model.

## Run locally

Use **Python 3.12** (64-bit). This TensorFlow version does not support Python 3.14.

```sh
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```sh
# macOS / Linux
source .venv/bin/activate
```

Install and start:

```sh
python -m pip install -r requirements.txt
python app.py
```

Open **http://localhost:5000**. On Windows, `start.bat` also starts the app once the environment is set up. Processing happens on the local server; uploads are not saved. Dashboard fonts and charts load from external CDNs.

The verified setup is Windows x86-64. Other platforms need compatible TensorFlow wheels. The Flask development server is for local use, not production hosting.

## Approach

1. Combine potato images from three public datasets, standardise formats, remove duplicates and create train/validation/test splits.
2. Establish an MLP baseline, then compare a basic CNN, augmentation and transfer learning.
3. Use a frozen MobileNetV2 backbone with a classification head: global average pooling, Dense(256), batch normalisation, dropout and a two-class softmax.
4. Evaluate the final model and analyse its errors.

The final training notebook uses **224 x 224 RGB images**, scaled to **[0, 1]**, with alphabetic class order: `fresh`, `rotten`. The app preserves this preprocessing, including the training generator's nearest-neighbour resizing. It loads the complete saved model through `tf-keras`, rather than reconstructing only part of its weights.

## Results and limits

The school project reports **97.32% test accuracy** on its held-out split from a curated collection of **3,712 images**. This is a recorded project result, not an independently reproduced full-dataset evaluation in this repository. The full dataset is not included.

These scores describe the project dataset, not food safety or real conveyor-belt performance. Confidence scores are model outputs, not guarantees. Conveyor integration and automatic physical sorting are future work.

## Repository structure

```text
app.py                   Flask API and full-model loading
index.html               Original dashboard (same-origin /predict API)
cnn_v4_best.keras        Trained classification model
potato_imgs/             16 original demonstration images
notebooks/               Clean copies of the research notebooks
tests/test_app.py        API, preprocessing and real-model smoke tests
requirements.txt         Pinned runtime dependencies
start.bat                Windows launcher
```

## Research notebooks and data

Notebooks are research artefacts with execution outputs cleared. They require Jupyter, matplotlib, seaborn, scikit-learn and the original data layout; they are not a one-click reproduction pipeline. Start with `01_data_preparation.ipynb`, which documents the source directories and 70/20/10 split. Later notebooks expect `data/processed/{train,val,test}/{fresh,rotten}` and may need working-directory adjustments. The app works without that training data.

Dataset sources recorded in the preparation notebook:

- [Fresh and stale classification - swoyam2609](https://www.kaggle.com/datasets/swoyam2609/fresh-and-stale-classification)
- [Fruits and vegetables dataset - muhriddinmuxiddinov](https://www.kaggle.com/datasets/muhriddinmuxiddinov/fruits-and-vegetables-dataset/data)
- [Fresh and rotten fruits and vegetables - filipemonteir](https://www.kaggle.com/datasets/filipemonteir/fresh-and-rotten-fruits-and-vegetables)

Dataset images and pretrained components retain their original authorship and terms. No blanket licence is assigned to third-party materials by this repository.

## Tests

```sh
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Tests cover preprocessing, upload validation, API responses, static-file boundaries and actual predictions on two sample images.
