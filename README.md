# 🧠 Alzheimer's Detection App

Trained on 2141 MRI images, 77.6% accuracy (VGG16 + PCA + SVM).

## Quick Start (30s)
1. `pip install -r requirements.txt`
2. `streamlit run app_local.py`

Upload MRI → Instant 4-class prediction!

## Classes
- Nondemented (Normal)
- Verymilddemented (Early)
- MildDemented  
- Moderatedemented (Advanced)

## Retrain (Optional)
`python train_local.py` (add your images to parent dir folders).
