# Quick Setup Guide

## 🚀 Getting Started in 5 Minutes

### Step 1: Download the Dataset

1. Go to [Kaggle Dogs vs Cats](https://www.kaggle.com/c/dogs-vs-cats/data)
2. Sign in (create free account if needed)
3. Download `train.zip` (853 MB)
4. Extract to `data/train/` directory in this project

Your directory structure should look like:
```
dogs-vs-cats-classification/
└── data/
    └── train/
        ├── cat.0.jpg
        ├── cat.1.jpg
        ├── dog.0.jpg
        ├── dog.1.jpg
        └── ... (25,000 total images)
```

### Step 2: Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### Step 3: Launch Jupyter Notebook

```bash
jupyter notebook dogs_vs_cats_classification.ipynb
```

### Step 4: Run the Notebook

- Execute cells from top to bottom
- The first run will automatically organize the data
- Training takes ~1 hour on GPU (3-4 hours on CPU)

## ⚠️ Important Notes

### GPU Acceleration (Highly Recommended)

**Google Colab (Free GPU):**
1. Upload notebook to Google Drive
2. Open with Google Colab
3. Runtime → Change runtime type → GPU
4. Upload dataset or mount Google Drive

**Local GPU Setup:**
If you have NVIDIA GPU:
```bash
# Install CUDA-enabled TensorFlow
pip install tensorflow[and-cuda]
```

### Memory Requirements

- **RAM:** Minimum 8GB, recommended 16GB
- **Disk Space:** ~2GB for dataset + 500MB for models
- **GPU Memory:** 4GB+ recommended for faster training

### Common Issues

**Issue 1: Out of Memory**
```python
# Reduce batch size in notebook
BATCH_SIZE = 16  # or 8
```

**Issue 2: Dataset Not Found**
```python
# Verify path
print(TRAIN_DIR)
# Should show: /path/to/project/data/train
```

**Issue 3: ImportError**
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

## 📊 Expected Runtime

| Task | CPU | GPU |
|------|-----|-----|
| Data organization | 2-3 min | 2-3 min |
| EDA | 5-10 min | 5-10 min |
| Custom CNN training | 2-3 hours | 30-45 min |
| VGG16 training | 1-2 hours | 20-30 min |
| Evaluation | 5-10 min | 5-10 min |
| **Total** | **3-5 hours** | **1-2 hours** |

## 🎯 Assignment Submission Checklist

- [ ] Jupyter notebook with all outputs
- [ ] PDF cover page with names, IDs, and GitHub link
- [ ] GitHub repository with:
  - [ ] README.md
  - [ ] .gitignore
  - [ ] requirements.txt
  - [ ] Notebook (.ipynb file)
  - [ ] PDF cover page
- [ ] All markdown cells filled with insights
- [ ] All code cells executed with outputs visible
- [ ] At least 5 markdown talking points completed

## 🆘 Need Help?

1. Check the [main README](README.md) for detailed documentation
2. Review the notebook markdown cells for guidance
3. Contact course instructor or TA
4. Check TensorFlow/Keras documentation

## 📝 Customization Tips

### Update Student Information
1. Open `create_cover_pdf.py`
2. Update student names and IDs
3. Run: `python create_cover_pdf.py`

### Update GitHub Link
1. Create your repository on GitHub
2. Update the link in README.md and PDF

### Modify Models
- Experiment with different architectures
- Try other pre-trained models (ResNet, EfficientNet)
- Adjust hyperparameters (learning rate, dropout, etc.)

## 🎓 Learning Resources

- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- [Keras Documentation](https://keras.io/)
- [Deep Learning Specialization (Coursera)](https://www.coursera.org/specializations/deep-learning)
- [CS231n: CNN for Visual Recognition](http://cs231n.stanford.edu/)

---

Good luck with your assignment! 🚀
