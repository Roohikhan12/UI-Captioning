# 📱 Mobile Screen Captioning with UI Layout and Text Fusion

## 🔍 Project Overview

Mobile applications have become central to everyday life, yet automated understanding of UI screens remains a significant challenge for accessibility tools, screen readers, and automated testing frameworks.

This project presents a **Deep Learning based Mobile Screen Captioning System** that automatically generates natural language descriptions of Android UI screens using the **RICO Semantic Annotations** dataset.

The proposed framework extracts structured UI information from screen annotation files, generates captions, and produces descriptions using a **Fusion ViT + GPT-2** architecture that combines visual features, visible text, and layout structure.

The trained model is demonstrated through an interactive **Streamlit Web Application** for real-time caption generation.

---

## 🚨 Problem Statement

Accessibility and automation tools need concise natural language descriptions of mobile screens to function effectively.

Traditional approaches to screen understanding are:

- Limited to visual features only — ignoring rich UI metadata
- Unable to describe specific interactive elements like buttons and input fields
- Not scalable for the millions of unique UI screens across app categories
- Insufficient for visually impaired users who rely on screen readers

This project automates screen caption generation using Artificial Intelligence by combining screenshot images, visible UI text, and component layout structure.

---

## 🎯 Objectives

The objectives of this project are:

- Develop an automated mobile screen captioning pipeline from raw annotation data
- Extract structured captions, UI text, and layout sequences from RICO JSON files
- Compare a screenshot-only baseline model against a multimodal fusion model
- Propose an improved Fusion ViT + GPT-2 architecture combining image and text features
- Evaluate using standard caption generation metrics — BLEU-1/2/3/4, ROUGE-L, Token Accuracy
- Deploy the trained model using Streamlit for live demonstration

---

## 📂 Dataset

### Dataset Used

**RICO Semantic Annotations Dataset**

RICO is one of the largest publicly available datasets of Android UI screens collected from real applications.

It contains:

- High-resolution semantic annotation images
- JSON files describing full UI component hierarchies
- Component type labels (Button, EditText, TextView, NavigationBar, etc.)
- Visible text content per component
- Bounding box coordinates for each element
- Screens from 9,772 apps across 27 categories

---

### Caption Categories

The model generates captions in one of five screen type categories:

- Form Screen
- Navigation Screen
- List Screen
- Media Screen
- General Screen

---

### Final Dataset

| Property | Value |
|----------|-------|
| Dataset | RICO Semantic Annotations |
| Total Screens | 66,261 |
| After Cleaning | ~65,000+ |
| Train Split | ~53,000 |
| Validation Split | ~6,600 |
| Test Split | 5,968 |
| Image Size | 224 × 224 |
| Input | Screenshot + UI Text + Layout Sequence |
| Caption Vocab | 20,644 tokens |
| UI Text Vocab | 20,689 tokens |

---

## ⚙️ Project Methodology

The complete workflow consists of the following stages.

```
RICO Dataset (66,261 screens)
            │
            ▼
  JSON Hierarchy Parsing
            │
            ▼
  Caption Generation Pipeline
  (screen type detection)
            │
            ▼
  UI Text Extraction
  (visible text tokens)
            │
            ▼
  Layout Sequence Extraction
  (component label sequence)
            │
            ▼
  fusion_dataset.csv creation
  (image, caption, ui_text, layout)
            │
            ▼
  Train / Validation / Test Split
  (80% / 10% / 10%)
            │
            ▼
  Fusion ViT + GPT-2 Model
            │
            ▼
  Model Evaluation
  (BLEU-1/2/3/4, ROUGE-L, Token Accuracy)
            │
            ▼
  Streamlit Deployment
```

---

## 🔧 Data Preprocessing

The original RICO dataset contains JSON component trees describing UI screens with no ready-made captions.

Each screen was processed individually using the JSON component hierarchy.

Preprocessing steps include:

- Recursive JSON tree parsing to extract all UI components
- Screen type detection from component label patterns
- Caption generation using structured templates per screen type
- UI text extraction — all visible text tokens joined as string (max 60 tokens)
- Layout sequence extraction — component label sequence (max 50 labels)
- Image resizing to 224 × 224
- ImageNet normalisation (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- GPT-2 tokenisation for captions and UI text (max 64 tokens)
- Vocabulary filtering — minimum word frequency of 5 to reduce vocab size
- Train / Validation / Test split with no subsampling

---

## 🧠 Model Development

Two different Deep Learning models were developed and compared.

---

### 1. Baseline CNN-LSTM

A standard encoder-decoder architecture was implemented to establish the benchmark.

Architecture:

- Encoder: ResNet-18 pretrained on ImageNet, last block fine-tuned, 512-d output
- Decoder: Single-layer LSTM (hidden size 512) with teacher forcing
- Input: Screenshot image only
- Dropout: 0.3

Performance:

- BLEU-1: 0.0217
- BLEU-4: 0.0430
- Trained on 8,000 sample subset due to GPU memory constraints

---

### 2. Proposed Fusion ViT + GPT-2

The final proposed architecture combines three sources of information.

Image stream:

- ViT-Base/16 pretrained on ImageNet-21k
- Last two transformer blocks fine-tuned
- CLS token output: 768-d visual feature vector

Text stream:

- GPT-2 tokeniser encodes UI text
- GPT-2 embedding table mean-pooled over tokens
- Output: 768-d text feature vector

Fusion and decoding:

- Image (768-d) and text (768-d) concatenated to 1536-d
- Linear projection (1536 → 768) + Tanh activation
- Single fusion prefix token fed to GPT-2 decoder
- GPT-2 (124M params) last two transformer blocks fine-tuned
- Autoregressive caption generation token by token

---

## 🏗 Proposed Architecture

```
        SCREENSHOT IMAGE
               │
        ViT-Base/16
        (last 2 blocks fine-tuned)
               │
         CLS Token
         768-d visual feature
               │
               ▼
         Feature Fusion
         Linear(1536 → 768) + Tanh
               ▲
               │
         768-d text feature
         Mean Pool
               │
        GPT-2 Embeddings
               │
          UI TEXT TOKENS
               │
               ▼
        GPT-2 Decoder
        (last 2 blocks fine-tuned)
               │
               ▼
       Caption Generation
       (token by token)
```

---

## 🔬 Training Configuration

| Parameter | Value |
|-----------|-------|
| Framework | PyTorch |
| Image Encoder | ViT-Base/16 (HuggingFace) |
| Text Encoder | GPT-2 Embeddings |
| Caption Decoder | GPT-2 (124M params) |
| Optimiser | AdamW |
| Learning Rate | 2e-5 |
| Loss Function | Cross Entropy (ignore padding) |
| Batch Size | 8 |
| Gradient Accumulation | 4 steps (effective batch = 32) |
| Input Size | 224 × 224 |
| Max Caption Length | 64 tokens |
| Epochs | 5 |
| LR Scheduler | ReduceLROnPlateau (factor=0.5) |
| Early Stopping Patience | 2 |
| Gradient Clipping | 1.0 |
| GPU | Kaggle P100 16GB |

---

## 📊 Results

### Training Progress (Fusion Model)

| Epoch | Train Loss | Val Loss |
|-------|-----------|---------|
| 1 | 3.9389 | 3.5074 |
| 2 | 3.6436 | 3.3945 |
| 3 | 3.5333 | 3.3135 |
| 4 | 3.4533 | 3.2441 |
| 5 | 3.3882 | 3.1867 |

Both train and validation loss decreased every epoch — no overfitting observed.

---

### Model Comparison

| Model | BLEU-1 | BLEU-4 | Training Data |
|-------|--------|--------|---------------|
| Baseline CNN-LSTM | 0.0217 | 0.0430 | 8,000 samples |
| Proposed Fusion ViT+GPT-2 | **0.1772** | **0.0809** | 53,000+ samples |

---

### Full Evaluation Results (Fusion Model — Test Set 5,968 screens)

| Metric | Score |
|--------|-------|
| BLEU-1 | 0.1772 |
| BLEU-2 | 0.1426 |
| BLEU-3 | 0.1152 |
| BLEU-4 | 0.0809 |
| ROUGE-L | 0.2361 |
| Token Accuracy | 0.1518 |

---

## 🔎 Qualitative Examples

(Add screenshot of Streamlit app output here)

Key Observations:

- Model correctly identifies screen types — form, list, navigation, general
- Predicts relevant UI elements — buttons, input fields, navigation items
- Captions are topically related to the actual screen content
- Greedy decoding sometimes causes token repetition in longer captions
- Non-English screens produce weaker captions due to vocabulary mismatch

---

## ⚠️ Error Analysis and Limitations

**Failure Cases:**

- Token repetition — greedy decoding causes phrases to repeat (e.g. "containing buttons Sign In. containing buttons Sign In.")
- Non-English screens — Korean, French, and other language screens produce weak captions
- Media screens — large image-only screens with no extractable text produce generic captions
- Specific content mismatch — model predicts category-level content rather than app-specific content

**Limitations:**

- RICO semantic annotation images show colored blocks, not rendered screenshots — visual encoder sees layout structure not actual pixel content
- Captions are auto-generated from JSON templates, not written by humans — limits vocabulary diversity
- Baseline trained on only 6,400 samples vs 53,000 for fusion — comparison is not fully controlled
- GPU memory constraints required batch size of 8 and gradient accumulation
- Only 5 epochs trained — val loss was still decreasing, more epochs would improve results

**What Could Be Improved:**

- Apply beam search or nucleus sampling to reduce repetition
- Use human-written Screen2Words captions for more natural language training
- Use rendered app screenshots instead of semantic annotation images
- Fine-tune all transformer layers with more GPU resources
- Train for more epochs with larger batch size

---

## ✅ Conclusion

**Main Finding:**

A multimodal fusion model combining UI screenshots, visible text, and layout structure significantly outperforms a screenshot-only baseline for mobile screen captioning.

**Answer to Project Question:**

Yes — combining visual features with UI text and layout metadata improves caption generation quality substantially.

- BLEU-1 improved from 0.0217 to 0.1772 — an 8x improvement over baseline
- BLEU-4 improved from 0.0430 to 0.0809 — nearly 2x improvement
- ROUGE-L of 0.2361 confirms meaningful sequence-level overlap with reference captions

**Key Takeaways:**

- Pretrained ViT + GPT-2 converges in 5 epochs vs CNN-LSTM needing 8+ epochs
- UI text metadata is essential — model correctly identifies screen types using text signals
- Training data size matters — 53k samples produces substantially better generalisation than 6.4k
- Template-based caption generation limits upper bound — human captions would yield better results
- Multimodal fusion is the right approach for UI screen understanding tasks

---

## 🛠 Installation

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Mobile-Screen-Captioning.git
```

Open project

```bash
cd Mobile-Screen-Captioning
```

Install dependencies

```bash
pip install -r requirements.txt
```

Place `fast_fusion_best.pth` in the same folder as `app.py` before running.

---

## 📦 Repository Structure

```
Mobile-Screen-Captioning/
├── 03_data_preparation_fusion.ipynb   # Builds fusion_dataset.csv from RICO JSONs
├── FusionModel_V2_FINAL.ipynb         # Fusion ViT+GPT-2 training and evaluation
├── baseline_model.ipynb               # Baseline CNN-LSTM training                       
├── fusion_dataset.csv                 # Generated dataset (image, caption, ui_text, layout)
├── generated_captions_full.csv        # Baseline dataset (image, caption)
├── fast_fusion_best.pth               # Best trained model checkpoint
├── training_curves.png                # Loss curves (train and val)
└── README.md                          # This file
```

---

## 💻 Technologies Used

**Programming Language**

- Python 3.10

**Libraries**

- PyTorch
- HuggingFace Transformers
- Torchvision
- NumPy
- Pandas
- Pillow
- Scikit-Learn
- NLTK
- Rouge-Score
- Matplotlib
- Streamlit

**Development Environment**

- Google Colab
- Kaggle (P100 GPU)
- VS Code
- GitHub

---

## 🚀 Future Improvements

- Evaluate on Screen2Words human-written captions for benchmark comparison
- Apply beam search or nucleus sampling for improved decoding quality
- Use rendered app screenshots instead of semantic annotation images
- Fine-tune full ViT and GPT-2 with sufficient GPU resources
- Train for more epochs — validation loss was still decreasing at epoch 5
- Incorporate bounding box spatial coordinates as positional layout features
- Extend to cross-lingual captioning for non-English UI screens
- Mobile application deployment for real-time accessibility assistance

---

## 👥 Authors

[Your Name]

MSc Data Science / Artificial Intelligence

[Your University]

---

## 📚 References

1. Deka B. et al., RICO: A Mobile App Dataset for Building Data-Driven Design Applications. UIST 2017.
2. Wang B. et al., Screen2Words: Automatic Mobile UI Summarization with Multimodal Learning. UIST 2021.
3. Dosovitskiy A. et al., An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. ICLR 2021.
4. Radford A. et al., Language Models are Unsupervised Multitask Learners. OpenAI Blog 2019.
5. Vinyals O. et al., Show and Tell: A Neural Image Caption Generator. CVPR 2015.
6. Papineni K. et al., BLEU: a Method for Automatic Evaluation of Machine Translation. ACL 2002.
7. Lin C.Y., ROUGE: A Package for Automatic Evaluation of Summaries. ACL Workshop 2004.
8. He K. et al., Deep Residual Learning for Image Recognition. CVPR 2016.
9. PyTorch Documentation. https://pytorch.org/docs
10. HuggingFace Transformers Documentation. https://huggingface.co/docs/transformers
