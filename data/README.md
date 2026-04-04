---

# 📁 Dataset Description

## Overview

This directory contains datasets used for training and evaluating a **Vision-Language Model (VLM)** enhanced with **Reinforcement Learning (RL)** for medical reasoning and visual grounding. 

The dataset supports multiple multimodal tasks, including:

* Medical report generation
* Visual Question Answering (VQA)
* Visual grounding (abnormality localization with bounding boxes)

---

## 📦 Data Sources

We utilize publicly available medical datasets:

### 1. MIMIC-CXR

* Large-scale chest X-ray dataset
* Paired with radiology reports
* Widely used in clinical AI research

### 2. IU X-Ray Dataset

* Smaller but well-structured dataset
* Includes annotated reports
* Suitable for prototyping and validation

### 3. OmniMedVQA (for evaluation)

* Multi-task benchmark dataset
* Covers diverse clinical reasoning tasks

---

## 📊 Data Structure

Each sample consists of:

### 🔹 Input

* Medical image (X-ray / CT / MRI)
* Optional clinical question

### 🔹 Output

* Structured textual report
* Visual grounding annotations (if available)

---

### 📄 Example Format

```json id="u1j5gs"
{
  "image": "images/img_001.png",
  "question": "Is there cardiomegaly?",
  "report": "The cardiac silhouette is enlarged, consistent with mild cardiomegaly.",
  "bbox_2d": [x1, y1, x2, y2]
}
```

---

## 🧱 Directory Structure

```bash id="0d9ylp"
/data
├── raw/              # Original datasets (not uploaded if large)
├── processed/        # Cleaned and formatted data
├── sample/           # Small subset for testing/demo
├── annotations/      # Bounding boxes / labels (if available)
├── README.md
```

---

## 🎯 Why This Dataset?

* Provides **image-text alignment** for multimodal learning
* Enables **rule-based reward design** for RL training
* Supports diverse clinical tasks:

  * Anatomy identification
  * Disease diagnosis
  * Lesion grading
  * Modality recognition
  * Biological attribute prediction

---

## ⚠️ Challenges

* Specialized medical terminology
* Limited bounding box annotations
* Data imbalance across disease classes
* Variation in report style and quality

---

## 🛠️ Preprocessing Pipeline

### 1. Image Processing

* Resize and normalize images
* Convert to standard format
* Optional augmentation

### 2. Text Processing

* Clean and normalize reports
* Tokenization
* Format into structured prompts:

  * `<think>` (reasoning)
  * `<answer>` (final output)

### 3. Data Formatting

* Convert to multimodal input format
* Align images with reports and annotations
* Split into train / validation / test sets

---

## 🧪 Role in Reinforcement Learning

The dataset is used to define **reward functions** for RL training:

* **Report Quality Reward** → BLEU / ROUGE
* **Localization Reward** → IoU (bounding boxes)
* **Anti-Reward Hacking** → Greedy Precision, odLength penalty

These rewards guide the model toward **accurate and interpretable reasoning**.

---

## 🔒 Ethical Considerations

* All datasets are **anonymized**
* Used strictly for **research and educational purposes**
* Must comply with dataset licenses

---

## 📌 Notes

* Large datasets (e.g., MIMIC-CXR) are **not stored directly in this repository**
* Only sample data or metadata may be included
* Full datasets should be downloaded via external links or scripts

---
