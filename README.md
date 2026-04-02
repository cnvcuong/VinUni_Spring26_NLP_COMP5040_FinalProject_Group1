# 📌 1. Project Description

## 🧠 Vision-Language Models for Medical Report Understanding

This project explores the application of **Vision-Language Models (VLMs)** in the medical domain, specifically for **radiology image understanding and report generation**.

### 📖 Overview

Medical imaging (e.g., chest X-rays) plays a crucial role in clinical diagnosis. However, interpreting these images requires significant expertise and time. This project aims to develop a multimodal system that can:

* Understand medical images
* Generate structured textual descriptions (reports)
* Answer clinical questions related to the image

By leveraging recent advances in **multimodal deep learning**, the system combines **computer vision** and **natural language processing** to assist in medical analysis.

---

### 🎯 Objectives

* Build a system that maps **medical images → textual reports**
* Explore **Vision-Language Models (VLMs)** such as:

  * Encoder-decoder architectures
  * Pretrained multimodal transformers
* Evaluate model performance on:

  * Text generation (e.g., BLEU, ROUGE)
  * Question answering accuracy (optional extension)

---

### ⚙️ System Capabilities

The final system is expected to support:

#### 1. Report Generation

* Input: Medical image (e.g., chest X-ray)
* Output: Generated radiology-style report

#### 2. Visual Question Answering (Optional)

* Input: Image + natural language question
* Output: Answer based on image content

---

### 🚀 Motivation

* Reduce workload for radiologists
* Improve efficiency in medical workflows
* Provide automated assistance in low-resource settings
* Demonstrate the effectiveness of multimodal AI in healthcare

---

### 🧪 Methodology

The project follows a standard NLP + ML pipeline:

1. **Data preprocessing**

   * Image normalization
   * Text cleaning and tokenization

2. **Feature representation**

   * Image embeddings (CNN / Vision Transformer)
   * Text embeddings (BERT / domain-specific models)

3. **Modeling**

   * Vision-Language Models (e.g., BLIP, LLaVA, or similar)
   * Fine-tuning on medical datasets

4. **Evaluation**

   * BLEU / ROUGE for report generation
   * Accuracy for VQA

---

### 📊 Expected Challenges

* Domain-specific medical terminology
* Limited labeled data
* Noisy or inconsistent reports
* Ethical considerations in medical AI

---

### 👥 Team Members

| Name             | Student ID |
| ---------------- | ---------- |
| Le Anh Thu       | V202503040 |
| Luu Duc Toan     | V202502963 |
| Tran Trung Duc   | V202401788 |
| Nguyen Van Cuong | V202502961 |

---
