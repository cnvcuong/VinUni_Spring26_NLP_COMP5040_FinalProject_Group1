---

# 🧠 Reinforcement Learning-Enhanced Medical Reasoning and Visual Grounding

## 📖 Overview

This project explores advanced **Vision-Language Models (VLMs)** for medical applications, focusing on **medical report generation** and **Visual Question Answering (VQA)** with an emphasis on **clinical reasoning and visual grounding**.

Unlike traditional approaches based solely on Supervised Fine-Tuning (SFT), this project integrates **Reinforcement Learning (RL)** to improve reasoning capabilities and generalization across diverse medical scenarios.

The system is designed not only to generate accurate diagnoses but also to provide **interpretable outputs** by linking textual findings with **localized visual evidence (bounding boxes)**.

---

## 🎯 Objectives

* Develop a multimodal system for **medical image understanding**
* Apply **Reinforcement Learning (RL)** to enhance reasoning beyond SFT
* Enable **visual grounding** of medical findings
* Improve **generalization and robustness** in clinical tasks
* Balance **accuracy and explainability** using structured reasoning

---

## ⚙️ Core Approach

### 🔹 Vision-Language Modeling

* Joint processing of **medical images + text**
* Base model: Multimodal transformer (e.g., Qwen2-VL or similar)

### 🔹 Reinforcement Learning

* Framework: **Group Relative Policy Optimization (GRPO)**
* Uses **rule-based rewards** instead of expensive human annotations

### 🔹 Think-After Protocol

* Model predicts answer first, then explains reasoning
* Helps reduce hallucination and overly verbose reasoning chains

---

## 🚀 System Capabilities

### 1. Clinical Diagnostics

* Input: Medical images (X-ray, CT, MRI)
* Output: Structured medical report

### 2. Visual Grounding

* Detect and localize abnormalities
* Output bounding boxes: `[x1, y1, x2, y2]`

### 3. Multimodal Reasoning

* Step-by-step reasoning using structured tags:

  * `<think>`: reasoning process
  * `<answer>`: final prediction

---

## 🧪 Example Output

```json
<think>
The image shows a PA chest X-ray. The cardiac silhouette appears enlarged. No pleural effusion observed.
</think>

<answer>
{
  "findings": "mild cardiomegaly",
  "bbox_2d": [x1, y1, x2, y2]
}
```

---

## 💡 Motivation

### Problems with Existing Systems:

* Over-reliance on **Supervised Fine-Tuning (SFT)**
* Prone to **shortcut learning**
* Lack of **reasoning capability**
* Limited **interpretability**

### Why Reinforcement Learning?

* Encourages exploration of reasoning strategies
* Improves performance in **out-of-distribution scenarios**
* Reduces dependence on expensive expert annotations

### Key Insight:

The model can develop an **“aha moment”**—learning to reason about abnormalities before identifying them, leading to more accurate and reliable predictions.

---

## 📊 Expected Contributions

* A **3B parameter model** enhanced with RL that rivals larger systems
* Improved **clinical reasoning and robustness**
* A framework combining:

  * Accuracy
  * Explainability
  * Visual interpretability

---

## 📁 Dataset

We use publicly available medical datasets:

* **MIMIC-CXR**
* **IU X-Ray Dataset**
* **OmniMedVQA** (for evaluation)

### Why These Datasets?

* Provide **image-text pairs**
* Enable **rule-based RL rewards**
* Support multiple clinical tasks:

  * Anatomy identification
  * Disease diagnosis
  * Lesion grading
  * Modality recognition
  * Biological attributes

---

## ⚠️ Challenges

* Complex medical terminology
* Limited annotated reasoning data
* Risk of **black-box behavior**
* Ensuring **clinical reliability and interpretability**

---

## 🧪 Methodology

### 1. Data Processing

* Normalize medical images
* Format prompts using `<think>/<answer>` structure

### 2. Modeling

* Train multimodal VLM
* Apply RL with GRPO

### 3. Evaluation Metrics

* **BLEU / ROUGE** → report quality
* **IoU (Intersection over Union)** → bounding box accuracy
* **Greedy Precision** → detect reward hacking

---

## 🛡️ System Improvements

* Prevents **reward hacking** using **odLength reward**
* Supports **cross-modality generalization**
* Produces **interpretable and verifiable outputs**

---

## 👥 Team Members

| Name             | Student ID | Responsibilities                                         |
| ---------------- | ---------- | -------------------------------------------------------- |
| Le Anh Thu       | V202503040 | System development, demo interface, visualization        |
| Luu Duc Toan     | V202502963 | Feature engineering, embeddings, VLM integration         |
| Tran Trung Duc   | V202401788 | Model training, experimentation, evaluation              |
| Nguyen Van Cuong | V202502961 | Data preprocessing, dataset analysis, report structuring |

---

## 🔄 Project Workflow

* **Data pipeline:** Image normalization + structured prompt formatting
* **Modeling:** RL-based training on a 3B model
* **Evaluation:** Multi-metric validation (IoU, BLEU, ROUGE)
* **System:** Interactive demo with reasoning + visual outputs

---

## 📂 Repository Structure

```bash
.
├── data/
├── src/
├── notebooks/
├── proposal.md
├── README.md
```

---

## 🔗 Repository Link

👉 [https://github.com/cnvcuong/VinUni_Spring26_NLP_COMP5040_FinalProject_Group1](https://github.com/cnvcuong/VinUni_Spring26_NLP_COMP5040_FinalProject_Group1)

---

## 📌 Notes

* This project is part of **COMP5040 – Natural Language Processing**
* The repository is maintained as a **collaborative academic project**
* All datasets are used under appropriate licenses and for research purposes only

---
