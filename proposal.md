# NLP Project Proposal

**Course:** COMP4020 / COMP5040 – Natural Language Processing

**Project Title:** *Vision-Language Models for Medical Report Understanding and Assistance*

---

## 1. What is the project about?

This project focuses on applying **Vision-Language Models (VLMs)** to a medical NLP task.

* **NLP Task:** Medical report generation and visual question answering (VQA)
* **Approach:** Combine image understanding and natural language processing using VLMs
* **Domain:** Healthcare (radiology / clinical diagnostics)

The system will take **medical images (e.g., X-rays)** and optionally text queries, and generate:

* Descriptive medical reports, or
* Answers to clinical questions about the image

This project aligns with real-world applications such as assisting radiologists in diagnosis and reducing workload.

---

## 2. Why this project?

Medical imaging generates large volumes of data that require expert interpretation.

### Problem:

* Manual analysis of medical images is **time-consuming and expensive**
* There is a shortage of trained radiologists in many regions
* Existing systems often treat vision and language separately

### Motivation:

* VLMs enable **joint reasoning over images and text**
* Automating report generation can improve **efficiency and consistency**
* Clinical question answering can support **decision-making**

### Expected Contributions:

* A system that can interpret medical images and produce structured outputs
* Insights into how multimodal models perform in medical contexts
* Evaluation of challenges such as domain-specific terminology and data limitations

---

## 3. What is the final product?

The final product will be a **multimodal NLP system** with the following capabilities:

### Core Features:

* Input: Medical image (e.g., chest X-ray)
* Output:

  * Generated medical report (text generation), OR
  * Answers to user queries (VQA)

### Example:

* Input: Chest X-ray
* Output:

  > “Findings suggest mild cardiomegaly with no acute infiltrates.”

### Improvements over existing processes:

* Reduces manual workload
* Speeds up report generation
* Provides consistent descriptions
* Enables interactive querying of medical images

Optional:

* Simple demo interface (e.g., web UI or notebook interface)

---

## 4. Why this data?

### Dataset:

We plan to use a publicly available medical dataset such as:

* **MIMIC-CXR** or
* **IU X-Ray dataset**

### Description:

* Contains:

  * Medical images (X-rays)
  * Associated radiology reports (text)
* Structure:

  * Image + report pairs
  * May include sections (Findings, Impression)

### Domain:

* Clinical radiology

### Suitability:

* Ideal for **image-to-text generation** and **multimodal learning**
* Widely used benchmark dataset in medical AI research

### Challenges:

* Medical terminology is highly specialized
* Data may be noisy or inconsistent
* Possible class imbalance (rare conditions)
* Ethical considerations and data privacy

---

## 5. Team responsibilities

| Member           | Student ID | Responsibilities                                         |
| ---------------- | ---------- | -------------------------------------------------------- |
| Le Anh Thu       | V202503040 | System development, demo interface, visualization |
| Luu Duc Toan     | V202502963 | Feature engineering, embeddings, VLM integration         |
| Tran Trung Duc   | V202401788 | Model training, experimentation, evaluation              |
| Nguyen Van Cuong | V202502961 | Data preprocessing, dataset analysis, report structuring        |

### Workflow Distribution:

* **Data pipeline:** preprocessing, cleaning, formatting
* **Modeling:** baseline + VLM experiments
* **Evaluation:** metrics (BLEU, ROUGE, accuracy for VQA)
* **System:** demo + reporting

Roles are distributed to ensure balanced contribution across the pipeline.

---

## 📂 Repository Structure (for GitHub)

```
/project-root
│── proposal.md
│── /data
│   ├── dataset files
│   └── README.md
│── /src
│── /notebooks
```

---

## 📌 Notes

* The repository will be kept **private**
* Collaborators will include:

  * Dr Mo El-Haj (drelhaj)
  * Mr Nguyen Huy Hung (whistle-hikhi)

---
