# NLP Project Proposal

**Course:** COMP5040 – Natural Language Processing

**Project Title:** *Reinforcement Learning-Enhanced Medical Reasoning and Visual Grounding*

---

## 1. What is the project about?

This project focuses on applying **Vision-Language Models (VLMs)** to critical medical NLP tasks, specifically **medical report generation** and **Visual Question Answering (VQA)**. Unlike traditional models that rely solely on Supervised Fine-Tuning (SFT), this system implements **Reinforcement Learning (RL)** to enhance clinical reasoning and generalization.

* **Approach:** We will utilize the **Group Relative Policy Optimization (GRPO)** framework to train a model that performs joint reasoning over images and text without requiring expensive expert-annotated rationales.
* **Core Tasks:**
  - **Clinical Diagnostics**: Generating descriptive reports for modalities like X-rays, CT, and MRI.
  - **Visual Grounding**: Identifying and **highlighting specific medical abnormalities** via bounding box coordinates (e.g., [x1, y1, x2, y2]) within the model’s response.

This project aims to reduce radiologist workload by providing a system that doesn't just memorize shortcuts but develops a **generalizable reasoning pattern** applicable across different clinical scenarios.

---

## 2. Why this project?

**Problem**: Current medical VLMs primarily rely on SFT, which often leads to "shortcut learning" - where the model memorizes superficial patterns in training data rather than learning true medical logic. Furthermore, there is a severe **scarcity of high-quality Chain-of-Thought (CoT) annotations** for medical data, as expert rationales are expensive to curate.

### Motivation:

* **Beyond SFT**: RL allows the model to explore diverse reasoning strategies through rule-based rewards, making it more robust in out-of-domain tasks.
* **The "Aha Moment"**: By training on visual grounding tasks, the model can develop an "aha moment" where it spontaneously reasons about the presence of an object before identifying it, reducing false positives in diagnosis.
* **Clinical Reliability**: Specialized medical tasks like lesion grading or anatomy identification require **multi-step analysis** (e.g., morphology and context), which RL-driven models handle more efficiently than general-purpose models.


### Expected Contributions:

* A system that outperforms significantly larger models by using efficient RL-driven adaptation on a smaller 3B base.
* A framework that balances **accuracy and explainability** using the **"Think-After" protocol**, where the model predicts the answer first and rationalizes later to avoid lengthy, hallucinated reasoning chains.

---

## 3. What is the final product?

The final product is a multimodal system capable of high-stakes medical decision support with an emphasis on **visual interpretability**.

### Core Features:

* **Input**: Multi-modal medical images (Chest X-ray, CT, MRI, etc.) and clinical queries.
* **Output**:
  - **Grounded Medical Reports**: Textual findings accompanied by visual highlights **(bounding boxes)** of the regions of interest.
  - **Structured Reasoning**: A dedicated <think> process that provides a step-by-step clinical rationale for the generated diagnosis.

### Example Output:

* `<think> The image shows a PA view of a chest X-ray. I observe an enlarged cardiac silhouette. There is no evidence of pleural effusion. </think>`
* `<answer> {"findings": "mild cardiomegaly", "bbox_2d": [x1, y1, x2, y2]} </answer>`

### Improvements over existing processes:

* **Prevents Reward Hacking**: Uses an **odLength reward** to ensure the model only highlights relevant findings and does not over-predict boxes to "cheat" the accuracy metrics.
* **Cross-Modality Stability**: Reliable performance across eight different medical imaging modalities.

---

## 4. Why this data?

### Dataset:

**Dataset**: We will use publicly available radiology datasets like **MIMIC-CXR** or the **IU X-Ray** dataset, supplemented by benchmarks like **OmniMedVQA** for multi-task evaluation

### Suitability:

* **Benchmarking**: These datasets provide the "deterministic ground-truth" (diagnostic labels and report text) required for rule-based RL rewards.
* **Task Diversity**: They allow for training across five distinct clinical types: Anatomy Identification, Disease Diagnosis, Lesion Grading, Modality Recognition, and Biological Attributes.

### Challenges:

* **Domain-Specific Terminology**: We will address specialized medical language by initializing from a strong base model (like Qwen2-VL) and using RL to align its latent world knowledge with clinical requirements.
* **Interpretability**: To overcome the "black-box" nature of medical AI, we will implement Think-After strategies to ensure the reasoning traces are clinically sound and verifiable by human researchers.

---

## 5. Team responsibilities

| Member           | Student ID | Responsibilities                                         |
| ---------------- | ---------- | -------------------------------------------------------- |
| Le Anh Thu       | V202503040 | System development, demo interface, visualization |
| Luu Duc Toan     | V202502963 | Feature engineering, embeddings, VLM integration         |
| Tran Trung Duc   | V202401788 | Model training, experimentation, evaluation              |
| Nguyen Van Cuong | V202502961 | Data preprocessing, dataset analysis, report structuring        |

### Workflow Distribution:

* **Data pipeline:** Normalizing medical images and formatting prompts with `<think>/<answer>` tags.
* **Modeling:** Training a 3B-parameter model using RL to outperform much larger models through efficient reasoning.
* **Evaluation:** Using **IoU** for spatial accuracy, **BLEU/ROUGE**for reports, and **Greedy Precision** to check for "reward hacking".
* **System:** A demo interface showing the clinical rationale and corresponding visual evidence.

Roles are distributed to ensure balanced contribution across the pipeline.

---

## 🔗 GitHub Repository

*https://github.com/cnvcuong/VinUni_Spring26_NLP_COMP5040_FinalProject_Group1*

---
