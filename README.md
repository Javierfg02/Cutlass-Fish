# 🐟 Crafting Sign Language From Speech – *Cutlass Fish*
SLP Model that translates speech to sign language

**Contributors:** Nya Haseley-Ayende, Javier Fernandez Garcia, Halleluiah Girum, Bahar Birsel, Amaris Grondin  
**GitHub Repository:** [Cutlass Fish](https://github.com/Javierfg02/Cutlass-Fish)
**Poster:** [Canva Link](https://www.canva.com/design/DAGEMGYmTh8/tvMLaYGDblZ74kz5-fWmKA/edit?utm_content=DAGEMGYmTh8&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)

---

## 📌 Overview

Sign language translation plays a pivotal role in making communication more inclusive for over 466 million people worldwide with hearing impairments. While many models exist for converting sign language to speech (SLT), fewer efforts focus on the reverse process—translating speech to sign language (SLP). 

Our project, *Crafting Sign Language From Speech*, aims to address this gap by creating a tool that translates English sentences into American Sign Language (ASL) poses using a deep learning model known as a **Progressive Transformer**.

A writeup and reflection of our project is available [here (PDF)](./writeup_reflection.pdf).

---

## 📚 Introduction

Sign language is essential for equitable communication, especially in settings like education, healthcare, and social engagement. While prior research has prioritized SLT, true accessibility requires **two-way translation**. Our system focuses on producing **continuous ASL pose sequences from English input**—a process known as **Sign Language Production (SLP)**.

---

## 🧠 Related Work

We built upon work like:

- **Saunders et al.**'s *Progressive Transformers for End-to-End Sign Language Production* (used Progressive Transformer for SLP).
- **Zelinka et al.**'s work on synthesizing sign language skeletons from text using OpenPose and RNN-based encoder-decoder models.

Key improvements in our work:

- Focused on **ASL**, not German Sign Language.
- Used the **How2Sign** dataset, which includes **facial movement data**.
- Implemented our model in **TensorFlow**, adapting ideas from PyTorch-based architectures.

---

## 📦 Dataset

We use the [How2Sign Dataset](https://how2sign.github.io/), a multimodal resource with over **80 hours** of ASL videos paired with English translations.

From this, we use:

- **Pre-extracted 3D keypoints** representing body, face, and hands.
- Each sample links an English sentence with its corresponding ASL pose sequence.

---

## ⚙️ Methodology

### Data Preprocessing

- **Source:** Tokenized English sentences.
- **Target:** Corresponding ASL skeleton keypoints (face, hands, body).
- Created a vocabulary mapping and converted data to padded tensors.

### Model Architecture

We use a **Progressive Transformer** architecture with:

1. **Symbolic Encoder**  
   - Transforms text into semantic embeddings using multi-head self-attention.

2. **Progressive Decoder**  
   - Generates ASL poses and tracks sentence progression using a **counter-decoding** technique (values range from 0 to 1).

### Training Details

- **Loss Function:** Mean Squared Error (MSE).
- **Optimizer:** Adam.
- **Epochs:** 250–500.
- **Learning Rate:** Dynamic from 0.001 to 0.0002.
- **Hardware:** Trained on Apple M2 CPU (GPU used if available).

---

## 📊 Results

### Our Model

- **Training Loss:** Converged to ~0.00476 after ~65 epochs.
- **Testing Output:** No video generation yet due to bugs.
- Compared favorably in loss to the Saunders model (~0.007), though lacking in DTW comparison.

### Author's Model

- **Test Loss:** Stabilized after ~70 epochs.
- **Best DTW Score:** ~16, worsens over time.
- Likely overfitting by minimizing MSE without improving sequence alignment.

> 🧠 **Key Insight:** Minimizing MSE doesn't always mean better visual output—more advanced metrics and loss functions may be needed.

---

## ❗ Challenges

- **Data preprocessing:** Complex merging of face, hand, and body components into a unified skeleton format.
- **Model Implementation:** TensorFlow lacks direct support for counter-decoding; we built the transformer model from scratch.
- **Code Adaptation:** Understanding and adapting PyTorch-based research code was time-consuming.

---

## 💭 Reflection

### How did the project go?

- ✅ Achieved base goal: Train a speech-to-sign model with low loss.
- ❌ Did not achieve stretch goals: Video output and full two-way translation.
- 💡 Lessons: Importance of incremental testing, rigorous debugging, and team coordination.

### What would we do differently?

- Spend more time testing each component before integration.
- Adjust loss function to emphasize **finger movements**, the most meaningful part of ASL.
- Pace the work to reduce tech debt and implementation bugs.

---

## 🚀 Future Work

- **Fix video output** to evaluate pose sequences visually.
- Implement **BLEU/ROUGE scores** adapted for sign language.
- Improve **loss weighting** to prioritize meaningful joint movements.
- Incorporate **facial expressions** into output.
- Experiment with **noise injection** for regularization and better sequence generation.

---

## 🧠 Key Takeaways

- Deep dive into **cutting-edge research** and model design.
- Real-world experience with **large-scale data preprocessing** and **group collaboration**.
- Learned to **debug, optimize, and translate** complex deep learning systems from paper to implementation.

---

## 📸 Poster Preview

A visual summary of our project is available [here (PDF)](./poster.pdf).

---

## 💵 Acknowledgments

Supported in part by **$3 million in endowments** for inclusive tech research and education.
