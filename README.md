

# **DNNLS Final Assessment**
Author : Rachakatla Sai Ruthwik Raj

## **Introduction and Problem Statement**

The project repository includes the final project for the **Deep Neural Networks & Learning Systems (DNNLS)** course.
The project focuses on enhancing an existing **storytelling model** by integrating additional features and advanced learning techniques.
We have built a **high-tech story prediction system** equipped with state-of-the-art AI components for multimodal understanding and generation.

---

## **Problem Definition**

The system integrates:

* **Transformer-based cross-modal fusion**
* **Diffusion-based image generation**
* **Contrastive learning**
* **Curriculum training**
* These components collectively aim to achieve **coherent multimodal story continuation**, ensuring alignment between images, text, and semantic concepts over time.

---

## **Evaluation Metrics**

### **Image Quality Metrics**

* **SSIM (Structural Similarity Index)**
* **FID (Fréchet Inception Distance)**
* **CLIP Score**

### **Text Generation Metrics**

* **BLEU-4**
* **ROUGE-L**

### **Semantic Understanding**

* **Tag prediction accuracy**

---

## **Methods**
Here is the overview of the methods used :

* **Cross-Modal Temporal Transformer (CMTT)** integrates semantic tags with enhanced visual features to fuse image and text information across time.
* **Latent diffusion decoder** ensures high-quality and consistent image generation.
* **Temporal contrastive loss** improves alignment and consistency.
* **Early stopping** prevents overfitting.
* **Curriculum learning** stabilizes training.
* Dataset split:
  * **80% Training**
  * **10% Validation**
  * **10% Testing**

---

## **Model Architecture Overview**

![Image](https://github.com/user-attachments/assets/7b2a813b-6610-4589-a7d0-d91efe09b8fe)

The proposed architecture consists of:

* **Enhanced Visual Encoder**
  * CNN with batch normalization
* **Text Encoder**
  * BERT + LSTM
* **Semantic Tag Detection**
  * Objects, actions, and locations
* **Cross-Modal Temporal Transformer (CMTT)**
   * Multimodal temporal feature fusion
* **Latent Diffusion Decoder**
   * High-quality image generation
* **Multi-task Learning Objectives**
  * Image generation
  * Text generation
  * Tag prediction
  * Temporal contrastive learning

### **Training Enhancements**

* Curriculum learning
* Contrastive loss
* Early stopping

### **Evaluation Metrics Used**

* SSIM
* BLEU
* ROUGE
* CLIP

---

## **Reasoning-Aware Attention**

The proposed reasoning-aware attention mechanism enables the model to:

* Selectively focus on **relevant visual features**
* Attend to **textual context**
* Incorporate **semantic tags**
* Maintain **logical and temporal coherence**

This mechanism is implemented within the **Cross-Modal Temporal Transformer (CMTT)**, allowing the model to reason about **cause-effect relationships** and ensure consistent story evolution.

---

## **Code Snippet (Simplified)**

```python
# =========== ENHANCEMENT 1: CROSS-MODAL TEMPORAL TRANSFORMER ===========
class CrossModalTemporalTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=4):
        super().__init__()
        self.d_model = d_model

        # Intra-modal temporal self-attention
        self.visual_self_attn = nn.MultiheadAttention(d_model, nhead)
        self.text_self_attn = nn.MultiheadAttention(d_model, nhead)

        # Cross-modal cross-attention
        self.visual_cross_attn = nn.MultiheadAttention(d_model, nhead)
        self.text_cross_attn = nn.MultiheadAttention(d_model, nhead)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, visual_seq, text_seq):
        visual_self, _ = self.visual_self_attn(visual_seq, visual_seq, visual_seq)
        text_self, _ = self.text_self_attn(text_seq, text_seq, text_seq)

        visual_self = self.norm1(visual_seq + visual_self)
        text_self = self.norm1(text_seq + text_self)

        visual_cross, _ = self.visual_cross_attn(visual_self, text_self, text_self)
        text_cross, _ = self.text_cross_attn(text_self, visual_self, visual_self)

        visual_out = self.norm2(visual_self + visual_cross)
        text_out = self.norm2(text_self + text_cross)

        return visual_out, text_out
```

*(Full enhanced predictor implementation continues as provided.)*

---

## **Results**

The enhanced model demonstrates:

* Improved image quality
* More coherent future predictions
* Better alignment between modalities

### **Key Improvements that have being done**

* **CMTT** enhances cross-modal temporal reasoning
* **Diffusion decoding** produces sharper and more detailed images
* **Semantic tags + contrastive loss** improve logical story progression

Overall, early improvements are observed in:

* **SSIM**
* **CLIP score**
* **BLEU score**

---

## **Quantitative Analysis**

```
Results/loss_curve.png
```

---

## **Qualitative Analysis**


```
Results/sample generations.png
```
---

## **Conclusion**

We conclude that the proposed system successfully enhances multimodal story prediction by integrating advanced AI components.
The result is a **high-tech story prediction machine** with improved coherence, alignment, and generative quality across both images and text.

---

## **Future Work**

* Scaling to **longer and more complex stories**
* **Multilingual** story understanding and generation
* **Video-based inputs** instead of static images

---

