# Multi-Instrument Articulation Classifier (APS360-fp)

An engineering framework bridging acoustic signal processing with deep learning to automate the detection and classification of localized musical articulations from raw audio signals.

---

## Introduction

While macro-level Music Information Retrieval (MIR) is widely researched and utilized, detecting specific playing techniques remains a significant challenge. The goal of this project is to build a **Multi-Instrument Articulation Classifier** capable of identifying distinct musical articulations directly from raw audio. 

The model classifies localized acoustic behaviors across three target instruments:
* **Trumpet:** Open horn versus Harmon mute
* **Piano:** Dry staccato versus legato sustain pedal
* **Guitar:** Open strings versus palm-muted, and individual picking versus full chord strumming

For human musicians, developing an ear that can detect these acoustic nuances is a skill that can take years to master. By automating this process, the proposed model offers significant educational utility by providing immediate feedback on articulation and technique. Furthermore, this system can be integrated into transcription software, moving beyond basic pitch detection to capture the expressive nature of the original performance. 

Deep learning is an ideal tool for this task, as a neural network can learn to isolate these complex harmonic structures across different instrument timbres without relying on manual acoustic thresholding.

---

## System Illustration

<img src="https://github.com/user-attachments/assets/c90e13f4-70d1-4eec-9d25-3f4c125bcb6f" width="678" height="194" alt="Proposed Block Diagram" />

> **Figure 1:** Proposed block diagram mapping raw audio to instrument articulation classifications via Mel-spectrograms and a 2D CNN.

---

## Background & Related Work

1. **Smart Sample Managers (e.g., XLN Audio’s XO [1]):** These tools use machine learning to categorize and visualize drum samples by sonic similarity to aid library organization. This commercial success proves the viability and demand for using neural networks to visually and sonically categorize short audio recordings.
2. **Audio Fingerprinting (e.g., Shazam [2]):** Commercial applications like Shazam represent the industry standard for macro-level MIR, utilizing spectrogram peak-matching to identify specific recordings. However, this approach relies on matching exact audio fingerprints against a rigid database, making it incapable of reliably detecting performances that deviate from the original source track.
3. **Instrument Identification Research [3]:** Standard Convolutional Neural Networks are frequently trained on datasets like IRMAS (Instrument Recognition in Musical Audio Signals) to identify predominant instruments within a mixed track.
4. **Polyphonic Transcription Models (e.g., Google’s Magenta Project [4]):** Advanced research utilizing "Onsets and Frames" leverages deep learning for automated piano transcription. While highly accurate for note-on and note-off events, it largely ignores micro-level acoustic expressions like pedal noise or resonance.
5. **Guitar Technique Classification [5]:** Past research studies have utilized Support Vector Machines (SVMs) to classify specific guitar effects and techniques. However, these studies focused on identifying major spectral characteristics rather than subtle articulations, relying heavily on hand-crafted features rather than the dynamic feature extraction of a modern CNN.

---

## Data Processing Pipeline

To ensure a balanced dataset, this project utilizes procedural data generation in combination with an extensive sample library. 

### 1. Data Generation
A custom **Max for Live** device is built within Ableton Live 12 to programmatically trigger multi-sampled virtual instruments. This pipeline sequences thousands of audio chops, isolating specific articulations for each instrument. 
* **Target Dataset Size:** Approximately 1,000 unique samples per articulation class
* **Total Dataset Volume:** Roughly 6,000 to 7,000 unique samples
* **Format:** Mono `.wav` files sliced into short 1 to 2 second intervals

### 2. Feature Engineering
The Librosa library in Python is utilized to compute the Fast Fourier Transform (FFT) for each audio clip, converting the raw acoustic signals into **2D Mel-spectrogram images**. These spectrograms visually preserve the time-frequency energy distribution and serve as the direct input tensors for the network. The pipeline automatically divides these tensors into distinct training, validation, and testing datasets.

---

## Model Architecture

The core architecture consists of a **2D Convolutional Neural Network (CNN)**. Because the input tensors are 2D visual Mel-spectrograms, a CNN is uniquely suited to exploit spatial and structural patterns within the frequency domains.
