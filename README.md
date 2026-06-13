# APS360-fp

Introduction
While macro-level Music Information Retrieval (MIR) is widely researched and utilized, detecting
specific playing techniques remains a challenge.
To address this, the goal of this project is to build a Multi-Instrument Articulation Classifier capable
of identifying musical articulations from raw audio. This model will classify localized acoustic
behaviors across three instruments: Trumpet (open horn versus Harmon mute), Piano (dry staccato
versus legato sustain pedal), and Guitar (open strings versus palm-muted and individual picking
versus full chord strumming).
For human musicians, developing an ear that can detect these acoustic nuances is a skill that can take
years to master. By automating this process, the proposed model has significant educational utility
by providing immediate feedback on articulation and technique. Furthermore, this system can be
integrated into transcription software, moving beyond basic pitch detection to capture the expressive
nature of the original performance. Deep learning is an ideal tool for this task, as a neural network
can learn to isolate these harmonic structures across different instrument timbres without relying on
manual acoustic thresholding.
Illustration
A high-level overview of the data pipeline and model architecture is shown in Figure 1.

<img width="678" height="194" alt="image" src="https://github.com/user-attachments/assets/c90e13f4-70d1-4eec-9d25-3f4c125bcb6f" />
Figure 1: Proposed block diagram mapping raw audio to instrument articulation classifications via
Mel-spectrograms and a 2D CNN.
Background & Related Work
1. Smart Sample Managers: Tools like XLN Audio’s XO [1] use machine learning to categorize
and visualize drum samples by sonic similarity to aid with library organization and management.
This commercial success proves the viability and demand for using neural networks to visually and
sonically categorize short audio recordings.
2. Audio Fingerprinting (Shazam): Commercial applications like Shazam [2] represent the industry
standard for macro-level MIR. They utilize spectrogram peak-matching to identify specific recordings.
However, this approach relies on matching exact audio fingerprints against a rigid database, making
it incapable of reliably detecting songs from performances that deviate from the original recording.
3. Instrument Identification Research: Standard Convolutional Neural Networks are frequently
trained on datasets like IRMAS [3] (Instrument Recognition in Musical Audio Signals) to identify
predominant instruments in a mixed track.
4. Polyphonic Transcription Models: Google’s Magenta project [4] has researched techniques
utilizing "Onsets and Frames", using deep learning for automated piano transcription. While accurate
for note-on and note-off events, it largely ignores micro-level acoustic expressions like pedal noise or
resonance.
5. Guitar Technique Classification: Past research studies [5] have utilized Support Vector Machines
to classify specific guitar effects and techniques. However, this study focused on identifying major
spectral characteristics rather than articulations and relied on hand-crafted features rather than the
dynamic feature extraction of a modern CNN.
Data Processing
To ensure a balanced dataset, this project will utilize procedural data generation in combination with
an extensive sample library. A custom Max for Live device will be built within Ableton Live 12
to trigger multi-sampled virtual instruments. This pipeline will sequence thousands of audio chops
isolating specific articulations for each instrument. The target dataset size is approximately 1000
unique samples per articulation class, yielding a total dataset size of roughly 6000 to 7000 samples.
The raw audio will be exported in mono .wav format. It will be sliced into shorter 1-2 second intervals.
Librosa library [6] in Python will be used to compute the FFT for each clip, converting the audio
clips to Mel-spectrograkm images. These images will then be divided into training, validation, and
testing sets. These Mel-spectrograms will serve as the input tensors for the model.
Architecture
The core model will be a 2D Convolutional Neural Network (CNN). Since the input will be 2D visual
Mel-spectrograms, a CNN is the best choice to exploit the structural patterns within the frequency
domains. The architecture will rely on a series of convolutional layers to act as feature extractors.
These layers will learn to identify visual patterns on the spectrogram, such as the sharp vertical lines
of a guitar pick attack or the dense, lower frequency blocks of a legato pedal.
Once the convolutional layers have extracted the key patterns, the data is flattened out and passed into
a set of neural network layers. The final output layer of the model will generate a set of percentages,
predicting how likely the audio belongs to each specific instrument and technique.
Baseline Model
The baseline model used for comparison will be an SVM that is trained on 1D statistical summaries
of the audio clips. Standard audio features such as the brightness and general frequency texture will
be extracted from each clip. Because an SVM cannot easily interpret how a sound changes over time,
it must rely on these global averages. Comparing this traditional machine learning approach against a
CNN will show why a neural network is necessary to capture the subtle, shifting nuances of musical
articulation.
Ethical Considerations
A primary ethical concern in developing this articulation classifier is the risk of economic and cultural
bias within the training data. Because the dataset will be procedurally generated using high-quality,
multi-sampled virtual instruments, the model will inherently learn the acoustic properties of pristine,
professional-grade equipment.
If this model were integrated into educational or transcription software, this bias could inadvertently
alienate lower-income musicians or global music styles, creating a system that only functions
optimally for western, studio-standard audio
