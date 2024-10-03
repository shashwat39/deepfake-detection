# DL-Based Solution for Detection of Face-Swap Deep Fake Photos

## Project Overview

This project aims to develop a deep learning-based solution for detecting face-swap deep fake images. With the rapid rise of social media platforms, fake or edited images are increasingly being used to spread misinformation, causing significant harm. Our goal is to create an efficient image classification system capable of accurately identifying deep fake images. This solution can be scaled for real-world applications, ensuring robustness and efficiency in terms of both computational and memory usage.

## Team Members

| Name                        | Role                      | GitHub Profile                           |
|-----------------------------|---------------------------|------------------------------------------|
| **Ashish Tirupati Bollam**   | Backend Engineer           | [ashish4bollam](https://github.com/ashish4bollam)  |
| **Pragya Gupta**             | ML/MLOps Engineer          | [iampragyagupta](https://github.com/iampragyagupta)    |
| **Rohan Nijhawan**           | Data Engineer              | [Rohan-1704](https://github.com/Rohan-1704)      |
| **Samarth Malhotra**         | DevOps Engineer            | [samarthgh](https://github.com/samarthgh)  |
| **Shashwat Srivastava**      | Data Scientist             | [shashwat39](https://github.com/shashwat39)|
| **Siddhartha Arora**         | Software Engineer          | [siddhartha254](https://github.com/siddhartha254)|

---

## Table of Contents
1. [Problem Statement](#problem-statement)
2. [Data Sources](#data-sources)
3. [Data Pipeline](#data-pipeline)
4. [Model Development](#model-development)
5. [User Interface](#user-interface)
6. [Expected Results](#expected-results)
7. [References](#references)

---

## Problem Statement

With the rise of social media, individuals can share and disseminate photos globally within seconds. However, this has given rise to the propagation of manipulated or fake images, including deep fakes, which can mislead users and contribute to the spread of misinformation. The problem is amplified by the increasing sophistication of deep fake technology, making it difficult for the human eye to detect forgeries.

Our solution uses deep learning to detect fake images by analyzing image data and predicting whether an image is real or manipulated. We have decided to deploy in online. It's something that can be rethought and changed, but generally knowing the problem (and the target variable to be predicted) already gives us an idea which style of deployment will generate more value for the business.

![Screenshot 2024-09-30 235045](https://github.com/user-attachments/assets/f446f53e-fa99-4b55-af24-fe8e7cd32b95)


---

## Data Acquisition

Our dataset comprises a mix of publicly available resources for real and deep fake images. However, due to computational constraints, we performed **weighted sampling** to create a manageable dataset for training purposes. Below are the links to the data sources we utilized:

1. **[deepfake and real images](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images?resource=download)** - This dataset contains manipulated images and real images. The manipulated images are the faces which are created by various means. Each image is a 256 X 256 jpg image of human face either real or fake.
2. **[CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)** - CelebFaces Attributes Dataset (CelebA) is a large-scale face attributes dataset with more than 200K celebrity images, each with 40 attribute annotations. The images in this dataset cover large pose variations and background clutter. CelebA has large diversities, large quantities, and rich annotations.
3. **[Real and Fake Face Detection](https://www.kaggle.com/datasets/ciplab/real-and-fake-face-detection)** - This dataset contains expert-generated high-quality photoshopped face images. The images are composite of different faces, separated by eyes, nose, mouth, or whole face.
4. **[140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)** - This dataset consists of all 70k REAL faces from the Flickr dataset collected by Nvidia, as well as 70k fake faces sampled from the 1 Million FAKE faces (generated by StyleGAN) that was provided by Bojan.

---

## Data Preparation

We have designed a comprehensive data pipeline to process our dataset, including steps for data cleaning, preprocessing and sampling. The entire pipeline ensures a smooth flow of data from raw input to model-ready formats.

**Data Pipeline Overview:**
![Data Pipeline](path/to/data_pipeline_image.png)

---

## Model Development

We developed three models for detecting deep fake images, each leveraging state-of-the-art techniques in computer vision and deep learning. These models were benchmarked to provide a robust baseline and ensure continuous improvement through iterative testing.

1. **AVFF: Audio-Visual Feature Fusion for Video Deepfake Detection**  
   *Authors:* Oorloff, T., Koppisetti, S., Bonettini, N., Solanki, D., Colman, B., Yacoob, Y., Shahriyari, A., & Bharaj, G. (2024)
   *[Link to Paper](https://arxiv.org/abs/2406.02951)*  
   *Description:* This paper introduces Audio-Visual Feature Fusion (AVFF), a two-stage deepfake detection method that uses self-supervised learning to capture audio-visual correspondences, achieving state-of the-art results on the FakeAVCeleb dataset.
2. **UCF: Uncovering Common Features for Generalizable Deepfake Detection**  
   *Authors:* Yan, Z., Zhang, Y., Fan, Y., & Wu, B. (2023)  
   *[Link to Paper](https://arxiv.org/abs/2304.13949)*  
   *Description:* This paper introduces a disentanglement framework to generalize deepfake detection by uncovering common forgery features, addressing the overfitting problem and improving performance on unseen forgeries.

3. **Implicit Identity Driven Deepfake Face Swapping Detection**  
   *Authors:* Huang, B., Wang, Z., Yang, J., Ai, J., Zou, Q., Wang, Q., & Ye, D. (2023) 
   *[Link to Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Huang_Implicit_Identity_Driven_Deepfake_Face_Swapping_Detection_CVPR_2023_paper.pdf)*  
   *Description:* This paper introduces a novel implicit identity-driven framework for detecting face-swapped deepfakes by exploring differences between explicit and implicit identities, significantly improving detection performance across datasets.

---

## User Interface

A key part of our project is the user-facing interface, which allows end users to easily upload images and receive predictions about their authenticity.

**User Interface Screenshot:**
![Screenshot 2024-10-03 163459](https://github.com/user-attachments/assets/6366dca8-26e9-4b1b-b123-feb7cb6fc2eb)


The UI is designed to be intuitive, with a clean layout where users can upload an image and view results in real-time. It includes a dashboard to visualize key statistics and confidence scores generated by the models.

---

## Expected Results

Our system is expected to achieve the following:

1. **Baseline Model and Benchmarking**: Create a strong baseline for measuring model performance, guiding the development of more advanced solutions.
2. **Performance Metrics**: We aim to achieve high scores in key performance metrics such as accuracy, precision, recall, F1 score, and ROC-AUC curve.
3. **Software Efficiency**: Optimize the system for memory usage, compute efficiency, latency, and throughput to ensure it can scale efficiently in real-world scenarios.
4. **Iterative Testing and Cross-Validation**: The models will undergo continuous iterative testing with cross-validation to avoid overfitting and ensure consistent performance across diverse datasets.

---

## References

Below are the key research papers that guided our model development:

1. **He, K., et al. (2016)** - [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
2. **Zhou, P., Han, X., Morariu, V.I., & Davis, L.S. (2018)** - [Learning Rich Features for Image Manipulation Detection](https://arxiv.org/abs/1805.04953)
3. **Huh, M., Liu, A., Owens, A., & Efros, A.A. (2018)** - [Fighting Fake News: Image Splice Detection via Learned Self-Consistency](https://arxiv.org/abs/1805.04096)
