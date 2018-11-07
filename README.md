# Landmarks_recognition

This project presents different models used for classification of images with different touristic landmarks. The tourist landmarks are easily recognizable and well-known sites and buildings, such as a monument, church, bridge, etc.

## Dataset

The dataset contains 184.732 images from 17 different landmarks:
- Alcatraz
- Alhambra Palace
- Basilica of Saint Peter
- Casa Batllo
- Charles Bridge
- Eiffel
- Hollywood Sign
- Itsukushima Shrine
- Moulin Rouge
- National Art Museum of Catalonia
- Park Guell
- Petronas Twin Towers
- Rialto Bridge
- Roman Coliseum
- Rome Pantheon
- Statue of Liberty
- Triomphe

Was constructed based on five sources:
- Google Landmarks Dataset [1]
- Paris [2]
- Flickr
- Personal photos
- Web images

Using human labeling in order to ensure the reliability of the annotation. One of the aims was to solve annotation errors of the original datasets and inaccuracy introduced by outdoor and indoor images of the same landmark sharing the annotation despite being visually distinguishable.

The images were resizing to 512 x 521 with two techniques: black borders and squashing with Lanczos filter. The models present better results with the squashing technique.

Finally, the dataset was splitted in 3:
- Train: 80%
- Validation: 10%
- Test: 10%

[Download dataset](https://drive.google.com/open?id=1olkjN-mh8CE3IZN4AqzVAL7XLOVIHmyZ)

## Experiments

- **Convolutional Network:**
  - 5 convolutional layers and a fully connected layer
  - Activation function: ReLU
  - Data augmentation
  - L2 Regularization
  - Dropout
  - Batch Normalization
  - Max Pooling
  - Xavier initialization
- **Transfer Learning:** ResNet50 and VGG19 as fixed feature extractor pretrained on ImageNet.

## Results

![alt text](https://github.com/JimeMartinez/Landmarks_recognition/edit/master/Results.png "Results")

VGG19 acquired the best performance with a 97,97% of accuracy on the test set.

## References

[1] H. Noh, A. Araujo, J. Sim, T. Weyand, B. Han, "Large-Scale Image Retrieval with Attentive Deep Local Features", Proc. ICCV'17

[2] J. Philbin, O. Chum, M. Isard, J. Sivic, and A. Zisserman. Lost in quantization: Improving particular object retrieval in large scale image databases. In CVPR, 2008.
