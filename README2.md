Scenario 1:
A flower bouquet shop wants to enhance its customer experience by implementing an image classification component. The goal is to create a mobile app that allows customers to
take pictures of flowers they encounter and receive information about the flower species, care instructions, and the option to order similar flowers. A ML model has been built for image classification.

DATA OVERVIEW
An extensive flower image dataset.
Content
There are 10 different types of flowers, namely-
1.	Tulips
2.	Orchids
3.	Peonies
4.	Hydrangeas
5.	Lilies
6.	Gardenias
7.	Garden Roses
8.	Daisies
9.	Hibiscus
10.	Bougainvillea

DATA CLEANING & PRE_PROCESSING

1. Check Image Count per Category:
•	Creates a DataFrame df_count to count the number of images for each flower category.
•	Renames the axis and resets the index to display the label and count in separate columns.
2. Encode Label to Integer Category:
•	Initializes a LabelEncoder object LE.
•	Encodes the label column in raw_input DataFrame to integer categories using LE.fit_transform().
•	Checks the encoding by viewing the contents of raw_input[['label','category']].
•	Drops duplicate rows to ensure unique label-category combinations.
3. Prepare Images:
•	Initializes an empty list all_images to store resized images.
•	Iterates over the image paths in the paths list using tqdm for progress visualization.
•	Reads each image from the path using cv2.imread().
•	Resizes each image to the specified IMAGE_SIZE using cv2.resize() with interpolation method cv2.INTER_AREA.
•	Appends the resized image to the all_images list.

DATA VISUALIZATION
Looping through Flower Categories:
The code starts by iterating through different flower categories or labels. These labels represent distinct types of flowers.
Selecting Random Images:
For each flower category, the code randomly selects 9 images from that specific category. This random selection is performed by generating random indices associated with the images for that category.
Displaying Images:
It then sets up a 3x3 grid for displaying these 9 random images. Each subplot in the grid will display one image.
The code customizes the appearance of these subplots, such as removing axis ticks, adding a title that includes the flower category and the index of the image, and turning off grid lines.

DATA TRANSFORMATION 
Rotation and Shear:
The code first converts rotation and shear angles from degrees to radians. These angles represent the amount of rotation and shear to be applied to the image.
Rotation Matrix:
It calculates a 3x3 rotation matrix based on the rotation angle. This matrix is responsible for rotating the image.
Shear Matrix:
It computes a 3x3 shear matrix based on the shear angle. This matrix is responsible for applying shear transformation to the image.
Zoom Matrix:
A 3x3 zoom matrix is generated, which scales the image by different factors along its height and width. This matrix controls zooming in and out of the image.
Shift Matrix:
A 3x3 shift matrix is calculated, which shifts the image horizontally and vertically. This matrix controls image translation.
Transformation Composition:
The transformation matrices for rotation, shear, zoom, and shift are combined in a specific order using matrix multiplication. This composition results in a single 3x3 transformation matrix that represents the cumulative effect of all these transformations.
Return:
The function returns the final 3x3 transformation matrix that can be used to apply these transformations to an image 

MODEL BUILDING 
I have used a comparative model. Initially, VGG-16, then VGG-16 – Transfer Learning and then Efficient NetB7.
Input Layer:
The code starts by defining an input layer with a specified shape, which is (IMAGE_SIZE, IMAGE_SIZE, 3). This represents the dimensions of the input images, typically used in computer vision tasks.
Convolutional Blocks:
The VGG-16 architecture consists of five convolutional blocks. Each block contains a series of convolutional layers followed by a max-pooling layer. These blocks progressively extract and process features from the input image.
For each convolutional block, two convolutional layers are applied with 64, 128, 256, 512, and 512 filters, respectively. The convolutional layers use a 3x3 kernel size and 'same' padding and apply the ReLU activation function.
After each pair of convolutional layers, a max-pooling layer with a 2x2 pool size and 'same' padding is used to downsample the feature maps.
Flatten Layer:
After the five convolutional blocks, a Flatten layer is applied to reshape the 3D feature maps into a 1D vector. This step is necessary to connect the convolutional layers to the fully connected layers.
Fully Connected Layers:
Two fully connected layers follow the Flatten layer with 4096 units each. They also use the ReLU activation function. These layers provide high-level feature representation and help in classification.
Output Layer:
The final output layer has 10 units, which is often used for classifying images into one of 10 classes. It uses the softmax activation function, which is typical for multi-class classification tasks. The output units correspond to the different classes the model aims to predict.

MODEL EVALUATION

Out of all the models, EfficientNetB7 performs the best because Efficient Net can give better accuracy even with less data. 
EfficientNetB7 is known for its remarkable performance, especially in terms of accuracy and efficiency. One of its standout features is its ability to achieve impressive results even with limited data, thanks to its advanced scaling techniques and model architecture. 
RESULTS & DISCUSSION

*Due to time constraints, the model could not run through the desired number of epochs and hence has underperformed to potential.* 

CONCLUSION 
Integrating this model into a mobile application for a flower bouquet shop would significantly enhance the customer experience and provide several benefits:
1. Accurate Flower Recognition: The model's ability to identify flower species with high accuracy will allow users to easily and accurately determine the type of flower they are photographing.
3. Convenient Flower Ordering: Users will be able to seamlessly order flowers similar to the ones they have photographed directly through the mobile application. This will encourage impulse purchases and increase sales for the flower shop.
4. Enhanced Customer Engagement: The image classification feature will create a more interactive and engaging experience for users, fostering their interest in flowers and encouraging them to explore the shop's offerings.
5. Valuable Customer Insights: The collected image data and user interactions can be analyzed to gain insights into customer preferences, popular flower varieties, and seasonal trends. This information can be used to improve marketing strategies and product offerings.
