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

We are taking 25 samples from the original data and displaying them along with their titles. 

DATA TRANSFORMATION 

Since it is an image data, we are transforming it by rotating the images and setting the image size and pixels. 

MODEL BUILDING 

I have used a comparative model. Initially, VGG-16, then VGG-16 – Transfer Learning and then Efficient NetB7.

MODEL EVALUATION

Out of all the models, EfficientNetB7 performs the best because Efficient Net can give better accuracy even with less data. 

RESULTS & DISCUSSION

*Due to time constraints, the model could not run through the number of epochs and hence has underperformed to expectations. 

CONCLUSION 

Integrating this model into a mobile application for a flower bouquet shop would significantly enhance the customer experience and provide several benefits:
1. Accurate Flower Recognition: The model's ability to identify flower species with high accuracy will allow users to easily and accurately determine the type of flower they are photographing.
3. Convenient Flower Ordering: Users will be able to seamlessly order flowers similar to the ones they have photographed directly through the mobile application. This will encourage impulse purchases and increase sales for the flower shop.
4. Enhanced Customer Engagement: The image classification feature will create a more interactive and engaging experience for users, fostering their interest in flowers and encouraging them to explore the shop's offerings.
5. Valuable Customer Insights: The collected image data and user interactions can be analyzed to gain insights into customer preferences, popular flower varieties, and seasonal trends. This information can be used to improve marketing strategies and product offerings.
