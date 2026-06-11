Your mission is to build a Conditional Generative Adversarial Network (cGAN) to perform a blind 4x Super-Resolution upscale. You will take heavily degraded 32x32 inputs and reconstruct the high-frequency biological textures to output pristine 128x128 images.
Your model will be evaluated strictly on its mathematical faithfulness to the original structure using Mean Absolute Error (MAE).
Technical Constraints 
As a reminder from the official document:

No Pre-trained Weights: You must initialize your core Generator and Discriminator randomly. (You may only load the provided VGG-19 weights for Perceptual Loss).

Environment: Your Kaggle notebook must have Internet Access OFF and External Data DISALLOWED.

Format Check: Ensure you denormalize your final pixel predictions back to 0-255 integers before creating your flattened space-separated CSV string.






Overview
In modern Agri-Tech, automated drones and cheap mobile sensors are deployed to photograph crop leaves and detect early signs of disease. However, due to hardware limitations, thermal sensor noise, and severe 2G cellular compression, the images transmitted back to the central server are often heavily degraded. If an automated diagnostic system attempts to classify these 32 X 32 pixel distorted images, it will fail.
Your mission is to build a Conditional Generative Adversarial Network (cGAN) or its variants to perform a blind 4x Super-Resolution upscale. You must take the degraded 32 X 32 inputs and reconstruct the high-frequency biological textures (veins, chlorosis, and necrotic lesions) to output pristine 128 X 128 images.
The Technical Challenge
The low-resolution inputs provided in the dataset contain severe information loss, noise JPEG artifacts. You must train an architecture natively that learns to denoise and upscale simultaneously without hallucinating fake biological symptoms.
Strict Architectural Constraints
Your code must strictly adhere to the following:
Constraint 1: You must initialize your networks randomly. The use of pre-trained weights (e.g., via torchvision.models, Hugging Face, or loaded .pth files) to solve the core task is strictly prohibited.
Constraint 2: Your final inference notebook must be submitted via Kaggle with Internet Access OFF and External Data DISALLOWED.
Leaderboard Evaluation Metric


While you may use Adversarial and Perceptual losses to train your network, the Kaggle leaderboard will evaluate your final 495 submitted images against the hidden Ground Truth using Mean Absolute Error (MAE) calculated pixel-by-pixel across the flattened CSV. This metric specifically tests the accuracy of your reconstruction. If your GAN generates a beautifully realistic bacterial spot, but places it in the wrong physical coordinate compared to the hidden true image, the MAE penalty will be severe. Your model must not only generate textures that look real, but it must be mathematically faithful to the underlying structure of the degraded input. 


Note: This competition uses a minimization metric. A lower MAE score indicates better performance.

Evaluation
Submissions are evaluated using the Mean Absolute Error (MAE) between your predicted high-resolution images and the hidden ground-truth images.
MAE measures the average magnitude of the errors in a set of predictions, without considering their direction. In the context of this super-resolution task, MAE calculates the absolute difference between your predicted pixel intensities and the actual pixel intensities across all three RGB channels of the final 128x128 image. 
Why MAE for the Leaderboard? While you may use Perceptual and Adversarial losses to train your GAN to generate realistic textures, the Kaggle backend uses MAE to ensure those textures are not just hallucinations. MAE strictly penalizes spatial inaccuracies. If your model generates a realistic bacterial spot, but places it 5 pixels away from where it exists on the ground-truth leaf, the MAE will increase. To win, your model must be both visually realistic and mathematically faithful to the original degraded data.


Submission File
The Pixel Flattening Rules:
To get your score, your submission.csv must strictly adhere to the following mathematical format:
The Dimensions: Every 128x128 RGB image must be flattened into a single 1D sequence of exactly 49,152 numbers (128 x 128 x 3 color channels).
The Delimiter: The numbers in this sequence must be separated by SPACES ONLY. Do not use commas, brackets, or semicolons to separate the pixel values.
The Data Type: The entire sequence of 49,152 numbers must be treated as a single text string inside the CSV cell.
The Order: The pixels should be flattened using standard NumPy/OpenCV row-major order: (R1, G1, B1, R2, G2, B2...).
Denormalize their outputs back to 0-255 integers before creating the CSV. Note that each pixel value present in csv must have range [0,255].
The file submission.csv must contain a header and have the exact following format:
Id,Pixels
agrivision_test_0000.png,124 125 120 119 122 …
agrivision_test_0001.png,89 90 92 85 88 …
agrivision_test_0002.png,45 48 50 51 49 …
Dataset Description
The challenge involves a 4x Super-Resolution task where participants must reconstruct high-fidelity textures from severely degraded, low-resolution sensor data.
Files



You will find the following directory structure once you attach this dataset to your Kaggle Notebook:
train_High_Resolution/: Contains 1,642 pristine 128x128 images of crop leaves. These are your "Ground Truth" labels used for training.
train_Low_Resolution/: Contains 1,642 degraded 32x32 images. These are the inputs your model must learn to upscale.
test_Low_Resolution/: Contains 495 degraded 32x32 images. You must generate the high-resolution versions of these images for your final submission.
vgg19_weights.pth: The pre-trained weights for the VGG-19 network. This is provided to allow the calculation of Perceptual Loss in an offline environment (Internet Access: OFF) in case you want to use perceptual loss.
Data Format & Dimensions
All images are provided in .png format.
Low-Resolution (LR) Inputs: 32x32 pixels, 3 channels (RGB).
High-Resolution (HR) Outputs: 128x128 pixels, 3 channels (RGB).
What am I predicting?
Your task is to predict the high-resolution version of every image in the test_Low_Resolution folder. Your submission must be a CSV file where each row contains the filename as first column named Id(e.g., agrivision_test_0000.png) and a string of space-separated pixel values representing the 128x128x3 reconstructed image as second column named Pixels.
Key Acronyms & Terminology
HR: High-Resolution (The 128x128 targets).
LR: Low-Resolution (The 32x32 inputs).
cGAN: Conditional Generative Adversarial Network.
MAE: Mean Absolute Error (The metric used to score the leaderboard).
Perceptual Loss: A loss function calculated using the provided VGG-19 weights to ensure biological texture realism.
Column Definitions (for Submission)
Id: The exact filename of the test image as found in the test folder (e.g., test_042.png).
 Pixels: A space-separated string of 49,152 integers. This is the flattened 128x128x3 RGB array. 
Loading VGG Weights
Because the competition environment has Internet Access DISABLED, you cannot use pretrained=True. If you want to use perceptual loss, you can load vgg-19 weights given to you in the data file and use it to calculate the loss.


Scoring criteria: 
scoring will be based on private leaderboard scores (relative grading) and it will depend on baseline score(17.3529644) as well. If a student gets more or equal to baseline score, then 50% marks will be reduced. If the kaggle score is more than or equal to 30 then 0 marks will be given.
Note that: lower score indicates better performance, if you generate the perfect images for test data, your kaggle leaderboard score will be 0.


Submission File
**You must convert your final 495 generated 128x128 images into a flattened CSV format. Each image will be converted into a single sequence of 49,152 numbers (128 x 128 pixels x 3 RGB color channels). ** For each degraded image in the test_LR folder, you must predict the reconstructed high-resolution image, flatten its pixel values into a single space-separated string, and map it to its exact original filename.

The file submission.csv must contain a header and have the exact following format:

`Id,Pixels

agrivision_test_0000.png,124 125 120 119 122 …

agrivision_test_0001.png,89 90 92 85 88 …

agrivision_test_0002.png,45 48 50 51 49 …`







Dataset Description
The challenge involves a 4x Super-Resolution task where participants must reconstruct high-fidelity biological textures from severely degraded, low-resolution sensor data.

Files
You will find the following directory structure once you attach this dataset to your Kaggle Notebook:

train_High_Resolution/: Contains 1,642 pristine 128x128 images of crop leaves. These are your "Ground Truth" labels used for training.
train_Low_Resolution/: Contains 1,642 degraded 32x32 images. These are the inputs your model must learn to upscale.
test_Low_Resolution/: Contains 495 degraded 32x32 images. You must generate the high-resolution versions of these images for your final submission.
vgg19_weights.pth: The pre-trained weights for the VGG-19 network. This is provided to allow the calculation of Perceptual Loss in an offline environment (Internet Access: OFF).
Data Format & Dimensions
All images are provided in .png format.

Low-Resolution (LR) Inputs: 32x32 pixels, 3 channels (RGB).
High-Resolution (HR) Outputs: 128x128 pixels, 3 channels (RGB).
What am I predicting?
Your task is to predict the high-resolution version of every image in the test_Low_Resolution folder. Your submission must be a CSV file where each row contains the filename (e.g., agrivision_test_0000.png) and a string of space-separated pixel values representing the 128x128x3 reconstructed image.

Key Acronyms & Terminology
HR: High-Resolution (The 128x128 targets).
LR: Low-Resolution (The 32x32 inputs).
cGAN: Conditional Generative Adversarial Network.
MAE: Mean Absolute Error (The metric used to score the leaderboard).
Perceptual Loss: A loss function calculated using the provided VGG-19 weights to ensure biological texture realism.
Column Definitions (for Submission)
Id: The exact filename of the test image as found in the test folder (e.g., test_042.png).
* Pixels: A space-separated string of 49,152 integers. This is the flattened 128x128x3 RGB array.
Loading VGG Weights
you cannot use pretrained=True. If you want to use perceptual loss, you can load vgg-19 weights given to you in data file and use it to calculate the loss.




