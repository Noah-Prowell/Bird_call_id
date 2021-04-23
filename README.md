# Bird Call ID

My objective for this project is to be able to predict the species of a bird given their call.  My data is was provided by kaggle and is a huge data set of over 20,000 audio files with over 250 bird species.  Each bird species has a range of 20 - 100 different audio files to make predicitons on.  

## Method to the Madness
My method for tackling this classification problem is to change each adio file into a spectrogram image.  The reason for doing this is to encompass all the features of the audio file into an image that I can then use a CNN to train on the images and make predictions.  CNN's are very powerful neural networks and are trainable on a variety of different images making them perfect to try and solve this problem.  To make this problem a bit simpler to start I trained on only 5 classes instead of 254.  

![Bird](160820341-480px.jpg)

## Initial Model
My initial neural network was a simple CNN with only around 4 million paramaters to train on, which for a simple neural network is not a tiny amount but certainly is not the largest there could be.  This model was a constant struggle to get to see the patterns that are in these images.  In the ned after days of tinkering and training the model I was only able to get it to a measly 32% accuracy, and in this case random guessing is a 20% accuracy, so my model is not doing well at picking up the patterns in my bird calls.

<p align="center">
    <img src="twentythree_cont.png" size = '600x600'/>
</p>
<br/>
Obviously this is not a great model and so I wanted to improve upon it.

## Transfer Learning
In an attempt to improve my model I employed the technique of transfer learning to improve my results.  I chose to do this instead of trying to continue my current model because theres only so much I can do with my simple neural network.  The transfer learning models available from Tensorflow are extremely powerful and pretrained for much longer than I ever could with my model making them perfect for this problem.  The transfer model I chose to use is the Inception-ResNet V2 using the pretrained imagenet weights.  Using this model proved to be extremely successful!  Using this technique I was able to get my training accuracy to 99.99% and my validation accuracy to 60% which is a massive improvment!  Below is is an example of what the model looks like trained to 1000 epochs which is excessive, but I wanted to see where all of my accuracies and losses were moving to and whether they were still improving or not.

<p align="center">
    <img src="thousand_epochs.png" size = '600x600'/>
</p>
<br/>
This is still only trained on the five classes, but I believe it is good enough to be able to move onto the full 254 classes.  
