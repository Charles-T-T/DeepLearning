# Requirement

Upload a **pdf** file with codes, analysis/answers and pictures for all the questions.

# Chapter 2
1. Create a tensor `a` from `list(range(9))` . Predict then check what the size, offset, and strides are.
2. Create a tensor `b = a.view(3, 3)` . What is the value of `b[1, 1]` ?
3. Create a tensor `c = b[1:,1:]` . Predict then check what the size, offset, and strides are.

# Chapter 3
1. Take several pictures of red, blue, and green items with your phone or other digital camera, or download some from the internet if a camera isn’t available.
- Load each image, and convert it to a tensor.
- For each image tensor, use the `.mean()` method to get a sense of how bright the image is.
- Now take the mean of each channel of your images. Can you identify the red, green, and blue items from only the channel averages?

2. Select a relatively large file containing Python source code.
- Build an index of all the words in the source file. (Feel free to make your tokenization as simple or as complex as you like; we suggest starting by replacing `r"[^a-zA-Z0-9_]+"` with spaces.)
- Compare your index with the one you made for Pride and Prejudice. Which is larger?
- Create the one-hot encoding for the source code file.
- What information is lost with this encoding? How does that information compare with what’s lost in the Pride and Prejudice encoding?

# Chapter 4

1. Redefine the model to be `w2 * t_u ** 2 + w1 * t_u + b` .
- What parts of the training loop and so on must be changed to accommodate this redefinition? What parts are agnostic to swapping out the model?
- Is the resulting loss higher or lower after training? Is the result better or worse?
- Draw the relationship between the epoch and train/validation loss in one picture by matpolotlib.

# Chapter 5

1. Experiment with the number of hidden neurons in your simple neural network model, as well as the learning rate.
- What changes result in a more linear output from the model?
- Can you get the model to obviously overfit the data?

2. The third-hardest problem in physics is finding a proper wine to celebrate discoveries. Load the wine data from chapter 3 and create a new model with the appropriate number of input parameters.
- How long does it take to train compared to the temperature data you’ve been using?
- Can you explain what factors contribute to the training times?
- Can you get the loss to decrease while training on this data set?
- How would you go about graphing this data set?
