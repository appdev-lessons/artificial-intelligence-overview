# Intro to Artificial Intelligence

> "Writing is thinking. To write well is to think clearly. That's why it's so hard."
>
> — David McCullough

## Overview

One of the objectives of this lesson is to provide future developers, managers, and leaders with a foundational understanding of artificial intelligence (AI), one of the most transformative technologies of our time.

For the last two decades, the kind of technology ["eating the world"](https://a16z.com/why-software-is-eating-the-world/) was internet-based, CRUD apps like Instagram and Airbnb. So we set out to learn how to build that kind of software.

That is still most of what this course teaches. But in the last few years, artificial intelligence and machine learning (ML) have joined it. Developers now build applications on top of powerful AI models, such as the ones behind [ChatGPT](https://chatgpt.com/) and [Claude](https://claude.ai/), and more and more new software has a model somewhere inside it.

## AI for Developers

You don't need to be an ML expert to start using AI in your projects. Building your own intuition and mental model of how AI systems work can help you make your own AI-driven software. The purpose of this lesson is to give you some intuition, so that you can start applying this new technology to your own projects.

If you'd like to see these ideas animated before or after you read about them, [Large Language Models explained briefly](https://www.youtube.com/watch?v=LPZh9BOjkQs) by 3Blue1Brown covers much of the same ground in about eight minutes. To go a lot deeper, the [Fast.ai course](https://course.fast.ai/) teaches practical ML and AI development.

## You Don't Have to Build the Model

Building one of these models takes an enormous amount of data, computing power, and money, and only a handful of organizations can do it. *Using* one takes very little. Labs such as OpenAI, Anthropic, and Google make their models available to developers through APIs: your program sends some text over the internet, and the model's output comes back. You'll do exactly that later in this course.

Other models are released as **open weights**, which means anyone can download the model and run it on their own hardware. We'll see what "weights" are shortly.

OpenAI also created their own consumer-facing product called ChatGPT. It was supposed to mainly be an advertisement for their API, but it became one of the fastest-growing consumer applications ever launched.

## What a Model Like GPT Actually Does

Before we get into how these models are built, it's worth being precise about what one does when you use it, because it is a much narrower thing than it looks from the outside.

You give it a chunk of text, say `The best thing about AI is its ability to`, and it produces exactly one thing: a guess at the next small piece of text. That piece is called a **token**, and what comes back is called the **output**, or the **completion**. A token is often a whole word, but not always; long or unusual words get broken into several.

One token is not much of a sentence. To get a sentence, you take the prompt plus the token it just produced, feed the whole thing back in, and ask again. Then again. A paragraph of AI-generated prose is that loop run a few hundred times, each pass adding one more token to the end of its own output.

And the model does not actually pick the next token. It produces a **probability for every token that could come next**, across its entire vocabulary, and then something else has to choose among those candidates.

It chooses with a bit of randomness. The likely tokens usually win, but not always. That is why asking the same question twice gives you two differently worded answers: nothing is broken, a slightly different token got picked early on, and everything after it followed from there.

### Context

The thing that moves the output most is how much you give the model to work with. Consider what a model would do with each of these three prompts in turn:

- "Ruby"
- "Ruby is a"
- "Ruby is a programming"

The further we steer the prompt down a path, the more specific and predictable the results become. This is called providing **context**.

Context matters so much because the model has only two sources of information. One is its weights (more on those below), which were fixed when its training ended, often months or years before you are talking to it. The other is the context: the text in front of it right now. It does not learn from your conversation, and when you start a new chat, it starts from nothing. Products that seem to remember you are quietly pasting notes from earlier chats into the context.

Keep this in mind when you ask an AI for help with your code. It can't see your screen, your files, or your error message unless something puts them in the context. Paste in the code and the exact error, and the answers get much more specific, for the same reason that "Ruby is a programming" did.

### From Autocomplete to Assistant

Through experimentation, people realized that giving a GPT the first part of a conversation and then letting it write the next message led to good results. That trick became ChatGPT. The models have since been trained further on huge numbers of example conversations, so the helpful-assistant behavior is now built in. Underneath, though, it is the same loop: the whole conversation so far is the prompt, and the reply is produced one token at a time.

Notice how that discovery was made. This is more like biology than the computer science we're used to; rather than designing all capabilities up front and then painstakingly codifying them by hand, we develop a model and then poke at it to discover its emergent properties.

Two later developments are worth knowing about, and both are the same loop yet again:

- **Reasoning.** Newer models are trained to write out their working before they answer, the way you might on scratch paper: try an approach, check it, back up, try another. That working is just more tokens, generated before the ones you see. It costs time and money, and it makes the models much better at math, logic, and code. When you use these models from your own programs, you'll get to choose how much reasoning effort to spend.
- **Tools.** A model can only produce text. But the program running the model can watch for text in a special format that means "search the web for this" or "run this code", go and do it, and paste the result back into the context. The model then carries on with new information in front of it. A model working in a loop with tools like these is called an **agent**, and this is how AI products look things up, edit files, and test their own work.

You're probably asking, where do the probabilities come from? How can GPT predict which word is most likely to follow an arbitrary chunk of text?

Let's delve into some theory for a bit. Prepare yourself for some math.

## How AI Models Work

### What is a "Model"

There are many common definitions:
- A person who wears and displays clothing — a fashion model.
- An example or standard — a role model.
- A prototype, design, or variant — the model of a car or phone.
- A simplified representation of a thing — a scale model of a building.

In science and statistics, a "model" is a mathematical formula that you can plug inputs into, and it transforms them into an output. The outputs are predictions that the model is making about the world: "if x happens, then y will happen".

### Example: Ice cream stand
Imagine we run an ice cream stand. We want to predict how much ice cream to purchase in advance, to maximize sales but minimize waste. How can we make the best possible prediction?

One approach would be to diligently log temperature versus sales for some period of time:

| Temperature (F) | Sales |
| --------------- | ----- |
| 73              | 361   |
| 83              | 477   |
| 76              | 422   |
| 73              | 366   |
| 65              | 383   |

Then we could make a plot of our data:

![A scatter plot of five observations of temperature against ice cream sales](/assets/ice-cream-plot-1.png)

It's not too useful yet. But let's keep diligently collecting observations:

![The same scatter plot with many more observations added](/assets/ice-cream-plot-2.png)

And eventually, a pattern starts to emerge:

![The scatter plot with enough observations that an upward trend is visible](/assets/ice-cream-plot-3.png)

Let's say that next week the forecast is 50 degrees. I want to predict how much ice cream will be consumed, so I can buy the right amount. I could just sorta eyeball it, but to make more systematic predictions...

I could draw a line that approximates the data:

![The scatter plot with a straight line drawn through the middle of the points](/assets/ice-cream-plot-4.png)

Recall that we can describe a line with an equation of the form:

`y = mx + b`

Where `m` is the slope of the line and `b` is where the line crosses the y axis (intercept). You can play around with [this interactive visualization](https://www.desmos.com/calculator/p5tqihi9fq) to see how changing `m` and `b` affect the line.

This line that I drew:

![The scatter plot with my eyeballed line drawn through it](/assets/ice-cream-plot-5.png)

Is described by:

`y = 7x - 140`

Where y is ice cream sales and x is degrees Fahrenheit. This formula is a *model* of the ice cream/temperature phenomenon. Once I've determined the m and b that describe the line I drew through my data, I can plug in any temperature for x and predict how many ice cream sales.

`m` (7 in this case) and `b` (-140 in this case) are the **parameters** of this model, so this model has 2 of them. They are also called the model's **weights**, and we'll use the two words interchangeably from here on. Notice that `x` is not a parameter: `x` is the input we plug in, and it changes every time we make a prediction. The parameters are the numbers we had to go *find*, and once found they stay put.

But, someone else might eyeball the line differently:

![The same scatter plot with a second, differently sloped line drawn through it](/assets/ice-cream-plot-6.png)

Is there a way to decide, objectively, which line is a better model?

One approach is to measure the "error" of each prediction vs the data:

![The scatter plot with vertical segments marking the distance from each point to the line](/assets/ice-cream-plot-7.png)

If we add up all the errors, we can come up with a measure of "fit" (or lack of fit, to be precise). Another common approach is to square the errors before adding them up. This approach gives extreme outliers more influence over fit. [Let's play around with this interactive visualization.](https://www.geogebra.org/m/XUkhCJRj) Try checking the first three checkboxes and then playing around with the sliders for `m` and `b` until you find the best possible line. Then you can see how close you got to the "ideal" line by checking the last box.

This is a method called "least squares". The way I found the best fit line was:
1. Move the intercept a little; then play around with the slope until the sum of squares started going up; then reverse direction until the sum of squares slowed down and stopped.
2. Then I'd rinse and repeat by moving the intercept, play around with the slope, and see if I could beat my previous score.
3. I iteratively arrived at the best line by observing the rate of change of the fit as I nudged each parameter.

In this approach, the sum of the squares is called the **loss function**: it represents how far away from the data the model is. We played around by trial and error until we minimized the loss. If we run the least squares method for the ice cream stand data, we get:

`y = 6x + 60`

It's somewhere between the two eyeballed lines:

![The scatter plot with the least squares line falling between the two eyeballed lines](/assets/ice-cream-plot-8.png)

So far we've:
- collected some data: input/output pairs.
- generalized a model (above) into which we can plug any input to predict a corresponding output.

Let's use it. Plug a temperature of 0 degrees into `y = 6x + 60`.

- How many sales does our model predict when the temperature is 0 degrees?
- 60
  - Correct! `6 * 0 + 60 = 60`.
- any
  - Not quite. Replace `x` with 0 in `y = 6x + 60` and work out `y`.
{: .free_text_number #zero_degrees title="Predicting at 0 degrees" points="1" answer="1" }

How about -20 degrees? The model says `6 * -20 + 60 = -60` sales. Negative ice cream. The formula will hand back an answer for any input we give it, whether or not that answer means anything.

#### Hallucinations

Predictions made outside of the range of the training data are basically useless, or worse, deceptive. In the new world of generative AI, these are referred to as "hallucinations".

So we should keep collecting more data if we want to make predictions across a wider range of inputs.

![The scatter plot extended with observations across a much wider temperature range](/assets/ice-cream-plot-9.png)

Our old model isn't a very good fit anymore:

![The old straight line drawn over the wider dataset, missing most of the points](/assets/ice-cream-plot-10.png)

And "hallucinates" at 125 degrees:

![The old line predicting implausibly high sales at 125 degrees](/assets/ice-cream-plot-11.png)

We could run least squares on the new, larger dataset, but a line is no longer a very good fit:

![A new best fit straight line over the wider dataset, still fitting the curve poorly](/assets/ice-cream-plot-12.png)

We might need a new architecture for our model. [Let's try a parabola](https://www.desmos.com/calculator/3ltwect0mc):

![A parabola fitted to the wider dataset, following its curve much more closely](/assets/ice-cream-plot-13.png)

Better. But [a third degree polynomial can be made to fit even better](https://www.desmos.com/calculator/wqufjhhj8y):

![A third degree polynomial fitted to the wider dataset, hugging the points closely](/assets/ice-cream-plot-14.png)

This model is something like:

$$
\Large
y = -0.001x^3 + x^2 - 52x + 1200
$$

The additional terms of the third-degree polynomial (the new architecture) gave us more flexibility to fit the data. But notice what happened to the parameters: the straight line had 2 of them, and this model has 4 (-0.001, 1, -52, and 1200). The search for the right ones got a lot harder.

Linear regression is a 200 year old form of "machine learning". "Machine learning" is where we try to develop a solution by collecting data and **inferring** a pattern, rather than deducing the fundamental theory underlying the phenomenon.

### Deduction Example: Momentum

`p = mv`

`p` is momentum, `m` is mass, and `v` is velocity. If we know the mass and velocity of an object, this model allows us to predict its momentum (by multiplying them together). This is the definition of momentum in Newtonian mechanics, and nobody arrived at it by collecting a pile of measurements and fitting a line to them. Newton developed a theory of how the underlying mechanism worked and *deduced* the formula from it. By contrast, our formula for the temperature/ice cream relationship was *inferred*: we never worked out why warm weather sells ice cream, we just noticed in the data that it does.

That is the dividing line. Deduction gives you a formula you can explain. Inference gives you a formula that works, and machine learning is what we call it when we settle for the second kind.

### Takeaways from this exercise

#### 1. More data improves the model
We improve the fit of the model through trial and error: by measuring loss, adjusting weights, measuring again, and so on, until we get close to a minimum. The more times we iterate, the closer we get to the "true" minimum.

Here is what more data buys you. Each column below trains the same kind of model on ten times more examples than the column to its left. The orange curve is the loss, dropping as the training set grows, and the blue line underneath is the shape the model has learned. With 10,000 examples it has barely found a straight line; with 10,000,000 it has recovered the actual pattern hiding in the data (the dotted shape).

![Four columns showing loss falling and the learned shape sharpening as the training set grows from ten thousand to ten million examples](/assets/overfitting-1.png)

And here is the search itself. The two axes are two of the model's weights, and the colors are the loss for each combination of them: blue valleys are a good fit, red peaks are a bad one. The red arrows are the path taken while nudging the weights downhill, which is exactly what we were doing by hand with the sliders. With 2 weights we can draw this picture. Real models have billions, and nobody can draw that.

![A loss surface plotted over two weights, with red arrows tracing paths downhill into the blue valleys](/assets/overfitting-2.png)

#### 2. The predictions of the model are only useful within the range of the data it was trained on
Outside, it "hallucinates". More parameters give us more flexibility to fit the data. More parameters without more data can lead to **overfitting**: with enough flexibility, the model stops learning the general pattern and starts memorizing the particular examples it was trained on, wiggling through every last point. It scores beautifully on the data it has already seen and then falls apart on anything new, which is the opposite of what we wanted. More data and more parameters also increase the difficulty of iterating to minimize loss.

#### 3. We don't know what the meaning of the parameters is, just that they work
When someone "releases a model", that means releasing both the architecture and the weights.
To fully open source it, you would release the method you used to get there: the training data, the loss function you chose, and so on.

### Neural Networks

Linear regression only works well for input/output data that is numerical. More generalized machine learning techniques have been developed that work for other types of data, for example images, sound clips, and text. Two milestones are worth knowing by name:

- **The Perceptron**, built by Frank Rosenblatt in 1958 and roughly modeled on a single neuron. It is the ancestor of everything below.
- **The Transformer**, introduced in 2017, which is the architecture behind GPT. It works on text by first splitting the text into tokens and converting each token into numbers, because a model can only ever do arithmetic.

Play with one in the browser: [A Neural Network Playground](https://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=spiral&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2,2&seed=0.16614&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=true&xSquared=true&ySquared=true&cosX=false&sinX=true&cosY=false&sinY=true&collectStats=false&problem=classification&initZero=false&hideText=false). It describes itself this way:

> "It’s a technique for building a computer program that learns from data. It is based very loosely on how we think the human brain works. First, a collection of software 'neurons' are created and connected together, allowing them to send messages to each other. Next, the network is asked to solve a problem, which it attempts to do over and over, each time strengthening the connections that lead to success and diminishing those that lead to failure."
>
> — [A Neural Network Playground](https://playground.tensorflow.org/)

"Strengthening the connections that lead to success and diminishing those that lead to failure" is the same loop we ran by hand on the ice cream data: make a prediction, measure the loss, nudge the weights, repeat. Theoretically, neural networks can learn any function, unlike linear regression.

### Back to where we started

We can now answer the question we left hanging at the start: where do those probabilities come from?

GPT is a model in exactly the sense we've been using the word. Its input is a chunk of text converted to tokens, and its output is a probability for every token that could come next. Nobody sat down and wrote rules for which word follows which. The weights were found by iterating over an enormous amount of written text, making a prediction, measuring the loss, and nudging, billions of times over.

So the differences from our ice cream line are differences of scale and architecture, not of kind. Chat, reasoning, and tools are all built on top of that one trained guess at the next token.

And our three takeaways survive the jump. More data still helps. Nobody can tell you what any individual weight means. And a model asked about something far outside its training data still hallucinates, only now it does so in fluent prose. Asking about a Ruby gem released after the model's training ended is the 125 degree day: you will get a confident answer either way. That is also why tools and context matter so much. A web search, or a page of documentation pasted into the prompt, moves the question back inside the range of what the model has in front of it.

## Quiz

- How does a model like GPT produce a whole paragraph of text?
- It writes the full paragraph in one step, then checks it for errors.
  - Not quite. There is no separate drafting or checking step. Every piece of the output is produced the same way.
- It produces one token at a time, feeding its own output back in as part of the next input.
  - Correct! A paragraph is that loop run a few hundred times.
- It searches its training data for a matching paragraph and copies it.
  - Not quite. The model does not store or look up documents. It computes a probability for each possible next token.
{: .choose_best #token_loop title="Producing a Paragraph" points="1" answer="2" }

- You ask a chatbot the same question in two new chats and get two differently worded answers. Why?
- The model learned from your first chat and improved its answer.
  - Not quite. The model's weights do not change while you use it, and a new chat starts from nothing.
- One of the two answers must be a hallucination.
  - Not quite. Differently worded answers are normal, and both can be right.
- The next token is chosen from the probabilities with some randomness, and each choice affects everything after it.
  - Correct! A slightly different token early on sends the rest of the output down a different path.
{: .choose_best #same_prompt_twice title="Same Prompt, Different Answers" points="1" answer="3" }

- What does the equation `y = mx + b` represent in AI modeling?
- The formula for predicting outcomes in a linear regression model.
  - Correct! This equation describes the relationship between inputs and outputs in linear regression.
- A general formula for all types of machine learning models.
  - Not quite. This equation is specific to linear regression.
- The training process for a neural network.
  - Incorrect. Neural networks use a different process.
{: .choose_best #linear_regression title="Linear Regression Equation" points="1" answer="1" }

- Why is it important to limit predictions to the range of training data?
- Models always perform better with extreme data values.
  - Not quite. Extreme values can lead to poor or incorrect predictions.
- Predictions outside the training range may be inaccurate or misleading.
  - Correct! This limitation is due to a lack of data to guide the model.
- To reduce computation time during prediction.
  - Incorrect. Limiting the range isn't related to computation time.
{: .choose_best #training_range title="Limits of Training Data Range" points="1" answer="2" }

- What happens when a model is overfitted?
- It performs consistently well on all datasets.
  - Not quite. Overfitting impacts performance on unseen data.
- It becomes more accurate for predictions outside its training range.
  - Incorrect. Overfitting does not improve performance on new or extreme inputs.
- It memorizes training data but struggles with new data.
  - Correct! Overfitting reduces the model's ability to generalize.
{: .choose_best #overfitting title="Understanding Overfitting" points="1" answer="3" }

## Conclusion

Artificial intelligence represents a transformative leap in how software interacts with and learns from the world. This lesson introduced the foundational concepts: what a model is, how one is found rather than designed, and what that process explains about the way tools like GPT behave. With a basic intuition of how AI models work, you are equipped to experiment, innovate, and integrate these powerful technologies into your projects.

---

## Time Taken
- Approximately how long (in minutes) did this lesson take you to complete?
{: .free_text_number #time_taken title="Time taken" points="1" answer="any" }


