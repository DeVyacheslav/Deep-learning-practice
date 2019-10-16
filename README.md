# Challenge "Machine Learning in seconds"
Tasks:
  helloXor
  generalCpu
  findMe
  bb8
  votePrediction
  mnist
 
Location at "mlis/problems".
  
# helloXor
Hello Xor is a HelloWorld of Machine Learning.
In this problem you asked to design and learn neural network in such way, that it would be able to learn Xor function.
The function table for Xor function is:

    00 => 0
    01 => 1
    10 => 1
    11 => 0
Time limit: 2.0 seconds

# generalCpu
In General CPU problem you are asked to learn a random function with n [4-7] parameters.

Time limit: 2.0 seconds

# findMe
In Find Me problem you are asked to learn a random function with 8 inputs, but there are up to 32 random input added and you don’t know which input is good inputs and which is random inputs. Generated data splited in 2 equal parts, first one used for training, second one used for testing.

Time limit: 2.0 seconds

# bb8
There are 2 binary languages defined by states and probabilities of next states. You need to learn to determine where each language by a sentence.

NOTE: Make sure you solved HelloXor problem, it contains more instructions
Problem file: bb8.py
Data example:
Let’s say there only 2 states that define language:
First language defined by probability for next state based on current states and 0 or 1 during transition:
      A         B
  A 0.6500(1) 0.3500(1)
  B 0.0564(1) 0.9436(0)
In this language, if we will start in state A, then with probability 0.65 we will stay in state A and generate 1, then let’s say with probability 0.35 we will go to state B and generate 1, then with probability 0.9436 we will stay in state B for 6 moves and generate 000000
In result we will generate 1100000
Second language defined by:
       A        B
  A 0.3899(0) 0.6101(0)
  B 0.7771(1) 0.2229(1)

Here is example of generated sentences by this 2 languages:

  10011000=>0
  10100110=>1
  01010101=>1
  11100000=>0
  00000000=>0
  11010110=>1
  01110000=>0
  01010101=>1

Time limit: 2.0 seconds

# votePrediction
Your friend runs an evil government and he wants to influence elections in a foreign country. His evil spy network collected data about voters in a foreign country:

    1 if a voter older then 35 years, 0 otherwise
    1 if a voter male, 0 otherwise
    1 if a voter watched PythonNN in last month, 0 otherwise
    1 if a voter watched Rabbit News in last month, 0 otherwise
    1 if a voter lives in a big city, 0 otherwise
    1 if a voter voted last time, 0 otherwise
    1 if a voter likes ice cream, 0 otherwise
    1 if a voter has hair, 0 otherwise

An evil plan of your friend is following:

    Based on 8 features predict how a person will vote
    Model if watching Rabbit News influences voters to vote for needed option
    Go to national parks and feed the rabbits
    A population of rabbits will grow, more people will see them in park
    People who will see rabbits in a park will decide to watch Rabbit News

Your friend just notified you that they were able to collect information about voters, but they were not able to get information on how people voted before because that country employs secret vote system. They have information on how people voted in aggregate but not on voter level.
So, now it is your work to help your evil government and earn a hero status.
You are given data from previous elections. 8*(number of voters) features and binary result:

    1 if more then half voters voted in favor, 0 otherwise

Your task is based on this information to predict how people will vote on next election.
Data example:
Let’s say we have only 2 features instead of 8 and 3 voters.
Voters will vote based on following function:
  00=>1
  01=>0
  10=>1
  11=>0
Your train data will look like this:
  000111=>0 // 00+01+11=1+0+0=1 > 3/2=>false=>0
  111000=>1 // 11+10+00=0+1+1=2 > 3/2 =>true=>1
  011101=>0
  001011=>1
Your test data will look like this:
  001010
  110010
  011101
  110001
Time limit: 2.0 seconds

# mnist
MNIST is a HellowWorld of image classification. In this problem, you are given handwritten digit as 28x28 matrix and your task is to learn your model to classify what digit on the image

Time limit: 2.0 seconds

Machine Learning in second // Join us at https://www.facebook.com/groups/195065914629311/ 
