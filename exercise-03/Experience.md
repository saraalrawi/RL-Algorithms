# Experiences
## Monte Carlo and TD(λ)
We independently came to the same solution.
## First-visit MC Evaluation
While reading Reinforcement Learning: An Introduction - Chapter 5.1 we noticed the Pseudo-Code was named somethimes prediction and other times evaluation.

Also confusing:
When implementing exactly how it is written in the book it will not work! The correct Pseudo-Code would look like this:

```
Initialize:
	pi = policy to be evaluated
	V = an arbitrary state-value function 
	Returns(s) = an empty list, for all s in S
Repeat forever:
	Generate an episode using pi
	For each state s appearing in the episode:
		G = the return that follows the first occurrence of s 		
		Append G to Returns(s)
	For each state s in Returns:
		V(s) = average(Returns(s))
```
After that change the (new) tests pass and a nice plot shows what expected:
![BackJack](StepsBackjack.png)

## Mac OS vs. Linux
We use VirtualBox and Linux on MacOS because the path fix of the python tests doesn't work elsewhere.