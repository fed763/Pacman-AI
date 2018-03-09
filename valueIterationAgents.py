# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
	"""
		* Please read learningAgents.py before reading this.*

		A ValueIterationAgent takes a Markov decision process
		(see mdp.py) on initialization and runs value iteration
		for a given number of iterations using the supplied
		discount factor.
	"""
	def __init__(self, mdp, discount = 0.9, iterations = 100):
		"""
		  Your value iteration agent should take an mdp on
		  construction, run the indicated number of iterations
		  and then act according to the resulting policy.

		  Some useful mdp methods you will use:
			  mdp.getStates()
			  mdp.getPossibleActions(state)
			  mdp.getTransitionStatesAndProbs(state, action)
			  mdp.getReward(state, action, nextState)
			  mdp.isTerminal(state)
		"""
		self.mdp = mdp
		self.discount = discount
		self.iterations = iterations
		self.values = util.Counter() # A Counter is a dict with default 0
		self.newValues = util.Counter()

		# Write value iteration code here
		"""
		while still iterating
			for each state
				for each action
					set new qvalue for state,action pair
					if this qvalue is better
						new best value is this qvalue
			new values
		"""
		while(self.iterations != 0):
			for state in self.mdp.getStates():
				bestQValue = -10000.0
				for action in self.mdp.getPossibleActions(state):
					self.newValues[(state,action)] = self.getQValue(state,action)
					currentActionQValue = self.newValues[(state,action)]
					if(currentActionQValue > bestQValue):
						bestQValue = currentActionQValue
				self.newValues[state] = bestQValue

			for values in self.newValues:
				self.values[values] = self.newValues[values]
			self.iterations = self.iterations - 1
	def getValue(self, state):
		"""
		  Return the value of the state (computed in __init__).
		"""
		return self.values[state]

	def computeQValueFromValues(self, state, action):
		"""
		  Compute the Q-value of action in state from the
		  value function stored in self.values.
		"""
		"""
		for each landing state from performing action a in state s:
			if the landing state is terminal:
				its value is zero
			else 
				look up landing states value in counter
			qValue = previousqValue + (probability*(transitionReward + (discount*landingStateValue)))
			return The sum of all qValues for (s,a,s')
		"""
		qValue = 0.0
		for landingStateProb in self.mdp.getTransitionStatesAndProbs(state,action):
			landingState = landingStateProb[0]
			if self.mdp.isTerminal(landingState):
				landingStateValue = 0.0
			else:
				landingStateValue = self.getValue(landingState)
			probability = landingStateProb[1]
			transitionReward = self.mdp.getReward(state,action,landingState)
			qValue = qValue + (probability*(transitionReward + (self.discount*landingStateValue)))

		return qValue

	def computeActionFromValues(self, state):
		"""
		  The policy is the best action in the given state
		  according to the values currently stored in self.values.

		  You may break ties any way you see fit.  Note that if
		  there are no legal actions, which is the case at the
		  terminal state, you should return None.
		"""
		"*** YOUR CODE HERE ***"

		if (self.mdp.isTerminal(state)):
			return None
		else:
			bestQValue = self.getValue(state)
			for action in self.mdp.getPossibleActions(state):
				stateActionQValue = self.values[(state,action)]
				if(bestQValue == stateActionQValue):
					return action

	def getPolicy(self, state):
		return self.computeActionFromValues(state)

	def getAction(self, state):
		"Returns the policy at the state (no exploration)."
		return self.computeActionFromValues(state)

	def getQValue(self, state, action):
		return self.computeQValueFromValues(state, action)
