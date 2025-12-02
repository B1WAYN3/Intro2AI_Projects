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
import collections

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
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        for i in range(self.iterations):
            # Use batch version: compute all new values from the old values
            # Create a new Counter to store the updated values
            newValues = util.Counter()
            
            # Update value for each state
            for state in self.mdp.getStates():
                # Terminal states have value 0
                if self.mdp.isTerminal(state):
                    newValues[state] = 0
                else:
                    # Find the maximum Q-value over all actions
                    possibleActions = self.mdp.getPossibleActions(state)
                    if not possibleActions:
                        # No actions available
                        newValues[state] = 0
                    else:
                        # Compute Q-value for each action and take the max
                        maxQValue = max(self.computeQValueFromValues(state, action) 
                                      for action in possibleActions)
                        newValues[state] = maxQValue
            
            # Update self.values with the new values (batch update)
            self.values = newValues


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
        # Q(s,a) = Σ_{s'} T(s,a,s')[R(s,a,s') + γV(s')]
        qValue = 0  # FIXED: Initialize qValue
        for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):  # FIXED: self.mdp
            reward = self.mdp.getReward(state, action, nextState)  # FIXED: self.mdp
            qValue += prob * (reward + self.discount * self.values[nextState])  # FIXED: self.discount
        return qValue

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        possibleActions = self.mdp.getPossibleActions(state)  # FIXED: self.mdp
        
        # If no actions available (terminal state), return None
        if not possibleActions:
            return None

        # Find the action with the maximum Q-value
        actionQValues = util.Counter()
        for action in possibleActions:
            actionQValues[action] = self.computeQValueFromValues(state, action)

        return actionQValues.argMax()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        # Cyclic value iteration: update one state per iteration
        states = self.mdp.getStates()
        
        for i in range(self.iterations):
            # Determine which state to update (cycle through states)
            state = states[i % len(states)]
            
            # If the state is terminal, skip it (nothing to update)
            if self.mdp.isTerminal(state):
                continue
            
            # Get possible actions for this state
            possibleActions = self.mdp.getPossibleActions(state)
            
            # If no actions available, value remains 0
            if not possibleActions:
                self.values[state] = 0
            else:
                # Update the value of this state using the Bellman equation
                # V(s) = max_a Q(s,a)
                maxQValue = max(self.computeQValueFromValues(state, action) 
                              for action in possibleActions)
                self.values[state] = maxQValue

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        # Step 1: Compute predecessors of all states
        # predecessors[s] is a set of states that can reach state s
        predecessors = {}
        for state in self.mdp.getStates():
            predecessors[state] = set()
        
        # For each state, find all states that can transition to it
        for state in self.mdp.getStates():
            if self.mdp.isTerminal(state):
                continue
            for action in self.mdp.getPossibleActions(state):
                for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                    if prob > 0:
                        predecessors[nextState].add(state)
        
        # Step 2: Initialize priority queue
        pq = util.PriorityQueue()
        
        # Step 3: For each non-terminal state, compute diff and add to priority queue
        for state in self.mdp.getStates():
            if self.mdp.isTerminal(state):
                continue
            
            possibleActions = self.mdp.getPossibleActions(state)
            if not possibleActions:
                continue
            
            # Find the highest Q-value across all actions
            maxQValue = max(self.computeQValueFromValues(state, action) 
                          for action in possibleActions)
            
            # Compute diff: absolute difference between current value and max Q-value
            diff = abs(self.values[state] - maxQValue)
            
            # Push state into priority queue with priority -diff (negative for min heap)
            pq.push(state, -diff)
        
        # Step 4: Main loop - iterate for self.iterations
        for i in range(self.iterations):
            # If priority queue is empty, terminate
            if pq.isEmpty():
                break
            
            # Pop state with highest priority (lowest -diff, i.e., highest diff)
            state = pq.pop()
            
            # Update the value of state (if not terminal)
            if not self.mdp.isTerminal(state):
                possibleActions = self.mdp.getPossibleActions(state)
                if possibleActions:
                    maxQValue = max(self.computeQValueFromValues(state, action) 
                                  for action in possibleActions)
                    self.values[state] = maxQValue
            
            # For each predecessor of state
            for predecessor in predecessors[state]:
                if self.mdp.isTerminal(predecessor):
                    continue
                
                possibleActions = self.mdp.getPossibleActions(predecessor)
                if not possibleActions:
                    continue
                
                # Find the highest Q-value for the predecessor
                maxQValue = max(self.computeQValueFromValues(predecessor, action) 
                              for action in possibleActions)
                
                # Compute diff for predecessor
                diff = abs(self.values[predecessor] - maxQValue)
                
                # If diff > theta, push predecessor into priority queue
                if diff > self.theta:
                    pq.update(predecessor, -diff)

