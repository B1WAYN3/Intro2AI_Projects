# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    
    # Initialize the data structure for this algorithm. 
    frontier = util.Stack()

    # pushing the start state, initializing the search. 
    frontier.push((problem.getStartState(), [], 0))

    # To keep track of the already expanded nodes. 
    expanded_dataset = set() 

    while not frontier.isEmpty():
        state, actions_taken, cost_total =  frontier.pop()

        if(problem.isGoalState(state)):
            return actions_taken
        if state not in expanded_dataset:
            expanded_dataset.add(state)

            # Search the children of the current node. 
            for succ_state, action, succ_Cost in problem.getSuccessors(state):
                new_actions = actions_taken + [action]    # extend path
                new_cost = cost_total + succ_Cost          # accumulate cost of the whole path from the start
                frontier.push((succ_state, new_actions, new_cost))

    # If we get here, no solution was found unforch 
    return []

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # Identical to DFS, except with a different data structure
    # Initialize the data structure for this algorithm. 
    frontier = util.Queue()

    # pushing the start state, initializing the search. 
    frontier.push((problem.getStartState(), [], 0))

    # To keep track of the already expanded nodes. 
    expanded_dataset = set() 

    while not frontier.isEmpty():
        state, actions_taken, cost_total =  frontier.pop()

        if(problem.isGoalState(state)):
            return actions_taken
        if state not in expanded_dataset:
            expanded_dataset.add(state)

            # Search the children of the current node. 
            for succ_state, action, succ_Cost in problem.getSuccessors(state):
                new_actions = actions_taken + [action]    # extend path
                new_cost = cost_total + succ_Cost          # accumulate cost of the whole path from the start
                frontier.push((succ_state, new_actions, new_cost))

    # If we get here, no solution was found unforch 
    return []

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    frontier = util.PriorityQueue()

    # pushing the start state, initializing the search. Set priority 0. 
    frontier.push((problem.getStartState(), [], 0), priority=0)

    # For maping the cheapest path (MAPPING STATES) cost found so far. 
    # hashtag we love our dictionaries bruv 
    # best_g_cost: dict[state, g_value]
    best_g_cost = {}

    while not frontier.isEmpty():
        state, actions_taken, current_g_factor =  frontier.pop()

        if(problem.isGoalState(state)):
            return actions_taken
        
        # Avoid states when they are more expensive.
        if state in best_g_cost and  current_g_factor > best_g_cost[state]: 
            continue

        # add the state to the mapping bruv 
        best_g_cost[state] = current_g_factor 

        
        # Search the children of the current node. 
        for succ_state, succ_action, succ_Cost in problem.getSuccessors(state):
            new_actions = actions_taken + [succ_action]    # extend path
            new_g_spot = current_g_factor + succ_Cost          # accumulate cost of the whole path from the start
            frontier.update((succ_state, new_actions, new_g_spot), new_g_spot)

    # Lets goooo
    # PS C:\Users\Wayne\Desktop\FALL 2025\AI\Project_Repo\Intro2AI_Projects\project1-search> python pacman.py -l mediumMaze -p SearchAgent -a fn=ucs
    # [SearchAgent] using function ucs
    # [SearchAgent] using problem type PositionSearchProblem
    # Path found with total cost of 68 in 0.0 seconds
    # Search nodes expanded: 275
    # Pacman emerges victorious! Score: 442
    # Average Score: 442.0
    # Scores:        442.0
    # Win Rate:      1/1 (1.00)
    # Record:        Win

    
    # If we get here, no solution was found unforch.
    # If you're reading this, it's too late -- Drake. 
    return []



def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    """Oh fucc u have to implement the nullHeuristic function lmfao before running this"""
    frontier = util.PriorityQueue()

    # pushing the start state, initializing the search. 
    frontier.push((problem.getStartState(), [], 0), heuristic(problem.getStartState(), problem))

    # For maping the cheapest path (MAPPING STATES) cost found so far. 
    # hashtag we love our dictionaries bruv 
    # best_g_cost: dict[state, g_value]
    best_g_cost = {}

    while not frontier.isEmpty():
        state, actions_taken, current_g_factor =  frontier.pop()

        if(problem.isGoalState(state)):
            return actions_taken
        
        # Avoid states when they are more expensive.
        if state in best_g_cost and  current_g_factor > best_g_cost[state]: 
            continue

        # add the state to the mapping bruv 
        best_g_cost[state] = current_g_factor 

        
        # Search the children of the current node. 
        for succ_state, succ_action, succ_Cost in problem.getSuccessors(state):
            new_actions = actions_taken + [succ_action]    # extend path
            new_g_spot = current_g_factor + succ_Cost          # accumulate cost of the whole path from the start
            new_f_spot = new_g_spot + heuristic(succ_state, problem)
            frontier.update((succ_state, new_actions, new_g_spot), new_f_spot) # update the data structure

    # If we get here, no solution was found unforch.
    # If you're reading this, it's too late -- Drake. 
    return []

    

    



# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
