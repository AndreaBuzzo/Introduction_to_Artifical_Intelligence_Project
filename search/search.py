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
from game import Actions
import heapq

class Node:
    def __init__(self, state, actions, cost, heuristic):
        self.state = state
        self.actions = actions
        self.cost = cost
        self.heuristic = heuristic

    def __lt__(self, other):
        return (self.cost + self.heuristic) < (other.cost + other.heuristic)

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """
    def __init__(self, startingPosition, corners, walls):
        """
        Initializes the CornersProblem with the given starting position, corner
        coordinates, and wall positions.
        """
        self.startingPosition = startingPosition
        self.corners = corners
        self.walls = walls

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        return (self.startingPosition, tuple(self.corners))

    def isGoalState(self, state):
        """
        Returns True if and only if the state is a valid goal state.
        """
        return not state[1]  # Check if there are no remaining corners

    def getSuccessors(self, state):
        """
        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        successors = []
        pacman_position, remaining_corners = state

        for action in ['North', 'South', 'East', 'West']:
            x, y = pacman_position
            dx, dy = Actions.directionToVector(action)
            next_x, next_y = int(x + dx), int(y + dy)

            if not self.walls[next_x][next_y]:
                next_position = (next_x, next_y)
                next_corners = tuple([c for c in remaining_corners if c != next_position])
                successors.append(((next_position, next_corners), action, 1))  # Assuming constant cost

        return successors

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        return len(actions)


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
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
    visited = set()  # To keep track of visited states
    stack = util.Stack()  # Stack for DFS

    # Push the start state onto the stack with an empty list of actions
    stack.push((problem.getStartState(), []))

    while not stack.isEmpty():
        state, actions = stack.pop()

        if problem.isGoalState(state):
            return actions  # Return the list of actions if the goal state is reached

        if state not in visited:
            visited.add(state)

            # Get successors and push them onto the stack with updated actions
            for successor, action, _ in problem.getSuccessors(state):
                new_actions = actions + [action]
                stack.push((successor, new_actions))

    # If no solution is found
    return []


def breadthFirstSearch(problem: SearchProblem):
    """
    Search the shallowest nodes in the search tree first.
    """
    visited = set()  # To keep track of visited states
    queue = util.Queue()  # Queue for BFS

    # Enqueue the start state onto the queue with an empty list of actions
    queue.push((problem.getStartState(), []))

    while not queue.isEmpty():
        state, actions = queue.pop()

        if problem.isGoalState(state):
            return actions  # Return the list of actions if the goal state is reached

        if state not in visited:
            visited.add(state)

            # Get successors and enqueue them onto the queue with updated actions
            for successor, action, _ in problem.getSuccessors(state):
                new_actions = actions + [action]
                queue.push((successor, new_actions))

    # If no solution is found
    return []


def uniformCostSearch(problem: SearchProblem):
    """
    Search the node of least total cost first.
    """
    visited = set()  # To keep track of visited states
    priority_queue = util.PriorityQueue()  # PriorityQueue for UCS

    # Enqueue the start state onto the priority queue with an initial cost of 0 and an empty list of actions
    priority_queue.push((problem.getStartState(), [], 0), 0)

    while not priority_queue.isEmpty():
        state, actions, cost = priority_queue.pop()

        if problem.isGoalState(state):
            return actions  # Return the list of actions if the goal state is reached

        if state not in visited:
            visited.add(state)

            # Get successors and enqueue them onto the priority queue with updated actions and cost
            for successor, action, step_cost in problem.getSuccessors(state):
                new_actions = actions + [action]
                new_cost = cost + step_cost
                priority_queue.push((successor, new_actions, new_cost), new_cost)

    # If no solution is found
    return []


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    start_state = problem.getStartState()
    start_node = Node(state=start_state, actions=[], cost=0, heuristic=heuristic(start_state, problem))

    open_set = [start_node]
    closed_set = set()

    while open_set:
        current_node = heapq.heappop(open_set)

        if problem.isGoalState(current_node.state):
            return current_node.actions

        state_key = tuple(current_node.state)  # Convert set to tuple
        if state_key not in closed_set:
            closed_set.add(state_key)
            successors = problem.getSuccessors(current_node.state)

            for successor, action, step_cost in successors:
                new_actions = current_node.actions + [action]
                new_cost = current_node.cost + step_cost
                new_heuristic = heuristic(successor, problem)
                new_node = Node(state=successor, actions=new_actions, cost=new_cost, heuristic=new_heuristic)

                heapq.heappush(open_set, new_node)

    return None


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
