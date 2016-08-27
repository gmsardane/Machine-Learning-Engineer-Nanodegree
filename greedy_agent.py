import random
import numpy
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
      
        # TODO: Initialize any additional variables here
       
        #state variables
        self.next_waypoint = None
        self.state_0 = None
        self.reward_0 = None
        self.action_0 = None
        
        waypoints = ['left', 'right', 'forward']
        lights = ['red', 'green']
              
        #Qtable parameters
        self.alpha = 0.75    #Learning Rate
        self.gamma = 0.25    #Discounted Reward Factor   
        self.epsilon = 0.80  #exploitation-exploration
        
        #Create the Q-table and initialize all to zero:
        self.Qdict = {} #dictionary of states and and possible action, value pairs. Keys are (states), action
        waypoints = ['left', 'right', 'forward']
        lights = ['red', 'green']
        for waypoint in waypoints:
        	for light in lights:
        		for left in Environment.valid_actions:
        			for right in Environment.valid_actions:
        				for oncoming in Environment.valid_actions:
        					for act in Environment.valid_actions:
        						self.Qdict[(waypoint,light, left, right, oncoming), act] = 0. 
        			
        #Counters:
        self.ndeadline = []
        self.tlist = []
    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state      
        self.state = (self.next_waypoint, inputs['light'], inputs['left'], inputs['oncoming'],  inputs['right'])#, deadline)
        current_Q = numpy.finfo(numpy.float32).min

        # TODO: Select action according to your policy        
        #If all actions have the same Q-vals, choose a random action amongst those that have equal Q's:         	   
        tmp = []
        for act in Environment.valid_actions:
        	tmp.append(self.Qdict[(self.state), act])
        max_qval = max(list(set(tmp)))
        max_ind = numpy.array(numpy.where(numpy.array(tmp) == max_qval)[0])
        if len(tmp) != len(list(set(tmp))):
        	action = Environment.valid_actions[random.choice(max_ind)]
        else:
        	p = numpy.zeros(4) + (1. - self.epsilon)/3.
        	p[max_ind] = self.epsilon
        	ind = numpy.random.choice(numpy.arange(0, 4), p=p)
        	action = Environment.valid_actions[ind]
        if self.Qdict[(self.state), action] > current_Q:
        	current_Q = self.Qdict[(self.state), action]
        	reward = self.env.act(self, action)
        
         
        # TODO: Learn policy based on state, action, reward
        if (self.state_0, self.action_0, self.reward_0) != (None, None, None):
        	prev_Q = self.Qdict[(self.state_0), self.action_0] 
        	#Update the Q(state, action) after getting the reward
        	self.Qdict[(self.state_0), self.action_0] = (1. - self.alpha)*prev_Q + \
        	self.alpha*(self.reward_0 + self.gamma*current_Q)
        #Updated state, action, reward
        (self.state_0, self.action_0, self.reward_0) = (self.state, action, reward)
        #print("LearningAgent.update(): deadline = {0}, inputs = {1}, action = {2}, reward = {3}".format(deadline, inputs, action, reward))  # [debug]
        
        if reward >= 10.0:
        	print('Successfully completed within deadline = {0} vs {1} '.format(deadline, t)) 
        	self.ndeadline.append(deadline)
        	self.tlist.append(t)
        	
    def show_result(self):
    	numpy.savetxt('greedy_out.txt', numpy.transpose([numpy.array(self.ndeadline), \
    	numpy.array(self.tlist)]),  fmt='%2.1f', header = 'ndeadline, t', delimiter=',')

def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.01, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False
    
    sim.run(n_trials=500)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
    a.show_result()

if __name__ == '__main__':
    run()
    
    
    
    
    
