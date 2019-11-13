# Applying Algorithm to a Real World Problem
To prevent the robot to do dangerous actions it would be necessary to collect and interpret sensor data. Before applying an selected action it would be necessary to check if the sensor data allow this action. For example:

```
...
def askSensorPatternsForAllowedAction(action):
	# call subroutines to analyse patterns of multiple sensors
	...
	if badPattern:
		return False
	else:
		return True 

def run(self, no_rounds, exploration_strategy, **strategy_parameters):
  	...
	action = exploration_strategy(self, **strategy_parameters)
	if askSensorPatternsForAllowedAction(action):
		reward = self.pull(action)
	else:
		return badRegret, badReward
	...
```

That way the next visit the robot knows to choose another action.