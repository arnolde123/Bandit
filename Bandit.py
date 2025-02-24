import sys
import csv
import networkx as nx
import random

#Creates a bandit that is able to select which arm is selected next 
#based off of probabilities formed from normalized weights. The algorithm
#changes how weights are updated, which affects which arm is selected next.
class Bandit:
    #Initializes all variables for each bandit including some constants
    #The constants come from the variables from the formulas found in the lectures
    #Can change weight and exploration_rate for different results
    #recency sigma length is how far back we look in the reward history
    def __init__(self, n_arms, alg, reward_type, decay_param):
        self.n_arms = n_arms
        self.alg = alg
        self.reward_type = reward_type
        self.decay_param = decay_param
        self.pulls = [0] * n_arms
        self.w_values = [0] * n_arms
        self.uniform_dist_param = 1 / n_arms
        self.exploration_rate = 0.1
        self.reward_weight = 0.2
        self.reward_history = [[] for _ in range(n_arms)]
        self.recency_sigma_length = 10

    #Randomization for choosing arm
    def _get_random_number(self):
        return random.random()

    def choose_arm(self):
        # To stop divide by 0 errors
        total_weight = sum(self.w_values) or 1
        normalized_weights = [w / total_weight for w in self.w_values]

        # Uses formula from Lecture 7 page 3
        probabilities = [
            w * (1 - self.exploration_rate) + self.exploration_rate * self.uniform_dist_param
            for w in normalized_weights
        ]

        # Normalize probabilities
        total_prob = sum(probabilities) or 1  # Avoid divide by zero
        normalized_probabilities = [p / total_prob for p in probabilities]

        # Calculate cumulative probabilities
        cumulative_probabilities = []
        cumulative_sum = 0
        for prob in normalized_probabilities:
            cumulative_sum += prob
            cumulative_probabilities.append(cumulative_sum)

        # Generate a random number between 0 and 1 to determine which idx is returned
        rand_value = self._get_random_number()

        # Select arm based on cumulative probabilities
        for i, cum_prob in enumerate(cumulative_probabilities):
            if rand_value <= cum_prob:
                return i
        return len(cumulative_probabilities) - 1 

    def update(self, chosen_arm, reward):
        self.pulls[chosen_arm] += 1
        if self.alg == "Static":
            self._static_weight_update(chosen_arm, reward)
        elif self.alg == "Recency":
            self._recency_update(chosen_arm, reward)
    #Update weight using static and recency algorithms found in Lecture
    def _static_weight_update(self, chosen_arm, reward):
        self.w_values[chosen_arm] = self.decay_param * self.w_values[chosen_arm] + self.reward_weight * reward

    def _recency_weight_update(self, chosen_arm, reward):
        self.reward_history[chosen_arm].append(reward)
        if len(self.reward_history[chosen_arm]) > self.recency_sigma_length:
            self.reward_history[chosen_arm].pop(0)
        self.w_values[chosen_arm] = sum(
            self.decay_param ** j * r for j, r in enumerate(reversed(self.reward_history[chosen_arm]))
        )
#Loads both files and create a dict to hold engagements and user, and also a graph
#of the users
def load_data(content_engagement_file, user_network_file):
    engagements = {}
    graph = nx.DiGraph()

    with open(content_engagement_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            content_id = row['content_id']
            for user_id, engagement in row.items():
                if user_id != 'content_id':
                    if user_id not in engagements:
                        engagements[user_id] = {}
                    engagements[user_id][content_id] = int(engagement)

    with open(user_network_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            user = row[0]
            followers = row[1].strip('"').split(',')
            for follower in followers:
                graph.add_edge(user, follower)

    return engagements, graph

#Calculates the reward based on the given params. This will be used to add to the cumulative
#reward. If the reward type is individual, simply return the engagement/preference
#score within the file. If the type is influence, we include the engagement score of 
#any followers the user has
def calculate_reward(user, content_id, network, engagements, reward_type):
    if reward_type == "Individual":
        return engagements[user].get(content_id, 0)
    elif reward_type == "Influence":
        followers = list(network.successors(user))
        #To calculate reward for influence, we multiply each engagement score of
        #every follower the user has by 0.5, and then sum them
        follower_engagements = sum(
            engagements[follower].get(content_id, 0) * 0.5 
            for follower in followers if follower in engagements
        )
        return engagements[user].get(content_id, 0) + 0.5 * follower_engagements

def grid_search(engagements, network, algorithm, reward_type, range_value, min_param, step, max_param):
    #Make an empty dict to store user_ids and the corresponding best param
    best_params = {}

    content_ids = list(next(iter(engagements.values())).keys())

    # Iterates over the content and finds the best parameter for each user
    for user in engagements:
        best_param = min_param
        best_reward = float('-inf')
        current_param = min_param
        
        #Continues searching for a better parameter/reward
        #If a better reward is found, best_reward and param are updated
        #The while loop continually increments the current_param until we
        #have stepped through the entire range
        while current_param <= max_param:
            bandit = Bandit(len(content_ids), algorithm, reward_type, current_param)
            total_reward = 0
            
            for _ in range(range_value):
                chosen_arm = bandit.choose_arm()
                reward = calculate_reward(user, content_ids[chosen_arm], network, engagements, reward_type)
                bandit.update(chosen_arm, reward)
                total_reward += reward

            if total_reward > best_reward:
                best_reward = total_reward
                best_param = current_param

            current_param += step

        best_params[user] = best_param

    return best_params

def run_bandit(engagements, network, algorithm, reward_type, range_value, min_param, step, max_param):
    #Print best params found in grid search
    best_params = grid_search(engagements, network, algorithm, reward_type, range_value, min_param, step, max_param)
    print(f"Initial parameters for {algorithm} algorithm with {reward_type} reward type")
    for user, param in best_params.items():
        print(f"User {user}: best_param={param}")

    #Stores bandits and rewards for each user
    bandits = {user: Bandit(len(engagements[user]), algorithm, reward_type, param) for user, param in best_params.items()}
    cumulative_rewards = {user: 0 for user in engagements}
    #List of all ids
    content_ids = list(next(iter(engagements.values())).keys())

    #Now goes through all engagements with the best param for each user and 
    #calculates the cumulative reward
    #Each step is printed with the user, the content selected, reward, and cumulative reward
    for content_idx, content_id in enumerate(content_ids):
        print(f"\nStep {content_idx + 1}, Content ID: {content_id}")
        for user, bandit in bandits.items():
            chosen_arm = bandit.choose_arm()
            chosen_content = content_ids[chosen_arm]
            reward = calculate_reward(user, chosen_content, network, engagements, reward_type)
            bandit.update(chosen_arm, reward)
            cumulative_rewards[user] += reward
            print(f"User {user}: Chosen content={chosen_content}, Reward={reward:.2f}, Cumulative Reward={cumulative_rewards[user]:.2f}")

    # Prints the overall relative number of pulls for each content arm
    print("\nOverall relative number of pulls:")
    total_pulls = [0] * len(content_ids)
    # Sum pulls across all users
    for bandit in bandits.values():
        for i in range(len(total_pulls)):
            total_pulls[i] += bandit.pulls[i]
    # Calculate relative pulls
    total_pulls_sum = sum(total_pulls) 
    relative_pulls = [count / total_pulls_sum for count in total_pulls]
    print(f"Overall relative pulls: {dict(zip(content_ids, relative_pulls))}")

    #Finds the user with the highest cumulative reward and prints
    best_user = max(cumulative_rewards, key=cumulative_rewards.get)
    print(f"\nBest performing user: {best_user} with cumulative reward: {cumulative_rewards[best_user]:.2f}")

if __name__ == "__main__":
    if len(sys.argv) != 9:
        print("Usage: python Bandit.py <alg> <reward> <range> <min> <step> <max> <engagement_file> <network_file>")
    
    #Assigns input values to variables and converts to usable types
    algorithm, reward_type, range_value, min_param, step, max_param, engagement_file, network_file = sys.argv[1:]
    range_value = int(range_value)
    min_param, step, max_param = float(min_param), float(step), float(max_param)

    # Loads the engagements and network
    engagements, network = load_data(engagement_file, network_file)
    run_bandit(engagements, network, algorithm, reward_type, range_value, min_param, step, max_param)