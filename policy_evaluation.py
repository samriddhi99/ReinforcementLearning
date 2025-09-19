import numpy as np
from Assignment2Tools import prob_vector_generator, markov_matrix_generator
from itertools import product
import time

start = time.time()
def action_space_func(phi, b, p, t, z, eta, Swind):
    if p == 0:
        return [-1]
    elif p == 1 and b < eta:
        return [-1]
    elif p == 1 and b >= eta and t == 0:
        return [-1]
    elif p == 1 and b >= eta and t > 0:
        return [-1] + list(range(len(Swind)))
    
def qfunc(phi, b, p, t, z, a, V, P, lmbda, B, eta, alpha, gamma, tau_max, beta, Swind):
    
    q_val = 0
  
    for phi_prime in range(P.shape[1]):
        p_phi = P[phi, phi_prime] 
        for delta, alpha_delta in enumerate(alpha):
       
            if p == 0:  
                reward = -((Swind[phi] - Swind[z]) ** 2)  
                b_prime = min(b + delta, B)  
                q_val_active = reward + beta * V[phi_prime, b_prime, 1, tau_max, z] 
                q_val_passive = reward + beta * V[phi_prime, b_prime, 0, 0, z]  
                q_val += p_phi * alpha_delta * (gamma * q_val_active + (1 - gamma) * q_val_passive)
           
            elif p == 1:  
                if t > 1: 
                    p_prime = 1
                    tau_prime = t - 1
                else: 
                    p_prime = 0
                    tau_prime = 0
                if a == -1: 
                    reward = -((Swind[phi] - Swind[z]) ** 2)
                    b_prime = min(b + delta, B)  
                    q_val += p_phi * alpha_delta * (reward + beta * V[phi_prime, b_prime, p_prime, tau_prime, z])                    
                else:  
                    reward = -((Swind[phi] - Swind[a]) ** 2)  
                    b_prime = min(b - eta + delta, B)  
                    z_success = a 
                    z_fail = z                    
                    q_success = beta * V[phi_prime, b_prime, p_prime, tau_prime, z_success]
                    q_fail = reward + beta * V[phi_prime, b_prime, p_prime, tau_prime, z_fail]
                    q_val += p_phi * alpha_delta * (lmbda * q_success + (1 - lmbda) * q_fail)
    return q_val


def policy(phi, b, p, t, z, V, P, lmbda, B, eta, alpha, gamma, tau, beta, Swind):
    action_space = action_space_func(phi, b, p, t, z, eta, Swind)
    max_q = -np.inf
    best_action = None
    for a in action_space:
        q_val = qfunc(phi, b, p, t, z, a, V, P, lmbda, B, eta, alpha, gamma, tau, beta, Swind)
        if q_val > max_q:
            max_q = q_val
            best_action = a
    return best_action

def policy_evaluation(Swind, P, lmbda, B, eta, Delta, alpha, tau, gamma, beta, theta, Kmin):
    n_phi = len(Swind)
    n_b = B + 1
    n_p = 2
    n_tau = tau + 1
    n_z = len(Swind)
    V = -np.random.uniform(1, 10, size=(n_phi, n_b, n_p, n_tau, n_z))
    V_new = np.zeros_like(V)
    Delta_val = np.inf
    iteration = 1
    state_space = product(range(n_phi), range(n_b), range(n_p), range(n_tau), range(n_z))

    while Delta_val > theta or iteration <= Kmin:
        print(iteration, Delta_val)
        for phi, b, p, t, z in state_space:
           
            best_action = policy(phi, b, p, t, z, V, P, lmbda, B, eta, alpha, gamma, tau, beta, Swind)
           
            V_new[phi, b, p, t, z] = qfunc(phi, b, p, t, z, best_action, V, P, lmbda, B, eta, alpha, gamma, tau, beta, Swind)
          
        Delta_val = np.max(np.abs(V_new - V))
        V = np.copy(V_new)
        iteration += 1
        
    return V
   

# System parameters (set to default values)
Swind = np.linspace(0, 1, 21)                      # The set of all possible normalized wind speed.
mu_wind = 0.3                                      # Mean wind speed. You can vary this between 0.2 to 0.8.
z_wind = 0.5                                       # Z-factor of the wind speed. You can vary this between 0.25 to 0.75.
                                                   # Z-factor = Standard deviation divided by mean.
                                                   # Higher the Z-factor, the more is the fluctuation in wind speed.
stddev_wind = z_wind*np.sqrt(mu_wind*(1-mu_wind))  # Standard deviation of the wind speed.
retention_prob = 0.9                               # Retention probability is the probability that the wind speed in the current and the next time slot is the same.
                                                   # You can vary the retention probability between 0.05 to 0.95.
                                                   # Higher retention probability implies lower fluctuation in wind speed.

P = markov_matrix_generator(Swind, mu_wind, stddev_wind, retention_prob)  # Markovian probability matrix governing wind speed.

lmbda = 0.7  # Probability of successful transmission.

B = 10         # Maximum battery capacity.
eta = 2        # Battery power required for one transmission.
Delta = 3      # Maximum solar power in one time slot.
mu_delta = 2   # Mean of the solar power in one time slot.
z_delta = 0.5  # Z-factor of the slower power in one time slot. You can vary this between 0.25 to 0.75.                  
stddev_delta = z_delta*np.sqrt(Delta*(Delta-mu_delta))  # Standard deviation of the solar power in one time slot.
alpha = prob_vector_generator(np.arange(Delta+1), mu_delta, stddev_delta)  # Probability distribution of solar power in one time slot.

tau = 4       # Number of time slots in active phase.
gamma = 1/15  # Probability of getting chance to transmit. It can vary between 0.01 to 0.99.

beta = 0.95   # Discount factor.
theta = 0.01  # Convergence criteria: Maximum allowable change in value function to allow convergence.
Kmin = 10     # Convergence criteria: Minimum number of iterations to allow convergence.




V = policy_evaluation(Swind, P, lmbda, B, eta, Delta, alpha, tau, gamma, beta, theta, Kmin)
end = time.time()
print(end - start)