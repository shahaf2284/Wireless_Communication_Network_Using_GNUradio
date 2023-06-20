
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from timeit import default_timer as timer
import itertools

#   -------------------------------------- global variables -----------------------------------
rep_times = 5000 # number of times to repeat for each simulation
k = 30    # number of receiving users
t = 2     # number of transmitting antennae
sigma_squared = 0.5     # variance for each complex gaussian
P = 5   # the total power allowed
num_bins = 150  # number of bins in the histogram
num_processes = 10  # how many processes\cores do we want to divide the 5000 runs on
images_dir = "C:\\Users\\shaha\\OneDrive\\שולחן העבודה\\final project mininet\\pic"    # where to save images of histograms\
list_of_colors = ['red', 'blue', 'green', 'orange', 'purple','cyan','gray','magenta','brown']   #histogram colors


chaotic_means = np.random.uniform(0, 3, k)
chaotic_variances = np.random.uniform(0, 1, k)
iid_means = np.zeros(t)
iid_variances = sigma_squared*np.eye(t)     # covariance matrix

#   --------------------------------------------Functions ------------------------------------------------

def iid_gaussian_channel():
    # create a complex gaussian channel such that every row has real part has means vector(0,0,0) of length t and covariance matrix of diagonal with 0.25
    # the imaginary part has the same attributes per row and the matrix is just the sum of those vectors
    H = np.zeros((k, t), dtype=complex)
    for i in range(k):
        H_row_real = np.random.multivariate_normal(iid_means, (1/2) * iid_variances)
        H_row_imaginary = 1j * np.random.multivariate_normal(iid_means, (1/2) * iid_variances)
        H[i] = H_row_real + H_row_imaginary
    return H


def chaotic_channel():
    # create a chaotic channel, we rolled above the parameters for the means and variances
    # each row has REAL PART means means[i] as a vector and covariance matrix of 1/2*variance[i] on its diagonal
    # the same goes for imaginary part
    H = np.zeros((k, t), dtype=complex)
    for i in range(k):
        H_row_real = np.random.multivariate_normal((1/2) * chaotic_means[i]*np.ones(t), (1/2)*chaotic_variances[i] * np.eye(t))
        H_row_imaginary = 1j * np.random.multivariate_normal((1/2)*chaotic_means[i] * np.ones(t), (1/2)*chaotic_variances[i] * np.eye(t))
        H[i] = H_row_real + H_row_imaginary
    return H


def correlated_channel():
    # roll a random vector for real values, a random gaussian vector for complex and then sum them and duplicate k times
    H = np.zeros((k, t), dtype=complex)
    first_row_real = np.random.multivariate_normal(iid_means, (1/2) * iid_variances)
    first_row_imaginary = 1j * np.random.multivariate_normal(iid_means, (1/2) * iid_variances)
    first_row = first_row_real + first_row_imaginary
    for i in range(k):
        H[i] = first_row
    return H


def select_users(H, Q):
    # this function selects the best users (highest norm row in H) if needed or the first one or randomly , depending on Q.
    # it returns a vector of size (k) with 1 if its a good user
    if Q == 1:
        # just pick randomly a single user[in our case the first one] and give him all the power
        res = np.zeros(k)
        res[0] = P
        return res
    if Q == 2:
        # find highest norm row and give him all the power
        max_norm = 0
        best_user = 0
        for i in range(k):
            row_norm = np.linalg.norm(H[i])
            if row_norm > max_norm:
                max_norm = row_norm
                best_user = i
        res = np.zeros(k)
        res[best_user] = P
        return res
    if Q == 3:
        # pick t random elements and send to them at uniform power
        user_indices = np.random.choice(np.arange(k), size=t, replace=False)    # pick t random numbers in (0,1,...k-1) and return them as a vector
        res = np.zeros(k)
        for index in user_indices:
            res[index] = P/t
        return res
    if Q == 4:
        norms = np.linalg.norm(H, axis=1) # a matrix of norms
        indices = np.argpartition(norms, -t)[-t:]   # take the t row indices that have the biggest norm
        powers = np.zeros(k)
        for index in indices:
            powers[index] = P/t
        return powers
    if Q == 5:
        # now we need to pick t users  out of k, check the transmission rates for each combination and pick the best
        best_users = None   # will be a vector of powers for each user, our result
        best_rate = 0
        all_users = list(range(k))
        combinations = itertools.combinations(all_users,t)
        for combination in combinations:    # combination looks like [2,5] or [6,1] and combinations include [[1,2],[1,3],..[5,6]]
            powers = np.zeros(k)
            for element in combination:
                powers[element] = P/t
            res_rate = calc_rates(H,powers)
            if res_rate > best_rate:
                best_rate,best_users = res_rate , powers
        return best_users
    if Q == 8:
        #we take the highest norms, we calculate their  ratios, for example if norm A is 4 and norm B is 1 their rations are 4/5 and 1/5 , and then we feed those powers to them
        norms = np.linalg.norm(H, axis=1)  # a matrix of norms
        indices = np.argpartition(norms, -t)[-t:]  # take the t row indices that have the biggest norm
        sum_norms = 0
        for index in indices:
            sum_norms += norms[index]
        powers = np.zeros(k)
        for index in indices:
            powers[index] = norms[index]/sum_norms *P
        return powers


def calc_rates(H, users_powers):
    # H is a matrix and users_powers is a vector like (p/k,p/k,0,0,0...) for t users or 1 user
    # this function calculate the total rate using the formula below, where P_k is drawn from users_powers
    # vdot performs <h_k,v_k> where the inner product is done with complex conjugate of h_k
    # we take the i row of H and i column of V
    V = np.linalg.pinv(H)
    sum_rates = 0
    single_rate = lambda i: np.log((1 + users_powers[i]*abs((np.vdot(H[i], V[:, i]/np.linalg.norm(V[:, i]))))**2))
    # calculate the rate for each user
    for i in range(k):
        #print("Inner product<h_i*,v_i> = ", abs((np.vdot(H[i], V[:, i] / np.linalg.norm(V[:, i])))) ** 2,"Pi=",users_powers[i],"i=",i)
        sum_rates += single_rate(i)
    return sum_rates

def f(n:int):
    # builds a tuple of (iid_res[500],chaotic_res[500],correlated_res[500] for a given question number
    iid_res = np.zeros(rep_times// num_processes)
    chaotic_res = np.zeros(rep_times // num_processes)
    correlated_res = np.zeros(rep_times // num_processes)
    for i in range(rep_times//num_processes):
        # IID PART

        H_iid = iid_gaussian_channel()
        iid_user_powers = select_users(H_iid, n)
        iid_rate = calc_rates(H_iid, iid_user_powers)
        iid_res[i] = iid_rate

        # Chaotic part
        H_chaotic = chaotic_channel()
        chaotic_user_powers = select_users(H_chaotic, n)
        chaotic_rate = calc_rates(H_chaotic, chaotic_user_powers)
        chaotic_res[i] = chaotic_rate

        # Correlated part
        H_correlated = correlated_channel()
        correlated_user_powers = select_users(H_correlated, n)
        correlated_rate = calc_rates(H_correlated, correlated_user_powers)
        correlated_res[i] = correlated_rate
    return iid_res,chaotic_res,correlated_res

def Q(n: int):
    print(f"Running Q{n}")
    # n is the question number, it is relevant for the select_users function only
    # this function creates 3 matrices Hiid,Hchaos,Hcorr and does BF on 3 of them on a selected set of users calculated in select_users func
    # it calculates the rate and returns 3 np arrays of size 5000 for each of them.

    with Pool(processes=num_processes) as pool_inner:   #multiproccessing here, each process runs f(n)
        per_func_results = pool_inner.map(f,[n]*num_processes)  # 10 vectors of (IID,CHAOTIC,CORR) of length 500
    iid_res = np.concatenate([per_func_results[i][0] for i in range(num_processes)])
    chaotic_res = np.concatenate([per_func_results[i][1] for i in range(num_processes)])
    correlated_res = np.concatenate([per_func_results[i][2] for i in range(num_processes)])
    return (iid_res, chaotic_res, correlated_res)

#   -------------------------------------- MAIN --------------------------------------------------
if __name__ == "__main__":
    simulation_start = timer()
    funs_to_run = [1,2,3,4,5,8]     # what functions we want to run, optimally we finish 1-8 and we run them all on different processors
    colors = list_of_colors[:len(funs_to_run)]
    print(f"Running functions :{funs_to_run}")
    results=np.zeros((len(funs_to_run), 3 , rep_times))
    # Results[i] = 3 vectors of 5000
    if __name__ == "__main__":
        res_idx=0
        for i in (funs_to_run):
            results[res_idx]=Q(i)
            res_idx += 1
    total_iid_rates = np.zeros(0)
    total_chaotic_rates = np.zeros(0)
    total_correlated_rates = np.zeros(0)
    # for i in range(len(funs_to_run)):
    #     total_iid_rates = np.concatenate((total_iid_rates, results[i][0]))
    #     total_chaotic_rates = np.concatenate((total_chaotic_rates, results[i][1]))
    #     total_correlated_rates = np.concatenate((total_correlated_rates, results[i][2]))

    simulation_end = timer()
    print(f"The simulation took {simulation_end-simulation_start} seconds.")

#   ----------------------------------------- PLOTTING STAGE ---------------------------------------------

    #Results[i] = 3 vectors of 5000 for i = Question i+1
    a = plt.hist([results[i][0] for i in range(len(funs_to_run))], alpha=0.3, bins=num_bins, label=[f"Histogram Q{i}" for i in funs_to_run], density=True, color=colors,stacked=True)
    plt.xlabel("Transmission rates")
    plt.ylabel("Probability")
    plt.grid()
    plt.legend(loc='upper right')
    plt.title(f"IID Channel with k={k} and t={t}")
    plt.savefig(images_dir + "\\Iid_Channel.png")  # images_dir can be found in the global variables to change the wanted path
    plt.show()

    b = plt.hist([results[i][1] for i in range(len(funs_to_run))], alpha=0.3, bins=num_bins, label=[f"Histogram Q{i}" for i in funs_to_run], density=True, color=colors,stacked=True)
    plt.xlabel("Transmission rates")
    plt.ylabel("Probability")
    plt.grid()
    plt.legend(loc='upper right')
    plt.title(f"Chaotic Channel with k={k} and t={t}")
    plt.savefig(images_dir + "\\Chaotic_Channel.png")  # images_dir can be found in the global variables to change the wanted path
    plt.show()

    c = plt.hist([results[i][2] for i in range(len(funs_to_run))], alpha=0.3, bins=num_bins, label=[f"Histogram Q{i}" for i in funs_to_run], density=True, color=colors,stacked=True)
    plt.xlabel("Transmission rates")
    plt.ylabel("Probability")
    plt.grid()
    plt.legend(loc='upper right')
    plt.title(f"Correlated Channel with k={k} and t={t}")
    plt.savefig(images_dir + "\\Correlated_Channel.png")    # images_dir can be found in the global variables to change the wanted path
    plt.show()
