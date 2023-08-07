import numpy as np

def compute_likelihood_awgn_custom(received_codeword, noise_variance):
    """
    Compute likelihood for a received codeword in an Additive White Gaussian Noise (AWGN) model
    using custom formula 2y / sigma^2 as the likelihood function.
    
    Parameters:
        received_codeword (numpy.ndarray): Received codeword (floating-point vector).
        noise_variance (float): Variance of the AWGN (non-negative value).
        
    Returns:
        numpy.ndarray: Likelihood for the received codeword (floating-point vector).
    """
    likelihood = 2 * received_codeword / noise_variance
    return likelihood

# Пример использования
received_codeword = np.array([0.1, -0.3, 0.2, -0.4, 0.5])  # Пример полученного кодового слова (может быть с плавающей точкой)
noise_variance = 0.1  # Пример дисперсии шума на канале AWGN

likelihood = compute_likelihood_awgn_custom(received_codeword, noise_variance)
print("Likelihoods:", likelihood)

#_____________________________________________

import numpy as np

def compute_likelihood_awgn_normal(received_codeword, noise_variance):
    """
    Compute likelihood for a received codeword in an Additive White Gaussian Noise (AWGN) model
    using normal distribution as the likelihood function.
    
    Parameters:
        received_codeword (numpy.ndarray): Received codeword (floating-point vector).
        noise_variance (float): Variance of the AWGN (non-negative value).
        
    Returns:
        numpy.ndarray: Likelihood for the received codeword (floating-point vector).
    """
    likelihood = (1 / np.sqrt(2 * np.pi * noise_variance)) * np.exp(-0.5 * (received_codeword**2) / noise_variance)
    return likelihood

# Пример использования
received_codeword = np.array([0.1, -0.3, 0.2, -0.4, 0.5])  # Пример полученного кодового слова (может быть с плавающей точкой)
noise_variance = 0.1  # Пример дисперсии шума на канале AWGN

likelihood = compute_likelihood_awgn_normal(received_codeword, noise_variance)
print("Likelihoods:", likelihood)



#____

import numpy as np

def compute_likelihood_awgn_laplace(received_codeword, noise_variance):
    """
    Compute likelihood for a received codeword in an Additive White Gaussian Noise (AWGN) model
    using Laplace distribution as the likelihood function.
    
    Parameters:
        received_codeword (numpy.ndarray): Received codeword (floating-point vector).
        noise_variance (float): Variance of the AWGN (non-negative value).
        
    Returns:
        numpy.ndarray: Likelihood for the received codeword (floating-point vector).
    """
    likelihood = (1 / (2 * noise_variance)) * np.exp(-np.abs(received_codeword) / noise_variance)
    return likelihood

# Пример использования
received_codeword = np.array([0.1, -0.3, 0.2, -0.4, 0.5])  # Пример полученного кодового слова (может быть с плавающей точкой)
noise_variance = 0.1  # Пример дисперсии шума на канале AWGN

likelihood = compute_likelihood_awgn_laplace(received_codeword, noise_variance)
print("Likelihoods:", likelihood)


#————————

import numpy as np

def compute_likelihood_awgn(received_codeword, noise_variance):
    """
    Compute likelihood for a received codeword in an Additive White Gaussian Noise (AWGN) model.
    
    Parameters:
        received_codeword (numpy.ndarray): Received codeword (floating-point vector).
        noise_variance (float): Variance of the AWGN (non-negative value).
        
    Returns:
        numpy.ndarray: Likelihood for the received codeword (floating-point vector).
    """
    likelihood = np.exp(-0.5 * (received_codeword**2) / noise_variance) / np.sqrt(2 * np.pi * noise_variance)
    return likelihood

# Пример использования
received_codeword = np.array([0.1, -0.3, 0.2, -0.4, 0.5])  # Пример полученного кодового слова (может быть с плавающей точкой)
noise_variance = 0.1  # Пример дисперсии шума на канале AWGN

likelihood = compute_likelihood_awgn(received_codeword, noise_variance)
print("Likelihoods:", likelihood)


#--

Для использования алгоритма Box-Plus-Phi (BP-Phi) для LDPC-декодирования, нам нужно добавить функцию активации `phi()`
и изменить обновление сообщений в функции `sum_product_ldpc_decoder`. Алгоритм BP-Phi обычно позволяет ускорить сходимость 
декодера по сравнению с обычным Sum-Product. Вот измененный код:

```python
import numpy as np

def phi(x):
    """
    Activation function used in Box-Plus-Phi algorithm.
    
    Parameters:
        x (float): Input value.
        
    Returns:
        float: Output value after applying the activation function.
    """
    return np.log((np.exp(x) + 1) / (np.exp(x) - 1))

def sum_product_bp_phi_decoder(H, received_codeword, max_iterations=50, verbose=True):
    """
    LDPC decoder using the Box-Plus-Phi algorithm (BP-Phi).
    
    Parameters:
        H (numpy.ndarray): Parity-check matrix (binary matrix).
        received_codeword (numpy.ndarray): Received codeword (binary vector).
        max_iterations (int): Maximum number of iterations for decoding.
        verbose (bool): If True, print iteration-wise messages during decoding.
        
    Returns:
        numpy.ndarray: Decoded codeword (binary vector).
    """
    num_checks, num_codewords = H.shape
    codeword_length = len(received_codeword)
    
    # Initialize messages
    L_c = np.zeros((num_checks, num_codewords)) # Messages from checks to variables
    L_v = np.zeros((num_checks, num_codewords)) # Messages from variables to checks
    decoded_codeword = received_codeword.copy() # Initialize decoded codeword with the received codeword
    
    for iteration in range(max_iterations):
        # Update check-to-variable messages
        for i in range(num_checks):
            for j in range(num_codewords):
                neighbors = np.where(H[i] == 1)[0] # Indices of neighboring variables connected to check i
                L_c[i, j] = phi(received_codeword[j] + np.sum(L_v[i, neighbors]) - L_v[i, j])
        
        # Update variable-to-check messages
        for i in range(num_checks):
            for j in range(num_codewords):
                neighbors = np.where(H[:, j] == 1)[0] # Indices of neighboring checks connected to variable j
                L_v[i, j] = phi(np.sum(L_c[neighbors, j]) - L_c[i, j])
        
        # Perform hard decision based on variable-to-check messages
        decoded_codeword = np.sign(L_v).astype(int)
        
        # Check if decoded codeword satisfies the parity-check equations
        syndrome = np.dot(H, decoded_codeword) % 2
        if np.all(syndrome == 0):
            if verbose:
                print(f"Decoding successful after {iteration+1} iterations.")
            break

        if verbose:
            print(f"Iteration {iteration+1}, Syndrome: {syndrome}")

    return decoded_codeword
```

Здесь мы добавили функцию активации `phi()`, которая является частью алгоритма BP-Phi, и заменили обновление сообщений 
на использование этой функции. Теперь LDPC-декодер должен использовать алгоритм Box-Plus-Phi для обработки сообщений и ускорить 
сходимость в сравнении с обычным Sum-Product.


#-- -


Конечно! Ниже приведен код LDPC-декодера с алгоритмом Sum-Product Belief Propagation с добавленными комментариями:

```python
import numpy as np

def sum_product_ldpc_decoder(H, received_codeword, max_iterations=50, verbose=True):
    """
    LDPC decoder using the Sum-Product Belief Propagation algorithm.
    
    Parameters:
        H (numpy.ndarray): Parity-check matrix (binary matrix).
        received_codeword (numpy.ndarray): Received codeword (binary vector).
        max_iterations (int): Maximum number of iterations for decoding.
        verbose (bool): If True, print iteration-wise messages during decoding.
        
    Returns:
        numpy.ndarray: Decoded codeword (binary vector).
    """
    num_checks, num_codewords = H.shape
    codeword_length = len(received_codeword)
    
    # Initialize messages
    L_c = np.zeros((num_checks, num_codewords)) # Messages from checks to variables
    L_v = np.zeros((num_checks, num_codewords)) # Messages from variables to checks
    decoded_codeword = received_codeword.copy() # Initialize decoded codeword with the received codeword
    
    for iteration in range(max_iterations):
        # Update check-to-variable messages
        for i in range(num_checks):
            for j in range(num_codewords):
                neighbors = np.where(H[i] == 1)[0] # Indices of neighboring variables connected to check i
                L_c[i, j] = received_codeword[j] + np.sum(L_v[i, neighbors]) - L_v[i, j]
        
        # Update variable-to-check messages
        for i in range(num_checks):
            for j in range(num_codewords):
                neighbors = np.where(H[:, j] == 1)[0] # Indices of neighboring checks connected to variable j
                L_v[i, j] = np.sum(L_c[neighbors, j]) - L_c[i, j]
        
        # Perform hard decision based on variable-to-check messages
        decoded_codeword = np.sign(L_v).astype(int)
        
        # Check if decoded codeword satisfies the parity-check equations
        syndrome = np.dot(H, decoded_codeword) % 2
        if np.all(syndrome == 0):
            if verbose:
                print(f"Decoding successful after {iteration+1} iterations.")
            break

        if verbose:
            print(f"Iteration {iteration+1}, Syndrome: {syndrome}")

    return decoded_codeword
```

Надеюсь, что комментарии помогут лучше понять каждый шаг в работе декодера.








