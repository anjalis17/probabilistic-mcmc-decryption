import pandas as pd
import numpy as np
from scipy import stats
import math
import matplotlib.pyplot as plt
import random
import string

LEN_ALPHABET = 26
NUM_ITERATIONS = 10000
WORD_TOTAL = 1000000000000
# Frequencies from https://www3.nd.edu/~busiforc/handouts/cryptography/Letter%20Frequencies.html
LETTER_PROBABILITIES =  {
        'A': 0.08167, 'B': 0.01492, 'C': 0.02782, 'D': 0.04253, 'E': 0.12702,
        'F': 0.02228, 'G': 0.02015, 'H': 0.06094, 'I': 0.06966, 'J': 0.00153,
        'K': 0.00772, 'L': 0.04025, 'M': 0.02406, 'N': 0.06749, 'O': 0.07507,
        'P': 0.01929, 'Q': 0.00095, 'R': 0.05987, 'S': 0.06327, 'T': 0.09056,
        'U': 0.02758, 'V': 0.00978, 'W': 0.02360, 'X': 0.00150, 'Y': 0.01974, 'Z': 0.00074
    }

# Generates random encryption key;
# Returns dictionary mapping from encrypted text to alphabet
def generate_random_key():
    alphabet = list(string.ascii_uppercase)

    # Deep copy of alphabet to shuffle for creating random encryption key
    key = alphabet[:]
    random.shuffle(key)

    # Create dict mapping
    encryption_map = {}
    for i in range(LEN_ALPHABET):
        letter = alphabet[i]
        mapped = key[i]
        encryption_map[letter] = mapped

    return encryption_map

# Decrypt encrypted text message given encryption map 
# Returns encrypted message string
def encrypt_text(plaintext, encryption_map):
    flipped_encryption_map = {}
    for encrypt_ch in encryption_map:
        orig_ch = encryption_map[encrypt_ch]
        flipped_encryption_map[orig_ch] = encrypt_ch
    
    plaintext = plaintext.upper().strip()
    encrypted_text = ""

    for ch in plaintext:
        if not ch.isalpha():
            encrypted_text += ch
        else:
            encrypted_text += flipped_encryption_map[ch]
    
    return encrypted_text

# Decrypt encrypted text message via encryption map 
# Returns decrypted message string
def decrypt_text(text, encryption_map):
    text = text.upper().strip()
    decrypted_text = ""

    for ch in text:
        if not ch.isalpha():
            decrypted_text += ch
        else:
            decrypted_text += encryption_map[ch]
    
    return decrypted_text

def get_letter_frequency(letter):
    return LETTER_PROBABILITIES[letter]

def get_word_frequencies():
    filename = "/Users/anjali/Library/CloudStorage/OneDrive-Stanford/Sophomore Year/Fall 2024/CS 109/Challenge Project/english_word_freq.csv"
    freqs = pd.read_csv(filename)
    freqs['count'] = freqs['count'] / WORD_TOTAL
    freqs = freqs.dropna(subset=['count', 'word'])
    freqs = freqs.reset_index(drop=True)
    word_prob_dict = {}
    for i in range(len(freqs)):
        word = freqs['word'][i]
        freq = float(freqs['count'][i])
        word_prob_dict[word.upper()] = freq
    return word_prob_dict

# def compute_likelihood_letterString(decrypted_str, swap_space):
#     log_likelihood = 0
#     for letter in decrypted_str:
#         if letter.isalpha():
#             log_likelihood += math.log(get_letter_frequency(letter)) 

#             # Update letter's entry in swap space
#             if letter not in swap_space:
#                 swap_space[letter] = 0
#             swap_space[letter] += 1

#     return log_likelihood

# def compute_likelihood_letterString(decrypted_str, swap_space, encryption_key):
#     flipped_encryption_map = {}
#     for encrypt_ch in encryption_key:
#         orig_ch = encryption_key[encrypt_ch]
#         flipped_encryption_map[orig_ch] = encrypt_ch

#     log_likelihood = 0
#     for letter in decrypted_str:
#         if letter.isalpha():
#             log_likelihood += math.log(get_letter_frequency(letter)) 

#             # Update encrypted letter's entry in swap space
#             encrypt_ch = flipped_encryption_map[letter]
#             if encrypt_ch not in swap_space:
#                 swap_space[encrypt_ch] = 0
#             swap_space[encrypt_ch] += 1

#     return log_likelihood

def compute_likelihood_letterString(decrypted_str):
    log_likelihood = 0
    for letter in decrypted_str:
        if letter.isalpha():
            log_likelihood += math.log(get_letter_frequency(letter)) 
    return log_likelihood
 
# Computes likelihood of text given encryption key (dict mapping)
# by (1) decrypting the encrypted text according to the key,
# and (2) finding probability of decrypted text based on English 
# letter frequencies
# Returns the log likelihood
# P (encrypted text | encryption key)
def compute_log_likelihood(encrypted_text, encryption_key, word_prob_dict):
    # 1) Decrypt encrypted text
    decrypted_text = decrypt_text(encrypted_text, encryption_key)

    # 2) Compute likelihood of decrypted text
    tokens = decrypted_text.split()
    # print(tokens)
    log_likelihood = 0
    for token in tokens:
        token = token.strip()
        # Clean token -- strip punctuation, numbers
        cleaned_token = ''.join(char for char in token if char.isalpha())
        if cleaned_token in word_prob_dict:
            log_likelihood += math.log(word_prob_dict[cleaned_token])
        else:
            smoothed_likelihood = math.log(1e-7)  # Small probability for unknown words
            log_likelihood += max(smoothed_likelihood, compute_likelihood_letterString(token))
    return log_likelihood

def bernoulli_coin_flip(p):
    return 1 if random.random() < p else 0

# def get_two_swap_letters(swap_space):
#     letters, freqs = zip(*swap_space.items())
#     # weights=freqs
#     letter1 = random.choices(letters, weights=freqs, k=1)[0]
#     letter2 = letter1
#     while letter2 == letter1:
#         letter2 = random.choices(letters, weights=freqs, k=1)[0]
#     return letter1, letter2

def generate_initial_key(encrypted_text):
    letters_in_freq_order = ""

    # Sort by value in descending order
    sorted_dict = dict(sorted(LETTER_PROBABILITIES.items(), key=lambda item: item[1], reverse=True))
    for key in sorted_dict:
        letters_in_freq_order += key

    encrypted_text = encrypted_text.upper().strip()
    encrypted_freq = {ch: encrypted_text.count(ch) for ch in set(encrypted_text) if ch.isalpha()}
    sorted_encrypted = dict(sorted(encrypted_freq.items(), key=lambda item: item[1], reverse=True))

    encrypted_in_freq_order = ""
    for key in sorted_encrypted:
        encrypted_in_freq_order += key
    
    # Map ciphertext letters to English frequencies
    initial_key = {encrypted_ch: orig_ch for encrypted_ch, orig_ch in zip(encrypted_in_freq_order, letters_in_freq_order)}
    print(initial_key)
    return initial_key


# P(key | text) = P(text | key) P(key) / P(text)
# P(text) = sum over all keys of P(text | key) * P(key)
# P(key) (prior) same for all keys -- 1 / 26!
# So P(key | text) proportional to P(text | key), our likelihood term -- need to maximize this
def predict_encryption_key(encrypted_text):
    word_prob_dict = get_word_frequencies()
    likelihoods = []

    # Generate random encryption key as initial prior -- equal likelihood of being any of the 26! possible keys
    # prior_key = generate_random_key()
    prior_key = generate_initial_key(encrypted_text)
    print(len(LETTER_PROBABILITIES))
    print(len(prior_key))
    # swap_space = {}
    best_likelihood = compute_log_likelihood(encrypted_text, prior_key, word_prob_dict)
    likelihoods.append(best_likelihood)

    for i in range(8000):
        # print(best_likelihood)
        # print(prior_key)

        # Consider swapping any two mapping values in encryption key guess
        keys = list(prior_key.keys())
        key1, key2 = random.sample(keys, 2)
        new_key = prior_key.copy()
        # key1, key2 = get_two_swap_letters(swap_space)
        new_key[key1] = prior_key[key2]
        new_key[key2] = prior_key[key1]

        new_likelihood = compute_log_likelihood(encrypted_text, new_key, word_prob_dict)
        # print(f"{key1} {key2}")
        print(new_likelihood)
        
        # If we get a higher likelihood / posterior (ratio > 1), we update our encryption key belief -- new best!
        # Else, we compute ratio between new posterior and previous (best seen) posterior
        # ratio = new_likelihood / best_likelihood
        # If new posterior is worse...
        # Flip a coin (Bernoulli) with probability success proportional to ratio -- purpose: to avoid getting stuck at local optima

        # Compute the log-likelihood ratio
        log_likelihood_ratio = new_likelihood - best_likelihood

        if new_likelihood > best_likelihood:
            prior_key = new_key
            best_likelihood = new_likelihood
            # swap_space = new_swap_space
            # print('HERE - changed')
        else:
            accept_prob = math.exp(log_likelihood_ratio)  # Exponential scaling of the likelihood ratio
            if bernoulli_coin_flip(accept_prob) == 1:
                prior_key = new_key
                best_likelihood = new_likelihood
                # swap_space = new_swap_space
                # print(f'random flip changed, {accept_prob}')
            # else:
            #     continue
                # print('not changed')
        
        likelihoods.append(best_likelihood)

    print(best_likelihood)
    print(f"Estimated key: {prior_key}")
    return prior_key, likelihoods


def main():
    get_word_frequencies()


if __name__ == "__main__":
    main()