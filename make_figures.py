import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import random
import string
import math
import os
from decryption import generate_random_key, encrypt_text, decrypt_text, compute_log_likelihood, predict_encryption_key, get_word_frequencies, generate_initial_key
from test_decryption import read_text_file
from itertools import cycle
import glasbey

PASSAGE_SIZES = ['short', 'medium', 'long']
BASE_PATH = "/Users/anjali/Library/CloudStorage/OneDrive-Stanford/Sophomore Year/Fall 2024/CS 109/Challenge Project/"
def visualize_convergences():
    word_prob_dict = get_word_frequencies()
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    for i in range(len(PASSAGE_SIZES)):
        size = PASSAGE_SIZES[i]
        passage_dir = f"{BASE_PATH}/{size}_passages"
        convergence_info = {}
        for filename in os.listdir(passage_dir):
            file_path = os.path.join(passage_dir, filename)
            if os.path.isfile(file_path) and filename.endswith('.txt'): 
                plaintext = read_text_file(file_path)

                encryption_map = generate_random_key()
                encrypted_text = encrypt_text(plaintext, encryption_map)
                likelihood_right_mapping = compute_log_likelihood(encrypted_text, encryption_map, word_prob_dict)
                print(f"CORRECT MAPPING LIKELIHOOD: {likelihood_right_mapping}")

                mapping_guess, likelihoods = predict_encryption_key(encrypted_text)
                decrypted = decrypt_text(encrypted_text, mapping_guess)
                print(decrypted)

                convergence_info[filename] = [likelihoods, likelihood_right_mapping]
        plt.sca(axs[i])  # Select the first subplot
        build_convergence_plot(convergence_info, size)

    fig.suptitle('Convergence of Log Likelihoods', fontsize=16, fontweight='bold', color='#6A0DAD')
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()


def build_convergence_plot(convergence_info, size):
    # Create a color palette using glasbey with 15 colors
    color_palette = glasbey.create_palette(palette_size=12, lightness_bounds=(20, 40), chroma_bounds=(40, 50))
    colors = cycle(color_palette)  # Cycle through the palette

    # plt.figure(figsize=(10, 6))
    for filename, likelihood_info in convergence_info.items():
        likelihoods, expected_likelihood = likelihood_info
        color = next(colors)

        # Plot the likelihoods and retrieve the line color
        plt.plot(likelihoods, label=f"{filename}", color=color, linewidth=2)
        
        # Plot the horizontal line with the same color
        plt.axhline(y=expected_likelihood, color=color, linestyle='--')

    plt.xlabel('Iteration')
    plt.ylabel('Log Likelihood')
    plt.title(f'{size.capitalize()}-length Passages', fontweight='bold') 
    plt.legend()

    # Show the plot
    # plt.show()

def main():
    visualize_convergences()

if __name__ == "__main__":
    main()
