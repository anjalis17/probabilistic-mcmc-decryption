import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import random
import string
import math
import os
from decryption import generate_random_key, encrypt_text, decrypt_text, compute_log_likelihood, predict_encryption_key, get_word_frequencies, generate_initial_key
from test_decryption import read_text_file, conduct_decryption_trials
from itertools import cycle
import glasbey
import ast

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

def get_passage_length(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    content = content.upper().strip()
    tokens = content.split()
    word_count = 0
    for token in tokens:
        if token != "":
            word_count += 1
    print(word_count)
    return word_count

def generate_trial_data():
    data = pd.DataFrame(columns=["Passage", "Group", "Length", "Accuracy Scores", "Execution Times"])
    csv_output_path = f"{BASE_PATH}/decryption_trial_data3.csv"

    for i in range(len(PASSAGE_SIZES)):
        size = PASSAGE_SIZES[i]
        passage_dir = f"{BASE_PATH}/{size}_passages"

        for filename in os.listdir(passage_dir):
            file_path = os.path.join(passage_dir, filename)
            if os.path.isfile(file_path) and filename.endswith('.txt'): 
                print(f"STARTING {filename}")
                word_count = get_passage_length(file_path)
                accuracy_scores, execution_times = conduct_decryption_trials(file_path)
                data = pd.concat([data, pd.DataFrame({
                        "Passage": [filename],
                        "Group": [size],
                        "Length": [word_count],
                        "Accuracy Scores": [accuracy_scores],
                        "Execution Times": [execution_times]
                    })], ignore_index=True)
                print(data)
                data.to_csv(csv_output_path, index = True)
        print(f"FINISHED {size}")

    print("ALL TRIALS DONE!")
    print(data)

def clean_passage_name(filename):
    cleaned_name = os.path.splitext(filename)[0]  # "Harry_Potter"
    cleaned_name = cleaned_name.replace('_', ' ')
    return cleaned_name

def process_trial_data(csv_path):
    data = pd.read_csv(csv_path)
    data['Passage'] = data['Passage'].apply(lambda x: clean_passage_name(x))
    data['Accuracy Scores'] = data['Accuracy Scores'].apply(ast.literal_eval)
    data['Execution Times'] = data['Execution Times'].apply(ast.literal_eval)

    data['Mean Accuracy Score'] = data['Accuracy Scores'].apply(lambda scores: sum(scores) / len(scores))
    data['Mean Execution Time'] = data['Execution Times'].apply(lambda scores: sum(scores) / len(scores))

    # Estimate population variance with ddof = 1 correction
    data['Variance of Mean Accuracy'] = data['Accuracy Scores'].apply(lambda x: np.var(x, ddof=1) / len(x))
    data['Standard Deviation of Mean Accuracy'] = data['Variance of Mean Accuracy'].apply(lambda x: np.sqrt(x))

    print(data)

    size = 'long'
    df = data[data['Group'] == size]
    # Create the bar plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        df['Passage'], 
        df['Mean Accuracy Score'], 
        yerr=df['Standard Deviation of Mean Accuracy'], 
        capsize=5,  # Size of error bar caps
        color='skyblue',  # Bar color
        edgecolor='black',  # Edge color
        alpha=0.8,  # Transparency
        label='Mean Accuracy'
    )

    # Adjust the labels
    plt.xlabel('Passage Name', fontsize=12)
    plt.ylabel('Mean Accuracy Score', fontsize=12)
    plt.title(f'Mean Accuracy Scores for {size.capitalize()} Passages', 
          fontsize=14, 
          color='#1CA3DE', 
          weight='bold')
    plt.xticks(fontsize=10, rotation=30)  # Adjust font size and rotate labels
    plt.ylim(0, 1)  # Ensure the y-axis is between 0 and 1 for clarity
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Adding a legend
    plt.legend()

    # Add the mean accuracy scores as labels on top of the bars
    # Add the mean accuracy scores as bold labels on top of the bars with a background
    for bar, mean in zip(bars, df['Mean Accuracy Score']):
        # Adjust the position of the text and make it bold, change color, and add a background
        plt.text(
            bar.get_x() + bar.get_width() / 2,  # X-coordinate (center of the bar)
            bar.get_height() + 0.02,           # Y-coordinate (slightly above the bar and error bar)
            f'{mean:.2f}',                     # Label text (formatted to 2 decimal places)
            ha='center',                       # Horizontal alignment
            va='bottom',                       # Vertical alignment
            fontsize=10,                       # Font size
            weight='bold',                     # Make text bold
            color='white',                     # Change the color to black for better visibility
            #0D85D8
            bbox=dict(facecolor='skyblue', edgecolor='none', boxstyle='round,pad=0.3', alpha = 0.85)  # Background settings
        )


    # Show the plot
    plt.show()

def main():
    # clean_passage_name("Pride_and_Prejudice.txt")
    # visualize_convergences()
    # generate_trial_data()
    process_trial_data(f"{BASE_PATH}/decryption_trial_data3.csv")

if __name__ == "__main__":
    main()
