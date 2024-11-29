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
COLORS = {'short': ['skyblue', '#1CA3DE'], 'medium': ['#B660CD', '#81007F'], 'long': ['#8BCA84','#46923C']}
def visualize_convergences():
    word_prob_dict = get_word_frequencies()
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    # file_subset = ["CS109_Course_Reader.txt", "To_Kill_A_Mockingbird.txt", "Harry_Potter.txt", "Federalist_Papers.txt"]

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

                convergence_info[clean_passage_name(filename)] = [likelihoods, likelihood_right_mapping]
        plt.sca(axs[i])  # Select subplot
        build_convergence_plot(convergence_info, size)

    fig.suptitle('Convergence of Log Likelihoods', fontsize=16, fontweight='bold', color='#6A0DAD')
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(f'figures/convergence_plot_ALL.png', format='png', dpi=300)
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

def plot_mean_accuracy_scores(csv_path):
    data = pd.read_csv(csv_path)
    data['Passage'] = data['Passage'].apply(lambda x: clean_passage_name(x))
    data['Accuracy Scores'] = data['Accuracy Scores'].apply(ast.literal_eval)
    data['Mean Accuracy Score'] = data['Accuracy Scores'].apply(lambda scores: sum(scores) / len(scores))

    # Estimate population variance with ddof = 1 correction
    data['Variance of Mean Accuracy'] = data['Accuracy Scores'].apply(lambda x: np.var(x, ddof=1) / len(x))
    data['Standard Deviation of Mean Accuracy'] = data['Variance of Mean Accuracy'].apply(lambda x: np.sqrt(x))
    data.to_csv('decryption_trials_processed.csv', index = False)

    print(data)

    passage_order = data[data['Group'] == 'long'].sort_values(by='Mean Accuracy Score')['Passage'].tolist()

    for size in PASSAGE_SIZES:
        palette = COLORS[size]
        print(palette)
        df = data[data['Group'] == size]
        
        df['Passage'] = pd.Categorical(df['Passage'], categories=passage_order, ordered=True)
        df = df.sort_values(by='Passage')

        # Create the bar plot
        plt.figure(figsize=(10, 6))
        bars = plt.bar(
            df['Passage'], 
            df['Mean Accuracy Score'], 
            yerr=df['Standard Deviation of Mean Accuracy'], 
            capsize=5,  # Size of error bar caps
            color=palette[0],  # Bar color
            edgecolor='black',  # Edge color
            alpha=0.8,  # Transparency
            label='Mean Accuracy'
        )

        # Adjust the labels
        plt.xlabel('Passage Name', fontsize=12)
        plt.ylabel('Mean Accuracy Score', fontsize=12)
        plt.title(f'Mean Accuracy Scores for {size.capitalize()} Passages', 
            fontsize=14, 
            color=palette[1], 
            weight='bold')
        plt.xticks(fontsize=10, rotation=25)  # Adjust font size and rotate labels
        plt.ylim(0, 1)  # Ensure the y-axis is between 0 and 1 for clarity
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        # Adding a legend
        plt.legend()

        # Add the mean accuracy scores as labels on top of the bars w/ background
        for bar, mean in zip(bars, df['Mean Accuracy Score']):
            plt.text(
                bar.get_x() + bar.get_width() / 2, # X-coordinate (center of the bar)
                bar.get_height() + 0.02,           # Y-coordinate (slightly above the bar and error bar)
                f'{mean:.2f}',                     # Label text (formatted to 2 decimal places)
                ha='center',                       # Horizontal alignment
                va='bottom',                       # Vertical alignment
                fontsize=10,                       # Font size
                weight='bold',                     # Make text bold
                color='white',                     # Change the color to black for better visibility
                #0D85D8
                bbox=dict(facecolor=palette[0], edgecolor='none', boxstyle='round,pad=0.3', alpha = 0.85)  # Background settings
            )

        plt.savefig(f'figures/mean_accuracy_scores_{size}.png', format='png', dpi=300)
        # Show the plot
        plt.show()


def plot_median_accuracy_scores(csv_path):
    data = pd.read_csv(csv_path)
    data['Accuracy Scores'] = data['Accuracy Scores'].apply(ast.literal_eval)
    data['Median Sample Accuracy Score'] = data['Accuracy Scores'].apply(lambda scores: np.median(scores))
    passage_order = data[data['Group'] == 'long'].sort_values(by='Mean Accuracy Score')['Passage'].tolist()

    for size in PASSAGE_SIZES:
        palette = COLORS[size]
        df = data[data['Group'] == size]
        
        df['Passage'] = pd.Categorical(df['Passage'], categories=passage_order, ordered=True)
        df = df.sort_values(by='Passage')

        # Create the bar plot
        plt.figure(figsize=(10, 6))
        bars = plt.bar(
            df['Passage'], 
            df['Median Sample Accuracy Score'], 
            capsize=5,  # Size of error bar caps
            color=palette[0],  # Bar color
            edgecolor='black',  # Edge color
            alpha=0.8,  # Transparency
            label='Median Sample Accuracy'
        )

        # Adjust the labels
        plt.xlabel('Passage Name', fontsize=12)
        plt.ylabel('Median Sample Accuracy Score', fontsize=12)
        plt.title(f'Median Sample Accuracy Scores for {size.capitalize()} Passages', 
            fontsize=14, 
            color=palette[1], 
            weight='bold',
            pad=25)
        plt.xticks(fontsize=10, rotation=25)  # Adjust font size and rotate labels
        plt.ylim(0, 1)  # Ensure the y-axis is between 0 and 1 for clarity
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.legend()

        for bar, median in zip(bars, df['Median Sample Accuracy Score']):
            # Adjust the position of the text and make it bold, change color, and add a background
            plt.text(
                bar.get_x() + bar.get_width() / 2, # X-coordinate (center of the bar)
                bar.get_height() + 0.02,           # Y-coordinate (slightly above the bar and error bar)
                f'{median:.2f}',                   # Label text (formatted to 2 decimal places)
                ha='center',                       # Horizontal alignment
                va='bottom',                       # Vertical alignment
                fontsize=10,                       # Font size
                weight='bold',                     # Make text bold
                color='white',                     # Change the color to black for better visibility
                #0D85D8
                bbox=dict(facecolor=palette[0], edgecolor='none', boxstyle='round,pad=0.3', alpha = 0.85)  # Background settings
            )

        plt.savefig(f'figures/median_accuracy_scores_{size}.png', format='png', dpi=300)
        plt.show()

def plot_execution_time_scatterplot(csv_path):
    data = pd.read_csv(csv_path)
    data['Execution Times'] = data['Execution Times'].apply(ast.literal_eval)
    data['Mean Execution Time'] = data['Execution Times'].apply(lambda scores: sum(scores) / len(scores))
   
    # Estimate population variance with ddof = 1 correction
    data['Variance of Mean Execution Time'] = data['Execution Times'].apply(lambda x: np.var(x, ddof=1) / len(x))
    data['Standard Deviation of Mean Execution Time'] = data['Variance of Mean Execution Time'].apply(lambda x: np.sqrt(x))
    data.to_csv('decryption_trials_processed.csv', index = False)

    # Compute line of best fit
    x = data['Length']
    y = data['Mean Execution Time']
    coefficients = np.polyfit(x, y, 1)  # Degree 1 for a straight line
    slope, intercept = coefficients
    line_of_best_fit = slope * x + intercept

    dot_color = '#6A0DAD'  
    background_color = '#E6DFF5' 
    line_color = '#4B0082' 

    plt.figure(figsize=(10, 6))

    # Create the scatter plot with error bars
    plt.errorbar(
        data['Length'],  # x-axis: length column
        data['Mean Execution Time'],  # y-axis: mean execution time column
        yerr=data['Standard Deviation of Mean Execution Time'],  # Error bars on y-axis
        fmt='o',  # 'o' for circular markers
        color=dot_color,  # Color of the points
        ecolor='gray',  # Color of the error bars
        capsize=5,  # Size of the error bar caps
        linestyle='None',  # No connecting lines between points
        label='Mean Execution Time'
    )
    
    # Add best fit line
    plt.plot(x, line_of_best_fit, color=line_color, label='Line of Best Fit', linestyle='dashed')

    plt.xlabel('Passage Length (Word Count)', fontsize=12, weight='bold', color=dot_color)
    plt.ylabel('Mean Execution Time (seconds)', fontsize=12, weight='bold', color=dot_color)
    plt.title('Passage Length vs. Mean Execution Time', 
            fontsize=14, weight='bold', color=dot_color, pad=20)
    
    # Add the equation of the line
    equation_text = f'Line: y = {slope:.2f}x + {intercept:.2f}'
    plt.text(
        0.05, 0.95, equation_text, fontsize=12, weight='bold', color=line_color, 
        transform=plt.gca().transAxes, 
        ha='left', va='top', bbox=dict(facecolor=background_color, alpha=0.7, edgecolor='none')
    )

    # Add a pastel background color
    plt.gca().set_facecolor(background_color)

    # Add grid lines
    plt.grid(axis='y', color='gray', linestyle='--', alpha=0.7)

    # Customize tick labels
    plt.xticks(fontsize=10, color=dot_color)
    plt.yticks(fontsize=10, color=dot_color)


    plt.legend(fontsize=10, loc='lower right')
    plt.tight_layout()

    plt.savefig(f'figures/length_vs_execution_time.png', format='png', dpi=300)
    plt.show()

def plot_length_vs_accuracy_scatterplot(csv_path):
    data = pd.read_csv(csv_path)
    data['Accuracy Scores'] = data['Accuracy Scores'].apply(ast.literal_eval)
    # data['Mean Accuracy Score'] = data['Mean Accuracy Score'].apply(ast.literal_eval)
    # data['Standard Deviation of Mean Accuracy'] = data['Standard Deviation of Mean Accuracy'].apply(ast.literal_eval)

    dot_color = '#4682B4'  # Steel Blue (rich and vibrant)
    background_color = '#B0E0E6'  # Powder Blue (soft but with contrast)

    x = data['Length']
    y = data['Mean Accuracy Score']
    correlation_coefficient = np.corrcoef(x, y)[0, 1]

    plt.figure(figsize=(10, 6))

    # Create the scatter plot with error bars
    plt.errorbar(
        data['Length'],  # x-axis: length column
        data['Mean Accuracy Score'],  # y-axis: mean accuracy column
        yerr=data['Standard Deviation of Mean Accuracy'],  # Error bars on y-axis
        fmt='o',  # 'o' for circular markers
        color=dot_color,  # Color of the points
        ecolor='#1E3A5F',  # Color of the error bars
        capsize=5,  # Size of the error bar caps
        linestyle='None',  # No connecting lines between points
        label='Mean Decryption Accuracy'
    )

    plt.xlabel('Passage Length (Word Count)', fontsize=12, weight='bold', color=dot_color)
    plt.ylabel('Mean Decryption Accuracy Score', fontsize=12, weight='bold', color=dot_color)
    plt.title('Passage Length vs. Mean Accuracy Score', 
            fontsize=14, weight='bold', color=dot_color, pad=20)

    # Add a pastel background color
    plt.gca().set_facecolor(background_color)

    # Add grid lines
    plt.grid(axis='y', color='gray', linestyle='--', alpha=0.7)

    # Customize tick labels
    plt.xticks(fontsize=10, color=dot_color)
    plt.yticks(fontsize=10, color=dot_color)

    # Add correlation coefficient on plot
    plt.text(
        0.05, 0.95, 
        f'Correlation Coefficient: {correlation_coefficient:.2f}', 
        fontsize=10, 
        color='black', 
        weight='bold', 
        transform=plt.gca().transAxes, 
        bbox=dict(facecolor=background_color, edgecolor='none', alpha=0.5, boxstyle='round,pad=0.3')
    )

    plt.legend(fontsize=10, loc='lower right')
    plt.tight_layout()

    plt.savefig(f'figures/length_vs_accuracy.png', format='png', dpi=300)
    plt.show()


def main():
    # clean_passage_name("Pride_and_Prejudice.txt")
    # visualize_convergences()
    # generate_trial_data()
    data_path = f"{BASE_PATH}/decryption_trials_processed.csv"
    # plot_mean_accuracy_scores(data_path)
    # plot_median_accuracy_scores(data_path)
    # plot_execution_time_scatterplot(data_path)
    plot_length_vs_accuracy_scatterplot(data_path)

if __name__ == "__main__":
    main()
