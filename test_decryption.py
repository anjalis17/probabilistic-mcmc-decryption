from decryption import generate_random_key, encrypt_text, decrypt_text, compute_log_likelihood, predict_encryption_key, get_word_frequencies, generate_initial_key
import matplotlib.pyplot as plt
import time

def test_encrypt_decrypt():
    num_iterations = 10
    for i in range(10):
        encryption_map = generate_random_key()
        plaintext = "  HelloOoOo my name is Anjali! I <3 CS109;"
        encrypted_text = encrypt_text(plaintext, encryption_map)
        assert decrypt_text(encrypted_text, encryption_map) == plaintext.upper().strip()

def test_compute_likelihood():
    encryption_map = generate_random_key()
    word_prob_dict = get_word_frequencies()

    # Excerpt taken from https://mlpp.pressbooks.pub/introphil/chapter/from-the-medieval-era/
    plaintext = """Aristotle, 384 - 322 BCE, was a student of Plato and teacher of Alexander the Great. 
    He wrote on physics, poetry, theater, music, logic, rhetoric, politics, government, ethics, biology and zoology. 
    Together with Plato and Socrates, Aristotle is one of the most important writers and people to be found in Western philosophy. 
    Aristotle himself described his subject matter in this collection of his work in a variety of ways: as beginning philosophy, or the study of being, 
    or sometimes simply as wisdom.  Metaphysics is a title that was attached to this work long after the time of Aristotle, and it simply refers to a 
    collection of work intended for use in the study of philosophy."""

    encrypted_text = encrypt_text(plaintext, encryption_map)
    decrypted = decrypt_text(encrypted_text, encryption_map)
    print(decrypted)
    assert decrypted == plaintext.upper().strip()

    likelihood_right_mapping = compute_log_likelihood(encrypted_text, encryption_map, word_prob_dict)
    print(likelihood_right_mapping)
    count = 0 
    for i in range(5000):
        new_map = generate_random_key()
        likelihood_wrong_mapping = compute_log_likelihood(encrypted_text, new_map, word_prob_dict)
        print(likelihood_wrong_mapping)
        if likelihood_wrong_mapping > likelihood_right_mapping: count += 1
    # Expect count to be 0 -- no randomly generated mapping should be better than the correct mapping
    assert count == 0

def test_predict():
    word_prob_dict = get_word_frequencies()
    passage_path = "/Users/anjali/Library/CloudStorage/OneDrive-Stanford/Sophomore Year/Fall 2024/CS 109/Challenge Project/long_passages/New_York_Times_Politics.txt"

    encryption_map = generate_random_key()
    plaintext = read_text_file(passage_path)
    encrypted_text = encrypt_text(plaintext, encryption_map)
    print(encrypted_text)

    likelihood_right_mapping = compute_log_likelihood(encrypted_text, encryption_map, word_prob_dict)
    # print(f"Correct Mapping Likelihood: {likelihood_right_mapping}")

    # Core Algorithm -- predicting correct encryption key ('mapping_guess') from encrypted passage
    mapping_guess, likelihoods = predict_encryption_key(encrypted_text)
    decrypted = decrypt_text(encrypted_text, mapping_guess)
    print(decrypted)
    print(f"Encryption Key Prediction: {mapping_guess}")
    print(f"Decryption Accuracy: {compute_decryption_accuracy(plaintext, decrypted)}")
    visualize_convergence(likelihoods, likelihood_right_mapping)

def visualize_convergence(likelihoods, expected_likelihood):
    plt.plot(likelihoods)
    plt.xlabel('Iteration')
    plt.ylabel('Log Likelihood')
    plt.title('Decryption Convergence')
    # Draw a horizontal line at y = expected_likelihood
    plt.axhline(y=expected_likelihood, color='r', linestyle='--', label=f'Expected Likelihood = {expected_likelihood}')
    plt.show()

def test_starting_key():
    encryption_map = generate_random_key()
    passage_path = "/Users/anjali/Library/CloudStorage/OneDrive-Stanford/Sophomore Year/Fall 2024/CS 109/Challenge Project/medium_passages/Pride_and_Prejudice.txt"
    plaintext = read_text_file(passage_path)
    encrypted_text = encrypt_text(plaintext, encryption_map)
    print(encryption_map)
    start_key = generate_initial_key(encrypted_text)
    print(start_key)

def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    print(content)  # Prints the content of the file as a string
    return content

def compute_decryption_accuracy(plaintext, decrypted):
    truth = plaintext.upper().strip()
    correct = 0
    count = 0
    for true, pred in zip(truth, decrypted):
        if true.isalpha():
            count += 1
            if true == pred:
                correct += 1
            else:
                print(f"true: {true}, pred: {pred}")
    accuracy_score = correct / count
    # print(f"ACCURACY SCORE: {accuracy_score}")
    return accuracy_score


def conduct_decryption_trials(passage_path):
    word_prob_dict = get_word_frequencies()
    plaintext = read_text_file(passage_path)
    num_repetitions = 10
    accuracy_scores = []
    execution_times = []

    for i in range(num_repetitions):
        print(f"Starting iteration {i}")
        encryption_map = generate_random_key()
        encrypted_text = encrypt_text(plaintext, encryption_map)
        likelihood_right_mapping = compute_log_likelihood(encrypted_text, encryption_map, word_prob_dict)
        print(f"Correct Mapping Likelihood: {likelihood_right_mapping}")

        start_time = time.time()
        mapping_guess, likelihoods = predict_encryption_key(encrypted_text)
        decrypted = decrypt_text(encrypted_text, mapping_guess)
        end_time = time.time()
        # print our encryption key guess
        # for key in sorted(mapping_guess.keys()):
        #     print(f"{key}: {mapping_guess[key]}")
        # print(encryption_map)
        accuracy_score = compute_decryption_accuracy(plaintext, decrypted)
        accuracy_scores.append(accuracy_score)

        execution_time = end_time - start_time
        execution_times.append(execution_time)
    print(accuracy_scores)
    print(execution_times)
    return accuracy_scores, execution_times


# 100 - 300 short
# 500-800 medium
# 1000+ long
def main():
    # test_encrypt_decrypt()
    # test_compute_likelihood()
    test_predict()
    # test_starting_key()

    # passage_path = "/Users/anjali/Library/CloudStorage/OneDrive-Stanford/Sophomore Year/Fall 2024/CS 109/Challenge Project/medium_passages/Harry_Potter.txt"
    # conduct_decryption_trials(passage_path)

if __name__ == "__main__":
    main()
        