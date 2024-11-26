from decryption import generate_random_key, encrypt_text, decrypt_text, compute_log_likelihood, predict_encryption_key, get_word_frequencies, generate_initial_key
import matplotlib.pyplot as plt

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

# def test_predict():
#     encryption_map = generate_random_key()

#     # Excerpt taken from https://mlpp.pressbooks.pub/introphil/chapter/from-the-medieval-era/
#     plaintext = """Aristotle, 384 - 322 BCE, was a student of Plato and teacher of Alexander the Great. 
#     He wrote on physics, poetry, theater, music, logic, rhetoric, politics, government, ethics, biology and zoology. 
#     Together with Plato and Socrates, Aristotle is one of the most important writers and people to be found in Western philosophy. 
#     Aristotle himself described his subject matter in this collection of his work in a variety of ways: as beginning philosophy, or the study of being, 
#     or sometimes simply as wisdom.  Metaphysics is a title that was attached to this work long after the time of Aristotle, and it simply refers to a 
#     collection of work intended for use in the study of philosophy."""

#     encrypted_text = encrypt_text(plaintext, encryption_map)
#     mapping_guess = predict_encryption_key(encrypted_text)
#     decrypted = decrypt_text(encrypted_text, mapping_guess)
#     print(decrypted)
#     print(encryption_map)

def test_predict_longer():
    encryption_map = generate_random_key()
    word_prob_dict = get_word_frequencies()
    # plaintext = """Little is known about Aristotle's life. He was born in the city of Stagira in northern Greece during the Classical period. His father, Nicomachus, died when Aristotle was a child, and he was brought up by a guardian. At around eighteen years old, he joined Plato's Academy in Athens and remained there until the age of thirty seven (c. 347 BC). Shortly after Plato died, Aristotle left Athens and, at the request of Philip II of Macedon, tutored his son Alexander the Great beginning in 343 BC. He established a library in the Lyceum, which helped him to produce many of his hundreds of books on papyrus scrolls.
    #     Though Aristotle wrote many treatises and dialogues for publication, only around a third of his original output has survived, none of it intended for publication. Aristotle provided a complex synthesis of the various philosophies existing prior to him. His teachings and methods of inquiry have had a significant impact across the world, and remain a subject of contemporary philosophical discussion.
    #     Aristotle's views profoundly shaped medieval scholarship. The influence of his physical science extended from late antiquity and the Early Middle Ages into the Renaissance, and was not replaced systematically until the Enlightenment and theories such as classical mechanics were developed. He influenced Judeo-Islamic philosophies during the Middle Ages, as well as Christian theology, especially the Neoplatonism of the Early Church and the scholastic tradition of the Catholic Church.
    #     Aristotle was revered among medieval Muslim scholars as "The First Teacher", and among medieval Christians like Thomas Aquinas as simply "The Philosopher", while the poet Dante called him "the master of those who know". His works contain the earliest known formal study of logic, and were studied by medieval scholars such as Peter Abelard and Jean Buridan. Aristotle's influence on logic continued well into the 19th century. In addition, his ethics, although always influential, gained renewed interest with the modern advent of virtue ethics."""
    
    passage_path = "/Users/anjali/Library/CloudStorage/OneDrive-Stanford/Sophomore Year/Fall 2024/CS 109/Challenge Project/long_passages/Harry_Potter.txt"
    plaintext = read_text_file(passage_path)
    encrypted_text = encrypt_text(plaintext, encryption_map)
    likelihood_right_mapping = compute_log_likelihood(encrypted_text, encryption_map, word_prob_dict)
    print(f"CORRECT MAPPING LIKELIHOOD: {likelihood_right_mapping}")

    # print(encrypted_text)
    mapping_guess, likelihoods = predict_encryption_key(encrypted_text)
    decrypted = decrypt_text(encrypted_text, mapping_guess)
    print(decrypted)
    print(encryption_map)
    print(likelihoods[:5])
    visualize_convergence(likelihoods, likelihood_right_mapping)
    # print(mapping_guess)

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
    word_prob_dict = get_word_frequencies()
    plaintext = """And when the party entered the assembly room it consisted only of five all together, – Mr. Bingley, his two sisters, the husband of the eldest, and another young man.

Mr. Bingley was good-looking and gentlemanlike; he had a pleasant countenance, and easy, unaffected manners. His sisters were fine women, with an air of decided fashion. His brother-in-law, Mr. Hurst, merely looked the gentleman; but his friend Mr. Darcy soon drew the attention of the room by his fine, tall person, handsome features, noble mien, and the report which was in general circulation within five minutes after his entrance, of his having ten thousand a-year. The gentlemen pronounced him to be a fine figure of a man, the ladies declared he was much handsomer than Mr. Bingley, and he was looked at with great admiration for about half the evening, till his manners gave a disgust which turned the tide of his popularity; for he was discovered to be proud; to be above his company, and above being pleased; and not all his large estate in Derbyshire could then save him from having a most forbidding, disagreeable countenance, and being unworthy to be compared with his friend.

Mr. Bingley had soon made himself acquainted with all the principal people in the room; he was lively and unreserved, danced every dance, was angry that the ball closed so early, and talked of giving one himself at Netherfield. Such amiable qualities must speak for themselves. What a contrast between him and his friend! Mr. Darcy danced only once with Mrs. Hurst and once with Miss Bingley, declined being introduced to any other lady, and spent the rest of the evening in walking about the room, speaking occasionally to one of his own party. His character was decided. He was the proudest, most disagreeable man in the world, and everybody hoped that he would never come there again. Amongst the most violent against him was Mrs. Bennet, whose dislike of his general behaviour was sharpened into particular resentment by his having slighted one of her daughters.

Elizabeth Bennet had been obliged, by the scarcity of gentlemen, to sit down for two dances; and during part of that time, Mr. Darcy had been standing near enough for her to overhear a conversation between him and Mr. Bingley, who came from the dance for a few minutes, to press his friend to join in.

“Come, Darcy,” said he, “I must have you dance. I hate to see you standing about by yourself in this stupid manner. You had much better dance.”

“I certainly shall not. You know how I detest it, unless I am particularly acquainted with my partner. At such an assembly as this it would be insupportable. Your sisters are engaged, and there is not another woman in the room whom it would not be a punishment to me to stand up with.”

“I would not be so fastidious as you are,” cried Bingley, “for a kingdom! Upon my honour, I never met with so many pleasant girls in my life as I have this evening; and there are several of them you see uncommonly pretty.”

“You are dancing with the only handsome girl in the room,” said Mr. Darcy, looking at the eldest Miss Bennet.

# “Oh! she is the most beautiful creature I ever beheld! But there is one of her sisters sitting down just behind you, who is very pretty, and I dare say very agreeable. Do let me ask my partner to introduce you.”"""
    encrypted_text = encrypt_text(plaintext, encryption_map)
    print(encryption_map)
    start_key = generate_initial_key(encrypted_text)
    print(start_key)

def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    print(content)  # Prints the content of the file as a string
    return content


#100 - 300 short
# 500-800 medium
# 1000+ long
def main():
    # test_encrypt_decrypt()
    # test_compute_likelihood()
    
    # test_predict()
    test_predict_longer()
    # test_starting_key()

if __name__ == "__main__":
    main()
        