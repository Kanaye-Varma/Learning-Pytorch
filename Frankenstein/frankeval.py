from frankensteinmodeller import model, save_to_path, unique_words, onehotencode, embed_encode
import torch 
import sys, random
import numpy as np

def main(words: int, randomness_index: int, starter: list[str]): 
    model.load_state_dict(torch.load(save_to_path, weights_only=True))

    model.eval()
    with torch.inference_mode():
        states = None
        pred, states = model(torch.tensor(embed_encode(starter)), states)
          
        counter = 0
        last_idx = -1
        while counter < words:
            counter += 1

            word = torch.tensor(embed_encode([starter[-1]]))
            pred, states = model(word, states)
            pred = pred.reshape((-1))
            if (last_idx > 0): pred[last_idx] /= 1.5

            pred = torch.softmax(pred, dim=0)

            _, indices = torch.topk(pred, randomness_index)
            # idx = torch.argmax(pred)
            # if random.randint(1,2) == 1:
            idx = np.random.choice(indices.numpy().squeeze())

            last_idx = idx
            starter.append(unique_words[idx])
        
        output = "" 
        caps = False
        for word in starter: 
            if word in output and random.randint(1, 2) == 1:
                output += word + '...\n'
                continue
            if caps:
                output += word.title()
                caps = False 
            else:
                output += word
            if random.randint(1, 10) == 1:
                output += '.\n'
                caps = True
            else: output += " "
        print(output)


if __name__ == '__main__':
    words = 0
    randomness = 0
    phrase = ""
    if len(sys.argv) < 4:
        print("Correct usage: python frankeval.py [number of words] [randomness index] [starting phrase]")
        sys.exit()
    try:
        words = int(sys.argv[1])
        assert(words > 0)
        randomness = int(sys.argv[2])
        assert(randomness > 0)
    except:
        print("Number of words and randomness index must both be positive integers")
        sys.exit()
    
    for word in sys.argv[3:]:
        if not (word.lower().strip() in unique_words):
            print("Uh oh! You can only use words from the Frankenstein novel.")
            print(f"Remove, replace or rewrite the word: {word}")
            print("Hint: ensure to remove all punctuation")
            sys.exit() 
    main(words, randomness, [word.lower().strip() for word in sys.argv[3:]])
    