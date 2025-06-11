import torch 
from trying_lstm import save_to_path, unique_chars, model, onehotencode
import sys, random
import numpy as np

def main(chars: int, starter: str):
    model.load_state_dict(torch.load(save_to_path, weights_only=True))

    model.eval()
    with torch.inference_mode():
        states = None
        for ch in starter:
            x = onehotencode(ch)
            x = torch.tensor(x, dtype=torch.float32)
            pred, states = model(x, states)

        counter = 0
        while starter[-1] != '.' and counter < chars:
            counter += 1
            x = onehotencode(starter[-1])
            x = torch.tensor(x, dtype=torch.float32)
            pred, states = model(x, states)
            pred = torch.softmax(pred, dim=1)

            
            # random otherwise output is too deterministic and repeatedly outputs the same phrase
            values, indices = torch.topk(pred, 5)
            idx = torch.argmax(pred)
            if random.randint(1, 7) == 1:
                idx = np.random.choice(indices.numpy().squeeze())

            starter += unique_chars[idx]

        print(starter)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Correct usage: python lstm_eval.py [number of characters] [starting string]")
        sys.exit()
    

    chars = 0
    try:
        chars = int(sys.argv[1])
        assert(chars > 0)
    except:
        print("Number of characters must be a positive integer.")
        sys.exit()
    
    starter = ""
    for w in sys.argv[2:]: starter += (w + " ")
    main(chars, starter)