import random
import json

import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"

def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    return "Unknown"

def bulk_response(text):
    sentences=text.split('.')
    sentences.pop()
    responses=[]
    for sentence in sentences:
        responses.append(get_response(sentence))
    return responses
if __name__ == "__main__":
    print("INPUT:")
    model001=['four_seater','petrol','low_range']
    model002=['six_seater','diesel','high_range']
    score0=0
    score1=0
    while True:
        # sentence = "do you use credit cards?"
        inp = input()
        if inp == "quit":
            break
        sentences=inp.split('.')
        sentences.pop()
        responses=[]
        for sentence in sentences:
            resp = get_response(sentence)
            responses.append(resp)

        for i in responses:
            print(i,end=' ')
            if i in model001:
                score0+=1
            if i in model002:
                score1+=1

    score0=score0/(score0+score1)*100
    score1=score1/(score1+score0)*100

    print("Model-001:",score0)
    print("Model-002:",score1)
        
        
