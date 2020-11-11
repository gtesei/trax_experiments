import pickle
import string
import ast
import numpy as np
import trax 
from trax.supervised import decoding
import textwrap 

wrapper = textwrap.TextWrapper(width=70)

########################
###   DATA 
########################

example_jsons = list(map(ast.literal_eval, open('data.txt')))

natural_language_texts = [example_json['text'] for example_json in example_jsons]

PAD, EOS, UNK = 0, 1, 2

def detokenize(np_array):
    return trax.data.detokenize(
        np_array,
        vocab_type='sentencepiece',
        vocab_file='sentencepiece.model',
        vocab_dir='.')


def tokenize(s):
    return next(trax.data.tokenize(
        iter([s]),
        vocab_type='sentencepiece',
        vocab_file='sentencepiece.model',
        vocab_dir='.'))
 
    
vocab_size = trax.data.vocab_size(
    vocab_type='sentencepiece',
    vocab_file='sentencepiece.model',
    vocab_dir='.')


def get_sentinels(vocab_size, display=False):
    sentinels = {}
    for i, char in enumerate(reversed(string.ascii_letters), 1):
        decoded_text = detokenize([vocab_size - i]) 
        # Sentinels, ex: <Z> - <a>
        sentinels[decoded_text] = f'<{char}>'    
        if display:
            print(f'The sentinel is <{char}> and the decoded token is:', decoded_text)
    return sentinels


sentinels = get_sentinels(vocab_size, display=False)    


def pretty_decode(encoded_str_list, sentinels=sentinels):
    # If already a string, just do the replacements.
    if isinstance(encoded_str_list, (str, bytes)):
        for token, char in sentinels.items():
            encoded_str_list = encoded_str_list.replace(token, char)
        return encoded_str_list
  
    # We need to decode and then prettyfy it.
    return pretty_decode(detokenize(encoded_str_list))


inputs_targets_pairs = []

# here you are reading already computed input/target pairs from a file
with open ('inputs_targets_pairs_file.txt', 'rb') as fp:
    inputs_targets_pairs = pickle.load(fp)  


def display_input_target_pairs(inputs_targets_pairs):
    for i, inp_tgt_pair in enumerate(inputs_targets_pairs, 1):
        inps, tgts = inp_tgt_pair
        inps, tgts = pretty_decode(inps), pretty_decode(tgts)
        print(f'[{i}]\n'
              f'inputs:\n{wrapper.fill(text=inps)}\n\n'
              f'targets:\n{wrapper.fill(text=tgts)}\n\n\n\n')
    
display_input_target_pairs(inputs_targets_pairs)    

########################
###   TRANSFORMER  
########################

# Initializing the model
model = trax.models.Transformer(
    d_ff = 4096,
    d_model = 1024,
    max_len = 2048,
    n_heads = 16,
    dropout = 0.1,
    input_vocab_size = 32000,
    n_encoder_layers = 24,
    n_decoder_layers = 24,
    mode='predict')

shape11 = trax.shapes.ShapeDtype((1, 1), dtype=np.int32)  # Needed in predict mode.
model.init_from_file('model.pkl.gz', weights_only=True, input_signature=(shape11, shape11))


# using the 3rd example
c4_input = inputs_targets_pairs[2][0]
c4_target = inputs_targets_pairs[2][1]

print('pretty_decoded input: \n\n', pretty_decode(c4_input))
print('\npretty_decoded target: \n\n', pretty_decode(c4_target))
print('\nc4_input:\n\n', c4_input)
print('\nc4_target:\n\n', c4_target)
print(len(c4_target))
print(len(pretty_decode(c4_target)))


# Temperature is a parameter for sampling.
#   # * 0.0: same as argmax, always pick the most probable token
#   # * 1.0: sampling from the distribution (can sometimes say random things)
#   # * values inbetween can trade off diversity and quality, try it out!
output = decoding.autoregressive_sample(model, inputs=np.array(c4_input)[None, :],
                                        temperature=0.0, max_length=5) # originally max_length = 50
print(wrapper.fill(pretty_decode(output[0])))




