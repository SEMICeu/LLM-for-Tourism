
import torch # For data handling
import pandas as pd # For data handling

from transformers import BertTokenizer, BertModel # For LLM models and tokenizer
import nltk


def DocEmbedding(tokenized_text, tokenizer, model):
    
    padded_tokens = tokenized_text + ['[PAD]' for _ in range(512-len(tokenized_text))]

    # Map the token strings to their vocabulary indeces.
    indexed_tokens = tokenizer.convert_tokens_to_ids(padded_tokens)

    # Display the words with their indeces.
    for tup in zip(tokenized_text, indexed_tokens):
            print('{:<12} {:>6,}'.format(tup[0], tup[1]))

    #Attention mask
    attn_mask = [ 1 if token != '[PAD]' else 0 for token in padded_tokens  ]
    
    # Mark each of the tokens as belonging to sentence "1".
    segments_ids = [1] * len(attn_mask)
    print (segments_ids)

    
    
    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    attention_tensors = torch.tensor([attn_mask])

    # Run the text through BERT, and collect all of the hidden states produced
    # from all 12 layers. 
    with torch.no_grad():

            outputs = model(tokens_tensor, attention_mask = attention_tensors, token_type_ids = segments_tensors)

            # Evaluating the model will return a different number of objects based on 
            # how it's  configured in the `from_pretrained` call earlier. In this case, 
            # becase we set `output_hidden_states = True`, the third item will be the 
            # hidden states from all layers. See the documentation for more details:
            # https://huggingface.co/transformers/model_doc/bert.html#bertmodel
            hidden_states = outputs[2]


            # Getting document vectors by averaging the second to last hidden layer of each token
            # `hidden_states` has shape [13 x 1 x 22 x 768]

            # `token_vecs` is a tensor with shape [22 x 768]
            token_vecs = hidden_states[-2][0]

            # Calculate the average of all 22 token vectors.
            document_embedding = torch.mean(token_vecs, dim=0)

    return document_embedding


def pledgeEmbedding(documents, tokenizer, model):
    
    pledgeEmbedding = []

    for doc in documents:
        text = doc

            # Add the special tokens.
        marked_text = "[CLS] " + text + " [SEP]"
        
        # Split the sentence into tokens.
        tokenized_text = tokenizer.tokenize(marked_text)
        

        if len(tokenized_text) > 512:
                #print(len(tokenized_text))
                print("Too long")

                sentences = nltk.sent_tokenize(text)

                text1 = ""
                sentEmbedding = []

                for sent in sentences: 
                        
                        marked_text = "[CLS] " + text1 + sent + " [SEP]"
        
                        # Split the sentence into tokens.
                        tokenized_text = tokenizer.tokenize(marked_text)
                        
                        if len(tokenized_text) < 512:
                                text1 = text1 + sent
                        else:
                                marked_text = "[CLS] " + text1 + " [SEP]"       
                                # Split the sentence into tokens.
                                tokenized_text = tokenizer.tokenize(marked_text)
                                sentEmbedding.append(DocEmbedding(tokenized_text, tokenizer, model))                  
                                
                                text1 = ""
                
                document_embedding = torch.mean(torch.cat(tuple(sentEmbedding)).view(len(sentEmbedding),768), dim = 0)

        else:              
                document_embedding = DocEmbedding(tokenized_text)

        pledgeEmbedding.append(document_embedding.tolist())

    return pledgeEmbedding