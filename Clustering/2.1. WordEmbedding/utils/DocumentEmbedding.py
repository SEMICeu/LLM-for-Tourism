
import torch # For data handling
import pandas as pd # For data handling

from transformers import BertTokenizer, BertModel # For LLM models and tokenizer
import nltk


def DocEmbedding(TokenizedText, tokenizer, model, Bert=True):
    """
    DocEmbedding takes a tokenized text as input and returns its embedding vector based on te provided tokenizer and embedding model

    :param TokenizedText: List of tokens
    :param tokenizer: Tokenizer object
    :param model: Embedding model
    :param Bert: Boolean indicating whether the model is BERT or not
    :return: A tensor of size 1x768 containing the embedding of the text
    """
    
    print(Bert)
    # Define the padding token in function of the model
    if Bert == True:
           pad = '[PAD]'
           seg = 1
    else:
           pad = '<pad>'
           seg = 0

    # Add padding to the text to have the size of the documents equal to 512
    PaddedTokens = TokenizedText + [pad for _ in range(512-len(TokenizedText))]

    # Map the token strings to their vocabulary indeces.
    IndexedTokens = tokenizer.convert_tokens_to_ids(PaddedTokens)

    # Display the words with their indeces.
    for tup in zip(TokenizedText, IndexedTokens):
            print('{:<12} {:>6,}'.format(tup[0], tup[1]))

    #Attention mask
    AttnMask = [ 1 if token != pad else 0 for token in PaddedTokens  ]
    print(len(AttnMask))

    # Mark each of the tokens as belonging to sentence "1".
    SegmentsIds = [seg] * len(AttnMask)
    print (SegmentsIds)

    # Convert inputs to PyTorch tensors
    TokensTensors = torch.tensor([IndexedTokens])
    SegmentsTensors = torch.tensor([SegmentsIds])
    AttentionTensors = torch.tensor([AttnMask])

    # Run the text through BERT, and collect all of the hidden states produced
    # from all 12 layers. 
    with torch.no_grad():

            outputs = model(TokensTensors, attention_mask = AttentionTensors, token_type_ids = SegmentsTensors)

            # Evaluating the model will return a different number of objects based on 
            # how it's  configured in the `from_pretrained` call earlier. In this case, 
            # becase we set `output_hidden_states = True`, the third item will be the 
            # hidden states from all layers. See the documentation for more details:
            # https://huggingface.co/transformers/model_doc/bert.html#bertmodel
            HiddenStates = outputs[2]


            # Getting document vectors by averaging the second to last hidden layer of each token
            # `HiddenStates` has shape [13 x 1 x 22 x 768]

            # `TokenVecs` is a tensor with shape [22 x 768]
            TokenVecs = HiddenStates[-2][0]

            # Calculate the average of all 22 token vectors.
            DocumentEmbedding = torch.mean(TokenVecs, dim=0)

    return DocumentEmbedding


def pledgeEmbedding(documents, tokenizer, model, Bert=True):
    """
    pledgeEmbedding takes a set of text as input and returns their embedding vectors based on the provided tokenizer and embedding model

    :param documents: List of strings
    :param tokenizer: Tokenizer object
    :param model: Embedding model
    :param Bert: Boolean indicating whether the model is BERT or not
    :return: List of tensors of size 1x768 representing the embeddings of the texts
    """
    
    pledgeEmbedding = []
    
    # Looping over the different documents
    for doc in documents:
        text = doc
        
        # Define the appropriate start and end tokens based on the model type
        if Bert==True: 
              cls = "[CLS] "
              sep = " [SEP]"
              bert = True

        else: 
              cls = "<s> "
              sep = " </s>"
              bert = False
        
        # Add the special tokens.
        MarkedText = cls + text + sep
        
        # Split the sentence into tokens.
        TokenizedText = tokenizer.tokenize(MarkedText)
        
        # When the text is longer than 512 (context window of BERT), split the text in sentences
        if len(TokenizedText) > 512:
                print("Too long")

                # Tokenize on sentences
                sentences = nltk.sent_tokenize(text)

                text1 = ""
                sentEmbedding = []

                # Looping over the sentences. Concatenate the sentences as long as their length is below 512
                for sent in sentences: 
                        
                        MarkedText = cls + text1 + sent + sep
        
                        # Split the sentence into tokens.
                        TokenizedText = tokenizer.tokenize(MarkedText)
                        
                         
                        if len(TokenizedText) < 512:
                                text1 = text1 + sent
                        else:
                                MarkedText = cls + text1 + sep       
                                # Split the sentence into tokens.
                                TokenizedText = tokenizer.tokenize(MarkedText)

                                # Obtain the sentences embedding and start a new sentence
                                sentEmbedding.append(DocEmbedding(TokenizedText, tokenizer, model, Bert=bert))                                                  
                                text1 = ""
                
                # The document embedding is then obtained by taking the average of all the sentences' embeddings
                DocumentEmbedding = torch.mean(torch.cat(tuple(sentEmbedding)).view(len(sentEmbedding),768), dim = 0)

        else:              
                DocumentEmbedding = DocEmbedding(TokenizedText, tokenizer, model, Bert = bert)

        pledgeEmbedding.append(DocumentEmbedding.tolist())

    return pledgeEmbedding