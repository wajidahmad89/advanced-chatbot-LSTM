


"""
## Consolidated
"""



import pickle
import numpy as np




with open('qanda_raw.txt') as f:
    lines = f.readlines()
    


#type(lines)


list_1 = []
for x in lines:
    while x != '\n':
        z = list(x)
        list_1.append(z)
        break


#len(list_1)


#list_1



for x in list_1:
    
    x.remove('\n')


#list_1


list_2 = []
for x in list_1:
    qanda_join = ''.join(x)
    list_2.append(qanda_join)


#list_2


#len(list_2)


#type(list_2)



list_3 = []

for x in list_2:
    y = list(x.split(" "))
    list_3.append(y)


#list_3


#len(list_3)



the_list = []

count = 0

while count < len(list_3):
    the_t = (list_3[count], list_3[count+1])
    the_list.append(the_t)
    count += 2



#the_list



#filename = 'qanda_output'
#outfile = open(filename,'wb')


#pickle.dump(the_list,outfile)
#outfile.close()



#with open('qanda_output', 'rb') as f:
#    train_data = pickle.load(f)



train_data = the_list


#print(train_data)



vocab = set()
for question, answer in train_data:
    
    vocab = vocab.union(set(question))
    vocab = vocab.union(set(answer))


vocab_list = list(vocab)
vocab_list.sort()



with open('the_vocabularly.txt', mode="w") as f:
    f.write('\n')
    for x in vocab_list:
        y = str(x)
        f.write(y + '\n')



vocab_len = len(vocab) + 1


all_data = train_data


all_questions_lens = [len(data[0]) for data in all_data]


max_question_len = (max(all_questions_lens))


max_answer_len = max([len(data[1]) for data in all_data])




"""
# Vectorizing the data
"""



from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer


tokenizer = Tokenizer(filters = [])
tokenizer.fit_on_texts(vocab)


#tokenizer.word_index


vocab_low = vocab
vocab_low = [x.lower() for x in vocab_low]


list1 = sorted(vocab_low)
#print(list1)


list2 = []
x = 1
for x in range(len(list1)):
    list2.append(x+1)
#print(list2)


my_dict = dict(zip(list1, list2))


#print(my_dict)


tokenizer.word_index = my_dict


#print(tokenizer.word_index)


train_question_text = []
train_answers = []


for question,answer in train_data:
    
    train_question_text.append(question) 
    train_answers.append(answer)


train_question_seq = tokenizer.texts_to_sequences(train_question_text)


#print(train_question_text)


#print(train_question_seq)


#print(tokenizer.word_index)


#Create a function for vectorizing the stories, questions and answers:
def vectorize_stories(data,word_index = tokenizer.word_index, max_question_len = max_question_len, max_answer_len = max_answer_len):
    #vectorized stories:
    X = []
    #vectorized questions:

    #vectorized answers:
    Y = []
    
    for question, answer in data:
        #Getting indexes for each word in the story
        
        #Getting indexes for each word in the story
        x = [word_index[word.lower()] for word in question]
        #For the answers
        y = np.zeros(len(word_index) + 1) #Index 0 Reserved when padding the sequences
        for word in answer:
            y[word_index[word.lower()]] = 1
        
        X.append(x)
        
        Y.append(y)
        
        
        
    #Now we have to pad these sequences:
    return(pad_sequences(X, maxlen=max_question_len), np.array(Y))


#print(vectorize_stories(train_data))


questions_train, answers_train = vectorize_stories(train_data)


#questions_train[11]


#train_question_text[11]


#train_question_seq[11]




"""
# Building the Network
"""


#Imports
from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers import Input, Activation, Dense, Permute, Dropout, add, dot, concatenate, LSTM


input_sequence = Input((max_question_len,))



input_encoder_m = Sequential()
input_encoder_m.add(Embedding(input_dim=vocab_len,output_dim = 64)) #From paper
input_encoder_m.add(Dropout(0.3))


input_encoder_c = Sequential()
input_encoder_c.add(Embedding(input_dim=vocab_len,output_dim = max_question_len)) #From paper
input_encoder_c.add(Dropout(0.3))


question_encoder = Sequential()
question_encoder.add(Embedding(input_dim=vocab_len,output_dim = 64,input_length=max_question_len)) #From paper
question_encoder.add(Dropout(0.3))


input_encoded_m = input_encoder_m(input_sequence)
input_encoded_c = input_encoder_c(input_sequence)

question_encoded = question_encoder(input_sequence)


#print(input_encoded_m.shape)


match = dot([input_encoded_m,question_encoded], axes = (2,2))
match = Activation('softmax')(match)


response = add([match,input_encoded_c])
response = Permute((2,1))(response) #Permute Layer: permutes dimensions of input



answer = concatenate([response, question_encoded])


#answer


answer = LSTM(32)(answer)


answer = Dropout(0.5)(answer)
#Output layer:
answer = Dense(vocab_len)(answer) #Output shape: (Samples, Vocab_size) #Yes or no and all 0s



answer = Activation('softmax')(answer)



model = Model([input_sequence], answer)


model.compile(optimizer='rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])


#model.summary()


history = model.fit([questions_train],answers_train, batch_size = 32, epochs = 1000)


#filename = 'z_ichatbot_1000_epochs.h5'



#model.load_weights('z_ichatbot_1000_epochs.h5')


# Cut, copy, paste, right-click disabled

# Creating GUI with tkinter
import tkinter
from tkinter import *


def send():
    msg = EntryBox.get("1.0", 'end-1c').strip()
    EntryBox.delete("0.0", END)

    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12))

        res_list = []

        split_msg = msg.split()

        the_msg = []

        for x in split_msg:
            if x in vocab:
                the_msg.append(x)

        # print(the_msg)

        my_data = [(the_msg, [''])]

        my_ques, my_ans = vectorize_stories(my_data)

        pred_results = model.predict(([my_ques]))
        pred_results[0]

        highest_values = pred_results[0].argsort()[-6:][::-1]

        perc_add = 0

        for x in highest_values:
            for key, val in tokenizer.word_index.items():
                if val == x:
                    k = key

            perc_add += pred_results[0][x]

            if perc_add > 0.99:
                break

            res_list.append(k)

            # res_list.append('---')
            # res_list.append(pred_results[0][x])

        # print(perc_add)

        res = ' '.join([str(elem) for elem in res_list])

        ChatLog.insert(END, "Bot: " + res + '\n\n')
        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)


base = Tk()
base.title("www.therexybot.com")
base.geometry("400x500")
base.resizable(width=FALSE, height=FALSE)

# Create Chat window
ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Arial", )

ChatLog.config(state=DISABLED)

ChatLog.bind('<Control-x>', lambda e: 'break') #disable cut
ChatLog.bind('<Control-c>', lambda e: 'break') #disable copy
ChatLog.bind('<Control-v>', lambda e: 'break') #disable paste
ChatLog.bind('<Button-3>', lambda e: 'break')  #disable right-click


# Bind scrollbar to Chat window
scrollbar = Scrollbar(base, command=ChatLog.yview)
ChatLog['yscrollcommand'] = scrollbar.set

# Create Button to send message
SendButton = Button(base, font=("Verdana", 12, 'bold'), text="Send", width="12", height=5,
                    bd=0, bg="#32de97", activebackground="#3c9d9b", fg='#ffffff',
                    command=send)

# Create the box to enter message
EntryBox = Text(base, bd=0, bg="white", width="29", height="5", font="Arial")
# EntryBox.bind("<Return>", send)


# Place all components on the screen
scrollbar.place(x=376, y=6, height=386)
ChatLog.place(x=6, y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
SendButton.place(x=6, y=401, height=90)

base.mainloop()











