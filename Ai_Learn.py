import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Đọc dữ liệu từ file CSV
data = pd.read_csv('AI.csv')

# Lấy câu hỏi và câu trả lời từ dữ liệu
questions = data['question'].tolist()
answers = ["<start> " + ans + " <end>" for ans in data['answer'].tolist()]

# Tạo tokenizer và mã hóa các câu
tokenizer = Tokenizer()
tokenizer.fit_on_texts(questions + answers)

# Biến đổi các câu thành chuỗi các số
question_sequences = tokenizer.texts_to_sequences(questions)
answer_sequences = tokenizer.texts_to_sequences(answers)

# Tìm kích thước tối đa của các câu
max_question_len = max([len(seq) for seq in question_sequences])
max_answer_len = max([len(seq) for seq in answer_sequences])

# Padding các chuỗi để có cùng chiều dài
question_sequences = pad_sequences(question_sequences, maxlen=max_question_len, padding='post')
answer_sequences = pad_sequences(answer_sequences, maxlen=max_answer_len, padding='post')

# Tạo từ điển ngược để giải mã
word_index = tokenizer.word_index
index_word = {v: k for k, v in word_index.items()}
vocab_size = len(word_index) + 1

# Tạo mô hình Seq2Seq
embedding_dim = 50
latent_dim = 256

# Bộ mã hóa
encoder_inputs = Input(shape=(max_question_len,))
encoder_embedding = Embedding(vocab_size, embedding_dim)(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

# Bộ giải mã
decoder_inputs = Input(shape=(max_answer_len,))
decoder_embedding = Embedding(vocab_size, embedding_dim)(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = Dense(vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Mô hình Seq2Seq
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Biên dịch mô hình
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Chuẩn bị dữ liệu đầu vào và đầu ra cho bộ giải mã
decoder_target_sequences = np.zeros((len(answers), max_answer_len), dtype="int32")
for i, seq in enumerate(answer_sequences):
    if len(seq) > 1:
        decoder_target_sequences[i, :len(seq)-1] = seq[1:]

# Huấn luyện mô hình
model.fit([question_sequences, answer_sequences], decoder_target_sequences, batch_size=64, epochs=100)

# Tạo mô hình dự đoán cho bộ mã hóa
encoder_model = Model(encoder_inputs, encoder_states)

# Tạo mô hình dự đoán cho bộ giải mã
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_embedding, initial_state=decoder_states_inputs
)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states
)

# Hàm dự đoán câu trả lời
def decode_sequence(input_seq):
    # Mã hóa câu hỏi
    states_value = encoder_model.predict(input_seq)
    
    # Khởi tạo chuỗi đầu vào của bộ giải mã với token "<start>"
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = word_index['<start>']
    
    # Khởi tạo câu trả lời
    stop_condition = False
    decoded_sentence = ""
    
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        
        # Chọn từ có xác suất cao nhất
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = index_word.get(sampled_token_index, '')
        decoded_sentence += ' ' + sampled_word
        
        # Điều kiện dừng
        if sampled_word == '<end>' or len(decoded_sentence.split()) > max_answer_len:
            stop_condition = True
        
        # Cập nhật chuỗi đầu vào của bộ giải mã
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        
        # Cập nhật trạng thái
        states_value = [h, c]
    
    return decoded_sentence.replace('<end>', '').strip()

# Hàm để tiền xử lý câu hỏi của người dùng
def preprocess_input(text):
    seq = tokenizer.texts_to_sequences([text])
    padded_seq = pad_sequences(seq, maxlen=max_question_len, padding='post')
    return padded_seq

# Hàm xử lý yêu cầu và phản hồi
def chatbot_response(user_input):
    # Tiền xử lý câu hỏi của người dùng
    input_seq = preprocess_input(user_input)
    
    # Giải mã câu trả lời từ mô hình Seq2Seq
    decoded_response = decode_sequence(input_seq)
    
    return decoded_response

# Đặt câu hỏi và nhận câu trả lời
user_input = "Tell me a joke"
response = chatbot_response(user_input)
print("Bot:", response)
